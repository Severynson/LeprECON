"""Fine-tune DistilBERT for binary economy-relevance classification.

Loads Gemini labels from cache, joins with raw articles, balances classes,
trains DistilBERT with chronological 80/20 split.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

# Set env BEFORE any transformers imports to avoid cache lock issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Use temp directory cache to avoid lock contention with ~/.cache
_tmp_cache = Path(tempfile.gettempdir()) / "distilbert_cache"
_tmp_cache.mkdir(exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(_tmp_cache)
os.environ["TOKENIZERS_CACHE"] = str(_tmp_cache / "tokenizers")


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[fine_tune_distilbert {ts}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Reuse dedup key logic from preprocessing.py (via importlib)
# ---------------------------------------------------------------------------

def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def build_dedup_key(row: dict[str, str]) -> str | None:
    """Build content_key using same priority as preprocessing.py."""
    article_id = normalize_text(row.get("article_id", ""))
    if article_id:
        return f"id:{article_id}"

    web_url = normalize_text(row.get("web_url", ""))
    if web_url:
        return f"url:{web_url}"

    published_at = normalize_text(row.get("published_at", ""))
    headline = normalize_text(row.get("headline", ""))
    if published_at and headline:
        return f"published_headline:{published_at}|{headline}"

    article_day = normalize_text(row.get("article_day", ""))
    if article_day and headline:
        return f"day_headline:{article_day}|{headline}"

    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_tsv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def load_label_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    """Load Gemini label cache as dict[content_key -> entry]."""
    cache: dict[str, dict[str, Any]] = {}
    with open(cache_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["content_key"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def load_config(config_path: str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config (expected YAML mapping): {config_path}")
    return cfg


def load_distilbert_module_loader() -> Any:
    """Load DistilBERT loader without relying on `src` package imports."""
    module_path = Path(__file__).resolve().parents[1] / "models" / "DistilBERT.py"
    spec = importlib.util.spec_from_file_location("distilbert_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model


def resolve_device(use_mps: bool = False) -> str:
    """Resolve device with MPS disabled by default on macOS (mutex lock issues).

    Args:
        use_mps: If True, try to use Metal Performance Shaders on macOS.
                 Default False to avoid known pthread/mutex deadlocks.
    """
    if torch.cuda.is_available():
        return "cuda"
    if use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_feature_text(
    row: dict[str, str],
    max_chars: int = 1000,
) -> str:
    """Build feature text from headline, article_text, snippet, etc.

    Args:
        row: Article row dict.
        max_chars: Max chars to include from article_text.

    Returns:
        Concatenated text with [SEP] tokens.
    """
    headline = normalize_text(row.get("headline", ""))
    article_text = normalize_text(row.get("article_text", ""))
    snippet = normalize_text(row.get("snippet", ""))
    abstract = normalize_text(row.get("abstract", ""))
    lead_paragraph = normalize_text(row.get("lead_paragraph", ""))
    section_name = normalize_text(row.get("section_name", ""))

    article_text = article_text[:max_chars]

    parts = []
    for part in [headline, article_text, snippet, abstract, lead_paragraph, section_name]:
        if part:
            parts.append(part)

    return " [SEP] ".join(parts)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArticleDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data balancing
# ---------------------------------------------------------------------------

def balance_dataset(
    texts: list[str],
    labels: list[int],
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Undersample majority class to match minority class.

    Args:
        texts: Feature texts.
        labels: Binary labels (0/1).
        seed: Random seed.

    Returns:
        (texts, labels) tuple with balanced distribution.
    """
    np.random.seed(seed)
    label_arr = np.array(labels)

    count_0 = np.sum(label_arr == 0)
    count_1 = np.sum(label_arr == 1)

    if count_0 == count_1:
        return texts, labels

    if count_0 < count_1:
        minority_label = 0
        majority_label = 1
        minority_count = count_0
    else:
        minority_label = 1
        majority_label = 0
        minority_count = count_1

    minority_idx = np.where(label_arr == minority_label)[0]
    majority_idx = np.where(label_arr == majority_label)[0]

    sampled_majority_idx = np.random.choice(
        majority_idx,
        size=minority_count,
        replace=False,
    )

    keep_idx = np.concatenate([minority_idx, sampled_majority_idx])
    keep_idx.sort()

    balanced_texts = [texts[i] for i in keep_idx]
    balanced_labels = [labels[i] for i in keep_idx]

    return balanced_texts, balanced_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: Any,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    device: str,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(
    model: Any,
    dataloader: DataLoader,
    device: str,
) -> tuple[float, float]:
    """Evaluate one epoch. Returns (accuracy, f1)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return accuracy, f1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    artifact_cfg = cfg.get("artifacts", {})

    seed = run_cfg.get("seed", 42)
    limit = run_cfg.get("limit", 0)
    use_mps = run_cfg.get("use_mps", False)  # Disabled by default (mutex issues on macOS)
    np.random.seed(seed)
    torch.manual_seed(seed)

    raw_path = data_cfg.get("raw_articles_path", "data/news_raw/new_york_times.tsv")
    cache_path = data_cfg.get("label_cache_path", "data/news_preprocessed/gemini_label_cache.jsonl")
    exclude_failures = data_cfg.get("exclude_labeling_failures", True)
    train_ratio = data_cfg.get("train_ratio", 0.8)
    max_text_chars = data_cfg.get("max_text_chars", 1000)

    model_name = model_cfg.get("pretrained", "distilbert-base-uncased")
    num_labels = model_cfg.get("num_labels", 2)
    max_length = model_cfg.get("max_length", 512)
    use_fast_tokenizer = model_cfg.get("use_fast_tokenizer", False)
    local_files_only = model_cfg.get("local_files_only", False)

    epochs = train_cfg.get("epochs", 3)
    batch_size = train_cfg.get("batch_size", 32)
    dataloader_workers = train_cfg.get("dataloader_workers", 0)
    lr = train_cfg.get("learning_rate", 2.0e-5)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)

    model_dir = artifact_cfg.get("model_dir", "artifacts/models/distilbert_relevance")
    report_path = artifact_cfg.get("report_path", "artifacts/reports/distilbert_finetune_report.json")
    env_path = artifact_cfg.get("env_file", ".env")
    hf_cache_dir = artifact_cfg.get("hf_cache_dir", "artifacts/huggingface_cache")

    device = resolve_device(use_mps=use_mps)
    log(f"Using device: {device}")
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(Path(hf_cache_dir).resolve())
    os.environ["HF_HUB_OFFLINE"] = "0"  # Allow downloads once
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    log(f"Using HF cache dir: {hf_cache_dir}")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Load labels
    log(f"Loading label cache from {cache_path}")
    label_cache = load_label_cache(cache_path)
    total_cached = len(label_cache)

    if exclude_failures:
        label_cache = {
            k: v for k, v in label_cache.items()
            if v.get("reason") != "labeling_failed"
        }
        filtered_count = total_cached - len(label_cache)
        log(f"Excluded {filtered_count} labeling_failed entries")

    # 2. Load articles
    log(f"Loading raw articles from {raw_path}")
    articles = read_tsv(raw_path)
    log(f"Loaded {len(articles)} articles")

    # 3. Build article dict keyed by content_key
    article_by_key: dict[str, dict[str, str]] = {}
    for article in articles:
        key = build_dedup_key(article)
        if key:
            article_by_key[key] = article

    # 4. Intersect labels with articles
    joined: list[tuple[str, int]] = []
    for content_key, label_entry in label_cache.items():
        if content_key in article_by_key:
            label = int(label_entry.get("label", 0))
            joined.append((content_key, label))

    log(f"Joined {len(joined)} labels with articles (dropped {len(label_cache) - len(joined)} unmatched)")

    if not joined:
        raise RuntimeError("No matching label-article pairs found!")

    # Apply limit to training data (after join)
    if limit > 0:
        joined = joined[:limit]
        log(f"Limited to {len(joined)} samples")

    # 5. Build texts and labels
    texts = []
    labels_list = []
    for content_key, label in joined:
        article = article_by_key[content_key]
        text = build_feature_text(article, max_chars=max_text_chars)
        texts.append(text)
        labels_list.append(label)

    # Count before balancing
    count_0_before = sum(1 for l in labels_list if l == 0)
    count_1_before = sum(1 for l in labels_list if l == 1)
    log(f"Before balancing: {count_0_before} label=0, {count_1_before} label=1")

    # 6. Balance
    texts, labels_list = balance_dataset(texts, labels_list, seed=seed)
    count_0_after = sum(1 for l in labels_list if l == 0)
    count_1_after = sum(1 for l in labels_list if l == 1)
    log(f"After balancing: {count_0_after} label=0, {count_1_after} label=1")

    # 7. Sort by article_day for chronological split
    text_with_day = []
    for content_key, label in zip([ck for ck, _ in joined], labels_list):
        if content_key in article_by_key:
            day = article_by_key[content_key].get("article_day", "")
            text_with_day.append((texts[len(text_with_day)], label, day))

    text_with_day.sort(key=lambda x: x[2])  # Sort by day
    texts_sorted = [t for t, _, _ in text_with_day]
    labels_sorted = [l for _, l, _ in text_with_day]

    # 8. Chronological 80/20 split
    split_idx = int(len(texts_sorted) * train_ratio)
    train_texts = texts_sorted[:split_idx]
    train_labels = labels_sorted[:split_idx]
    val_texts = texts_sorted[split_idx:]
    val_labels = labels_sorted[split_idx:]

    log(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # 9. Load model and tokenizer
    log(f"Loading {model_name}")
    load_distilbert = load_distilbert_module_loader()

    model, tokenizer = load_distilbert(
        num_labels=num_labels,
        pretrained=model_name,
        device=device,
        use_fast_tokenizer=use_fast_tokenizer,
        local_files_only=local_files_only,
    )
    log("Model and tokenizer loaded")

    # 10. Create datasets and dataloaders
    train_dataset = ArticleDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ArticleDataset(val_texts, val_labels, tokenizer, max_length)

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=pin_memory,
    )

    # 11. Optimizer and scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 12. Training loop
    best_f1 = 0.0
    best_checkpoint = None

    metrics_per_epoch = []
    for epoch in range(epochs):
        log(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_f1 = eval_epoch(model, val_loader, device)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
        }
        metrics_per_epoch.append(metrics)

        log(f"  train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_checkpoint = epoch
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            log(f"  Saved best checkpoint (F1={val_f1:.4f})")

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 13. Report
    report = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "model": model_name,
        "seed": seed,
        "dataset": {
            "total_labels_cached": total_cached,
            "after_exclude_failures": len(label_cache) if exclude_failures else total_cached,
            "matched_with_articles": len(joined),
            "before_balancing": {"label_0": count_0_before, "label_1": count_1_before},
            "after_balancing": {"label_0": count_0_after, "label_1": count_1_after},
            "train_size": len(train_texts),
            "val_size": len(val_texts),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "total_training_steps": total_steps,
            "warmup_steps": warmup_steps,
        },
        "metrics_per_epoch": metrics_per_epoch,
        "best_checkpoint_epoch": best_checkpoint,
        "best_f1": best_f1,
        "model_dir": model_dir,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log(f"Report written to {report_path}")
    log(f"Best checkpoint saved at {model_dir} (epoch {best_checkpoint + 1}, F1={best_f1:.4f})")

    return report


def main() -> int:
    # Add project root to sys.path so we can import src package
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default_distilbert_finetune.yaml"
    log(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    run_pipeline(cfg)
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
