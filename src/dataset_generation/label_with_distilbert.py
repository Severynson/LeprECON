"""Label articles using fine-tuned DistilBERT.

Loads a trained DistilBERT checkpoint and applies it to unlabeled articles.
Seeds the cache with Gemini labels first to preserve ground truth.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from src.dataset_generation.fine_tune_distilbert import (
    build_dedup_key,
    build_feature_text,
    load_label_cache,
    normalize_text,
    read_tsv,
)


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[label_with_distilbert {ts}] {message}", flush=True)


def load_config(config_path: str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config (expected YAML mapping): {config_path}")
    return cfg


def append_to_bert_cache(cache_path: str, entries: list[dict[str, Any]]) -> None:
    """Append entries to BERT label cache."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def seed_bert_cache_from_gemini(
    bert_cache_path: str,
    gemini_cache_path: str,
) -> int:
    """Seed BERT cache with Gemini labels. Returns count of entries added."""
    gemini_cache = load_label_cache(gemini_cache_path)

    # Load any existing BERT cache to avoid duplicates
    bert_p = Path(bert_cache_path)
    bert_cache_keys: set[str] = set()
    if bert_p.exists():
        with bert_p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        bert_cache_keys.add(entry.get("content_key", ""))
                    except json.JSONDecodeError:
                        pass

    # Seed with Gemini entries not yet in BERT cache
    entries_to_add = [
        {
            "content_key": ck,
            "label": entry["label"],
            "reason": entry.get("reason", ""),
            "model": "gemini-seeded",
            "confidence": entry.get("confidence", ""),
            "label_created_at_utc": entry.get("label_created_at_utc", ""),
        }
        for ck, entry in gemini_cache.items()
        if ck not in bert_cache_keys
    ]

    if entries_to_add:
        append_to_bert_cache(bert_cache_path, entries_to_add)
        log(f"Seeded BERT cache with {len(entries_to_add)} Gemini labels")

    return len(entries_to_add)


def run_labeling(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply fine-tuned DistilBERT to label unlabeled articles."""
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    artifact_cfg = cfg.get("artifacts", {})

    raw_path = data_cfg.get("raw_articles_path", "data/news_raw/new_york_times.tsv")
    gemini_cache_path = data_cfg.get("gemini_cache_path", "data/news_preprocessed/gemini_label_cache.jsonl")
    bert_cache_path = data_cfg.get("bert_cache_path", "data/news_preprocessed/BERT_label_cache.jsonl")
    max_text_chars = data_cfg.get("max_text_chars", 1000)

    model_dir = model_cfg.get("model_dir", "artifacts/models/distilbert_relevance")
    max_length = model_cfg.get("max_length", 512)

    report_path = artifact_cfg.get("report_path", "artifacts/reports/distilbert_inference_report.json")

    # Use CPU by default on macOS to avoid MPS mutex locks
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log(f"Using device: {device}")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Seed BERT cache from Gemini labels
    log(f"Seeding BERT cache from Gemini labels...")
    seeded_count = seed_bert_cache_from_gemini(bert_cache_path, gemini_cache_path)

    # 2. Load BERT cache (now includes seeded Gemini labels)
    bert_cache = load_label_cache(bert_cache_path)
    log(f"BERT cache now has {len(bert_cache)} entries")

    # 3. Load raw articles
    log(f"Loading raw articles from {raw_path}")
    articles = read_tsv(raw_path)
    log(f"Loaded {len(articles)} articles")

    # 4. Build article dict by content_key
    article_by_key: dict[str, dict[str, str]] = {}
    for article in articles:
        key = build_dedup_key(article)
        if key:
            article_by_key[key] = article

    # 5. Identify unlabeled articles
    unlabeled_keys = [k for k in article_by_key.keys() if k not in bert_cache]
    log(f"Found {len(unlabeled_keys)} unlabeled articles")

    if not unlabeled_keys:
        log("All articles already labeled!")
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "articles_processed": 0,
            "articles_labeled": 0,
            "articles_cached": len(bert_cache),
            "seeded_from_gemini": seeded_count,
        }

    # 6. Load fine-tuned model
    log(f"Loading fine-tuned DistilBERT from {model_dir}")
    from src.models.DistilBERT import load_model as load_distilbert

    model, tokenizer = load_distilbert(
        num_labels=2,
        pretrained=model_dir,
        device=device,
    )

    # 7. Label unlabeled articles in batches
    batch_size = 32
    articles_labeled = 0
    new_cache_entries = []
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for batch_start in range(0, len(unlabeled_keys), batch_size):
        batch_keys = unlabeled_keys[batch_start : batch_start + batch_size]
        batch_texts = [
            build_feature_text(article_by_key[k], max_chars=max_text_chars)
            for k in batch_keys
        ]

        # Tokenize and encode
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = torch.softmax(logits, dim=1).cpu().numpy()

        # Save predictions
        for key, pred, conf in zip(batch_keys, preds, confidences):
            label = int(pred)
            confidence = float(conf[label])

            entry = {
                "content_key": key,
                "label": label,
                "reason": "distilbert-inference",
                "model": "distilbert-fine-tuned",
                "confidence": confidence,
                "label_created_at_utc": now_utc,
            }
            new_cache_entries.append(entry)
            bert_cache[key] = entry
            articles_labeled += 1

            if articles_labeled % 500 == 0:
                log(f"Labeled {articles_labeled}/{len(unlabeled_keys)} articles")

        # Flush to disk
        if len(new_cache_entries) >= 100 or batch_start + batch_size >= len(unlabeled_keys):
            append_to_bert_cache(bert_cache_path, new_cache_entries)
            new_cache_entries = []

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Count labels in final cache
    label_0_count = sum(1 for e in bert_cache.values() if e["label"] == 0)
    label_1_count = sum(1 for e in bert_cache.values() if e["label"] == 1)

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "model_checkpoint": model_dir,
        "articles_processed": len(unlabeled_keys),
        "articles_labeled": articles_labeled,
        "articles_cached": len(bert_cache),
        "seeded_from_gemini": seeded_count,
        "label_distribution": {
            "label_0": label_0_count,
            "label_1": label_1_count,
        },
        "cache_path": bert_cache_path,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log(f"Report written to {report_path}")
    log(f"Labeled {articles_labeled} articles, cache now has {len(bert_cache)} total entries")
    log(f"Distribution: {label_0_count} label=0, {label_1_count} label=1")

    return report


def main() -> int:
    # Add project root to sys.path so we can import src package
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default_distilbert_inference.yaml"
    log(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    run_labeling(cfg)
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
