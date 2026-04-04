"""Preprocessing pipeline for NYT article data.

Reads raw TSV, deduplicates, sorts, labels via Gemini, writes clean output.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Shared utilities (duplicated from article_pull.py because the parent
# directory name contains a space, making normal Python imports impossible)
# ---------------------------------------------------------------------------

def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def build_dedup_key(row: dict[str, str]) -> str | None:
    article_id = normalize_text(row.get("article_id"))
    if article_id:
        return f"id:{article_id}"

    web_url = normalize_text(row.get("web_url"))
    if web_url:
        return f"url:{web_url}"

    published_at = normalize_text(row.get("published_at"))
    headline = normalize_text(row.get("headline"))
    if published_at and headline:
        return f"published_headline:{published_at}|{headline}"

    article_day = normalize_text(row.get("article_day"))
    if article_day and headline:
        return f"day_headline:{article_day}|{headline}"

    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_FIELDNAMES = [
    "article_id", "source", "query", "article_day", "published_at",
    "headline", "article_text", "snippet", "abstract", "lead_paragraph",
    "section_name", "subsection_name", "web_url", "keywords",
]

LABEL_COLUMNS = [
    "economy_relevance_label",
    "economy_relevance_reason",
    "economy_relevance_model",
    "economy_relevance_confidence",
    "label_created_at_utc",
]

OUTPUT_FIELDNAMES = INPUT_FIELDNAMES + LABEL_COLUMNS + ["is_interpolated_no_news"]

DEFAULT_CONFIG_PATH = "configs/default_preprocessing.yaml"


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[preprocessing {ts}] {message}", flush=True)


# ---------------------------------------------------------------------------
# 1. Read
# ---------------------------------------------------------------------------

def read_tsv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


# ---------------------------------------------------------------------------
# 2. Deduplicate
# ---------------------------------------------------------------------------

def deduplicate(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return (unique_rows, removals_by_key_type)."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    removals: dict[str, int] = {
        "id": 0,
        "url": 0,
        "published_headline": 0,
        "day_headline": 0,
    }

    for row in rows:
        key = build_dedup_key(row)
        if key is None:
            unique.append(row)
            continue
        if key in seen:
            key_type = key.split(":")[0]
            removals[key_type] = removals.get(key_type, 0) + 1
            continue
        seen.add(key)
        unique.append(row)

    return unique, removals


# ---------------------------------------------------------------------------
# 3. Sort
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_DATE = "9999-99-99"


def _sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    article_day = row.get("article_day", "")
    if not _DATE_RE.match(article_day):
        article_day = _MAX_DATE

    published_at = row.get("published_at", "") or ""
    tiebreaker = row.get("article_id", "") or row.get("web_url", "") or ""
    return (article_day, published_at, tiebreaker)


def sort_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    invalid_date_count = sum(
        1 for r in rows if not _DATE_RE.match(r.get("article_day", ""))
    )
    sorted_rows = sorted(rows, key=_sort_key)
    return sorted_rows, invalid_date_count


# ---------------------------------------------------------------------------
# 4. Gemini labeling
# ---------------------------------------------------------------------------

LABEL_PROMPT_TEMPLATE = """\
You are an economic relevance classifier.

For EACH article below, decide:
  label=1  if the article is likely relevant to the global economy, financial \
markets, macroeconomics, business, technology sector, politics with economic \
implications, or major world events that could affect the S&P 500.
  label=0  if the article is unlikely to impact the global economy or S&P 500 \
(e.g. arts, movies, lifestyle, sports, obituaries, recipes, travel, fashion).

Be conservative: only output 1 when a plausible macro / market linkage exists.

Return ONLY a JSON array (no markdown fences) with one object per article:
[
  {{"id": <int>, "label": <0 or 1>, "reason": "<max 20 words>"}},
  ...
]

Articles:
{articles_block}
"""


def _build_article_text_for_label(row: dict[str, str], max_chars: int) -> str:
    headline = row.get("headline", "")
    section = row.get("section_name", "")
    subsection = row.get("subsection_name", "")
    keywords = row.get("keywords", "")
    body = row.get("article_text", "")[:max_chars]

    parts = [f"Headline: {headline}"]
    if section:
        parts.append(f"Section: {section}")
    if subsection:
        parts.append(f"Subsection: {subsection}")
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if body:
        parts.append(f"Text: {body}")
    return "\n".join(parts)


def _content_key(row: dict[str, str]) -> str:
    """Stable key for caching label results."""
    key = build_dedup_key(row)
    if key:
        return key
    return f"content:{normalize_text(row.get('headline', ''))}|{normalize_text(row.get('article_text', '')[:200])}"


# -- Cache -----------------------------------------------------------------

def load_label_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    p = Path(cache_path)
    if not p.exists():
        return cache
    with p.open("r", encoding="utf-8") as fh:
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


def append_cache_entries(cache_path: str, entries: list[dict[str, Any]]) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# -- Gemini API wrapper ----------------------------------------------------

def _call_gemini(
    prompt: str,
    *,
    api_key: str,
    model: str,
) -> str:
    """Call Gemini REST API and return generated text."""
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    resp = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _parse_label_response(raw: str, expected_count: int) -> list[dict[str, Any]] | None:
    """Parse the JSON array from Gemini's response. Returns None on failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            return None
        if "id" not in item or "label" not in item:
            return None
        if item["label"] not in (0, 1):
            return None

    return items


def label_batch(
    batch: list[tuple[int, dict[str, str]]],
    *,
    api_key: str,
    model: str,
    max_text_chars: int,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Label a batch of (local_idx, row) tuples. Returns parsed label dicts."""
    articles_block = ""
    for i, (_, row) in enumerate(batch):
        text = _build_article_text_for_label(row, max_text_chars)
        articles_block += f"\n--- Article {i} ---\n{text}\n"

    prompt = LABEL_PROMPT_TEMPLATE.format(articles_block=articles_block)

    for attempt in range(max_retries):
        try:
            raw = _call_gemini(prompt, api_key=api_key, model=model)
            parsed = _parse_label_response(raw, len(batch))
            if parsed is not None and len(parsed) == len(batch):
                return parsed
            log(f"Malformed Gemini response (attempt {attempt + 1}), retrying with smaller context")
        except Exception as exc:
            log(f"Gemini API error (attempt {attempt + 1}): {exc}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    # Fallback: return label=0 for all
    return [
        {"id": i, "label": 0, "reason": "labeling_failed"}
        for i in range(len(batch))
    ]


def run_gemini_labeling(
    rows: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    batch_size: int,
    max_text_chars: int,
    cache_path: str,
) -> tuple[list[dict[str, str]], int, int, int]:
    """Attach label columns to every row.

    Returns (labeled_rows, cached_count, api_count, failure_count).
    """
    cache = load_label_cache(cache_path)
    cached_count = 0
    api_count = 0
    failure_count = 0

    # Pre-assign labels from cache
    unlabeled_indices: list[int] = []
    for i, row in enumerate(rows):
        ck = _content_key(row)
        if ck in cache:
            entry = cache[ck]
            row["economy_relevance_label"] = str(entry.get("label", 0))
            row["economy_relevance_reason"] = entry.get("reason", "")
            row["economy_relevance_model"] = entry.get("model", model)
            row["economy_relevance_confidence"] = str(entry.get("confidence", ""))
            row["label_created_at_utc"] = entry.get("label_created_at_utc", "")
            cached_count += 1
        else:
            unlabeled_indices.append(i)

    log(f"Labels from cache: {cached_count}, to label via API: {len(unlabeled_indices)}")

    # Process unlabeled in batches
    for batch_start in range(0, len(unlabeled_indices), batch_size):
        batch_idx = unlabeled_indices[batch_start:batch_start + batch_size]
        batch = [(idx, rows[idx]) for idx in batch_idx]

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        results = label_batch(
            batch, api_key=api_key, model=model, max_text_chars=max_text_chars,
        )

        new_cache_entries: list[dict[str, Any]] = []
        for (global_idx, row), result in zip(batch, results):
            is_failure = result.get("reason") == "labeling_failed"
            if is_failure:
                failure_count += 1
            else:
                api_count += 1

            row["economy_relevance_label"] = str(result["label"])
            row["economy_relevance_reason"] = result.get("reason", "")
            row["economy_relevance_model"] = model
            row["economy_relevance_confidence"] = str(result.get("confidence", ""))
            row["label_created_at_utc"] = now_utc

            ck = _content_key(row)
            entry = {
                "content_key": ck,
                "label": result["label"],
                "reason": result.get("reason", ""),
                "model": model,
                "confidence": result.get("confidence", ""),
                "label_created_at_utc": now_utc,
            }
            cache[ck] = entry
            new_cache_entries.append(entry)

        append_cache_entries(cache_path, new_cache_entries)

        done = min(batch_start + batch_size, len(unlabeled_indices))
        log(f"Labeled {done}/{len(unlabeled_indices)} rows")

    return rows, cached_count, api_count, failure_count


# ---------------------------------------------------------------------------
# 5. Interpolate missing relevant-news days
# ---------------------------------------------------------------------------

NO_NEWS_PLACEHOLDER = "[no news]"


def _make_interpolation_row(day_str: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for col in INPUT_FIELDNAMES:
        row[col] = NO_NEWS_PLACEHOLDER
    row["article_day"] = day_str
    row["published_at"] = NO_NEWS_PLACEHOLDER
    row["article_id"] = ""
    row["source"] = ""
    row["query"] = ""
    row["economy_relevance_label"] = "0"
    row["economy_relevance_reason"] = "interpolated_no_relevant_news"
    row["economy_relevance_model"] = ""
    row["economy_relevance_confidence"] = ""
    row["label_created_at_utc"] = ""
    row["is_interpolated_no_news"] = "1"
    return row


def interpolate_missing_days(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int, int]:
    """Add synthetic rows for calendar days without relevant (label=1) news.

    Returns (rows_with_interpolation, missing_days_count, synthetic_rows_added).
    """
    from datetime import date, timedelta

    # Collect valid dates and which ones have relevant news
    valid_dates: list[date] = []
    days_with_relevant: set[str] = set()

    for row in rows:
        day = row.get("article_day", "")
        if not _DATE_RE.match(day):
            continue
        valid_dates.append(date.fromisoformat(day))
        if row.get("economy_relevance_label") == "1":
            days_with_relevant.add(day)

    if not valid_dates:
        return rows, 0, 0

    min_day = min(valid_dates)
    max_day = max(valid_dates)

    # Build set of all days that have any rows
    days_with_rows: set[str] = set()
    for row in rows:
        day = row.get("article_day", "")
        if _DATE_RE.match(day):
            days_with_rows.add(day)

    # Walk the full calendar range and find days needing interpolation
    missing_days: list[str] = []
    current = min_day
    while current <= max_day:
        day_str = current.isoformat()
        if day_str not in days_with_relevant:
            missing_days.append(day_str)
        current += timedelta(days=1)

    if not missing_days:
        # Mark all existing rows as not interpolated
        for row in rows:
            row.setdefault("is_interpolated_no_news", "0")
        return rows, 0, 0

    # Mark existing rows
    for row in rows:
        row.setdefault("is_interpolated_no_news", "0")

    # Build synthetic rows and merge into sorted position
    synthetic: list[dict[str, str]] = []
    for day_str in missing_days:
        synthetic.append(_make_interpolation_row(day_str))

    combined = rows + synthetic
    combined.sort(key=_sort_key)

    return combined, len(missing_days), len(synthetic)


# ---------------------------------------------------------------------------
# 6. Write output
# ---------------------------------------------------------------------------

def write_output_tsv(rows: list[dict[str, str]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: str, report: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Config loading + main
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load preprocessing config from a YAML file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file (expected YAML mapping): {config_path}")
    return cfg


def load_env_value(key: str, env_path: str = ".env") -> str | None:
    direct = os.environ.get(key)
    if direct:
        return direct
    env_file = Path(env_path)
    if not env_file.exists():
        return None
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        env_key, env_value = m.groups()
        if env_key != key:
            continue
        val = env_value.strip()
        if val and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        return val
    return None


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the full preprocessing pipeline driven by *cfg* (parsed YAML)."""
    run_cfg = cfg.get("run", {})
    paths_cfg = cfg.get("paths", {})
    gemini_cfg = cfg.get("gemini", {})

    dry_run: bool = run_cfg.get("dry_run", False)
    limit: int = run_cfg.get("limit", 0)

    input_path: str = paths_cfg.get("input", "data/news_raw/new_york_times.tsv")
    output_path: str = paths_cfg.get("output", "data/news_processed/new_york_times_preprocessed.tsv")
    report_path: str = paths_cfg.get("report", "data/news_processed/preprocessing_report.json")
    cache_path: str = paths_cfg.get("label_cache", "data/news_processed/gemini_label_cache.jsonl")
    env_path: str = paths_cfg.get("env_file", ".env")

    model: str = gemini_cfg.get("model", "gemini-2.0-flash")
    batch_size: int = gemini_cfg.get("batch_size", 25)
    max_text_chars: int = gemini_cfg.get("max_text_chars", 2000)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Read
    log(f"Reading {input_path}")
    rows = read_tsv(input_path)
    input_rows = len(rows)
    log(f"Read {input_rows} rows")

    # 2. Deduplicate
    rows, removals = deduplicate(rows)
    dups_total = sum(removals.values())
    log(f"After dedup: {len(rows)} rows ({dups_total} duplicates removed)")

    # 3. Sort
    rows, invalid_dates = sort_rows(rows)
    log(f"Sorted. Invalid article_day count: {invalid_dates}")

    # Apply limit
    if limit > 0:
        rows = rows[:limit]
        log(f"Limited to {len(rows)} rows")

    # 4-7. Gemini labeling
    cached_count = 0
    api_count = 0
    failure_count = 0

    if dry_run:
        log("Dry-run mode: skipping Gemini labeling")
        for row in rows:
            row["economy_relevance_label"] = ""
            row["economy_relevance_reason"] = ""
            row["economy_relevance_model"] = ""
            row["economy_relevance_confidence"] = ""
            row["label_created_at_utc"] = ""
    else:
        api_key = load_env_value("GEMINI_API_KEY", env_path=env_path)
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in environment or .env file. "
                "Set dry_run: true in config to skip labeling."
            )
        rows, cached_count, api_count, failure_count = run_gemini_labeling(
            rows,
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_text_chars=max_text_chars,
            cache_path=cache_path,
        )

    # 8. Interpolate missing relevant-news days
    if not dry_run:
        rows, missing_days, synthetic_added = interpolate_missing_days(rows)
        log(f"Interpolation: {missing_days} days without relevant news, {synthetic_added} synthetic rows added")
    else:
        missing_days = 0
        synthetic_added = 0
        for row in rows:
            row.setdefault("is_interpolated_no_news", "0")

    # 9. Write output
    write_output_tsv(rows, output_path)
    log(f"Wrote {len(rows)} rows to {output_path}")

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 10. Report
    report = {
        "input_rows": input_rows,
        "rows_after_dedup": input_rows - dups_total,
        "duplicates_removed_total": dups_total,
        "duplicates_removed_by_key_type": removals,
        "rows_with_invalid_article_day": invalid_dates,
        "rows_labeled_from_cache": cached_count,
        "rows_labeled_from_api": api_count,
        "rows_label_failures": failure_count,
        "missing_relevant_news_days_interpolated": missing_days,
        "synthetic_no_news_rows_added": synthetic_added,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }
    write_report(report_path, report)
    log(f"Report written to {report_path}")

    return report


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    log(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    run_pipeline(cfg)
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
