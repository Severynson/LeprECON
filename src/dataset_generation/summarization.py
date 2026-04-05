"""Daily news summarization using Gemini API.

Pipeline:
1. Load BERT label cache and raw articles.
2. Keep only articles with label=1 (relevant).
3. Group by article_day.
4. Generate one daily summary per day via Gemini.
5. Append results to JSONL output with resume support.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[summarization {ts}] {message}", flush=True)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def normalize_date_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return normalize_text(str(value))


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
        value = env_value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return None


def load_config(config_path: str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config (expected YAML mapping): {config_path}")
    return cfg


def read_tsv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def read_sp500(path: str) -> dict[str, str]:
    prices: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            day = normalize_text(row.get("observation_date"))
            price = normalize_text(row.get("SP500"))
            if day and price:
                prices[day] = price
    return prices


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
            except json.JSONDecodeError:
                continue
            content_key = entry.get("content_key")
            if not content_key:
                continue
            cache[content_key] = entry
    return cache


def load_completed_days(output_path: str) -> set[str]:
    p = Path(output_path)
    if not p.exists():
        return set()

    completed: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            day = payload.get("article_day")
            if day:
                completed.add(day)
    return completed


def append_summary_rows(output_path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_daily_summaries(output_path: str) -> dict[str, str]:
    """Load day->summary from jsonl output (last duplicate wins)."""
    summaries: dict[str, str] = {}
    p = Path(output_path)
    if not p.exists():
        return summaries
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            day = normalize_text(payload.get("article_day", ""))
            summary = normalize_text(payload.get("summary", ""))
            if day and summary:
                summaries[day] = summary
    return summaries


def write_daily_summary_and_price_csv(
    *,
    summary_by_day: dict[str, str],
    sp500_by_day: dict[str, str],
    output_csv_path: str,
) -> tuple[int, int]:
    """Write date/SP500/summary CSV. Returns (rows_written, missing_price_days)."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    missing_price_days = 0
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["date", "sp500_close", "summary"],
        )
        writer.writeheader()
        for day in sorted(summary_by_day.keys()):
            if day not in sp500_by_day:
                missing_price_days += 1
                continue
            writer.writerow(
                {
                    "date": day,
                    "sp500_close": sp500_by_day[day],
                    "summary": summary_by_day[day],
                }
            )
            rows_written += 1
    return rows_written, missing_price_days


def _article_to_block(row: dict[str, str], max_chars_per_article: int) -> str:
    headline = normalize_text(row.get("headline"))
    article_text = normalize_text(row.get("article_text"))
    snippet = normalize_text(row.get("snippet"))
    abstract = normalize_text(row.get("abstract"))
    lead = normalize_text(row.get("lead_paragraph"))
    section = normalize_text(row.get("section_name"))
    subsection = normalize_text(row.get("subsection_name"))
    source = normalize_text(row.get("source"))
    published_at = normalize_text(row.get("published_at"))
    url = normalize_text(row.get("web_url"))

    merged = " ".join(x for x in [article_text, snippet, abstract, lead] if x)
    merged = merged[:max_chars_per_article]

    parts = []
    if published_at:
        parts.append(f"Published: {published_at}")
    if source:
        parts.append(f"Source: {source}")
    if section:
        parts.append(f"Section: {section}")
    if subsection:
        parts.append(f"Subsection: {subsection}")
    if headline:
        parts.append(f"Headline: {headline}")
    if merged:
        parts.append(f"Text: {merged}")
    if url:
        parts.append(f"URL: {url}")

    return "\n".join(parts)


def _build_prompt(day: str, article_blocks: list[str], max_tokens: int) -> str:
    articles_blob = ""
    for idx, block in enumerate(article_blocks):
        articles_blob += f"\n=== ARTICLE {idx + 1} ===\n{block}\n"

    return (
        f"You are a neutral news summarizer writing as of {day}.\n"
        f"Produce ONLY a factual bullet-point summary of all input articles combined.\n"
        f"Max length: {max_tokens} tokens.\n"
        "Focus on reported events, names, dates, and descriptions.\n"
        "No predictions, opinions, or macro forecasts.\n"
        "For each named person/org, add a short parenthetical definition from provided facts only.\n"
        "Structure:\n"
        "- One bullet per major event cluster.\n"
        "- Sub-bullets for key details.\n"
        "Return plain text bullets only.\n"
        f"{articles_blob}"
    )


def call_gemini_summary(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int,
    timeout_seconds: int,
    backoff_seconds: float,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=timeout_seconds,
            )
            if resp.status_code < 400:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()

            should_retry = resp.status_code == 429 or resp.status_code >= 500
            if not should_retry or attempt == max_retries:
                resp.raise_for_status()

            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = max(float(retry_after), backoff_seconds)
                except ValueError:
                    sleep_s = backoff_seconds * (attempt + 1)
            else:
                sleep_s = backoff_seconds * (attempt + 1)
            log(
                f"Gemini request failed status={resp.status_code}; retrying in {sleep_s:.1f}s "
                f"attempt={attempt + 1}/{max_retries}"
            )
            time.sleep(sleep_s)
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise
            sleep_s = backoff_seconds * (attempt + 1)
            log(
                f"Gemini request error: {exc}; retrying in {sleep_s:.1f}s "
                f"attempt={attempt + 1}/{max_retries}"
            )
            time.sleep(sleep_s)

    raise RuntimeError("Unreachable retry state in call_gemini_summary")


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    gemini_cfg = cfg.get("gemini", {})

    dry_run = bool(run_cfg.get("dry_run", False))
    max_days = int(run_cfg.get("max_days", 0))
    env_path = run_cfg.get("env_file", ".env")

    raw_articles_path = data_cfg.get("raw_articles_path", "data/news_raw/new_york_times.tsv")
    bert_cache_path = data_cfg.get("bert_cache_path", "data/news_preprocessed/BERT_label_cache.jsonl")
    output_path = data_cfg.get("daily_summary_output_path", "data/news_preprocessed/daily_summaries.jsonl")
    sp500_path = data_cfg.get("sp500_path", "data/SP500.csv")
    summary_price_csv_path = data_cfg.get(
        "daily_summary_price_output_path",
        "data/news_preprocessed/daily_summary_and_price.csv",
    )
    report_path = data_cfg.get("report_path", "data/news_preprocessed/daily_summarization_report.json")
    start_date = normalize_date_value(data_cfg.get("start_date", ""))
    end_date = normalize_date_value(data_cfg.get("end_date", ""))

    model = gemini_cfg.get("model", "gemini-2.0-flash")
    max_tokens = int(gemini_cfg.get("max_summary_tokens", 400))
    max_articles_per_day = int(gemini_cfg.get("max_articles_per_day", 120))
    max_chars_per_article = int(gemini_cfg.get("max_chars_per_article", 2000))
    parallel_workers = int(gemini_cfg.get("parallel_workers", 4))
    timeout_seconds = int(gemini_cfg.get("timeout_seconds", 120))
    max_retries = int(gemini_cfg.get("max_retries", 4))
    backoff_seconds = float(gemini_cfg.get("backoff_seconds", 2.0))

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    log(f"Loading BERT cache from {bert_cache_path}")
    label_cache = load_label_cache(bert_cache_path)
    relevant_keys = {
        ck
        for ck, entry in label_cache.items()
        if int(entry.get("label", 0)) == 1 and entry.get("reason") != "labeling_failed"
    }
    log(f"Relevant labels in cache: {len(relevant_keys)}")

    log(f"Reading raw articles from {raw_articles_path}")
    articles = read_tsv(raw_articles_path)
    log(f"Loaded {len(articles)} raw rows")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in articles:
        key = build_dedup_key(row)
        if not key or key not in relevant_keys:
            continue
        day = normalize_text(row.get("article_day"))
        if not day:
            continue
        if start_date and day < start_date:
            continue
        if end_date and day > end_date:
            continue
        grouped.setdefault(day, []).append(row)

    all_days = sorted(grouped.keys())
    if max_days > 0:
        all_days = all_days[:max_days]
    log(f"Days with relevant articles in range: {len(all_days)}")

    completed_days = load_completed_days(output_path)
    pending_days = [d for d in all_days if d not in completed_days]
    log(f"Resume: completed={len(completed_days)} pending={len(pending_days)}")

    if not pending_days:
        sp500_by_day = read_sp500(sp500_path)
        summary_by_day = load_daily_summaries(output_path)
        summary_price_rows, missing_price_days = write_daily_summary_and_price_csv(
            summary_by_day=summary_by_day,
            sp500_by_day=sp500_by_day,
            output_csv_path=summary_price_csv_path,
        )
        log(
            f"Wrote {summary_price_rows} rows to {summary_price_csv_path} "
            f"(missing_price_days={missing_price_days})"
        )
        finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report = {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "total_days": len(all_days),
            "completed_days": len(completed_days),
            "pending_days": 0,
            "summaries_written": 0,
            "summary_price_csv_path": summary_price_csv_path,
            "summary_price_rows": summary_price_rows,
            "summary_days_missing_price": missing_price_days,
            "dry_run": dry_run,
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    api_key = ""
    if not dry_run:
        api_key = load_env_value("GEMINI_API_KEY", env_path=env_path) or ""
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in env or .env")

    def worker(day: str) -> dict[str, Any]:
        rows = grouped[day][:max_articles_per_day]
        blocks = [
            _article_to_block(row, max_chars_per_article=max_chars_per_article)
            for row in rows
        ]
        prompt = _build_prompt(day=day, article_blocks=blocks, max_tokens=max_tokens)
        if dry_run:
            summary = f"[dry-run] articles={len(rows)}"
        else:
            summary = call_gemini_summary(
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                backoff_seconds=backoff_seconds,
            )
        return {
            "article_day": day,
            "summary": summary,
            "summary_model": model,
            "source": "new_york_times",
            "relevant_article_count_used": len(rows),
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    summaries_written = 0
    failures = 0
    workers = max(1, parallel_workers)
    log(f"Starting summarization workers={workers} model={model}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker, day): day for day in pending_days}
        for i, future in enumerate(as_completed(future_map), start=1):
            day = future_map[future]
            try:
                record = future.result()
                append_summary_rows(output_path, [record])
                summaries_written += 1
            except Exception as exc:
                failures += 1
                log(f"Failed day={day}: {exc}")
            if i % 20 == 0 or i == len(pending_days):
                log(
                    f"Progress {i}/{len(pending_days)} "
                    f"written={summaries_written} failures={failures}"
                )

    sp500_by_day = read_sp500(sp500_path)
    summary_by_day = load_daily_summaries(output_path)
    summary_price_rows, missing_price_days = write_daily_summary_and_price_csv(
        summary_by_day=summary_by_day,
        sp500_by_day=sp500_by_day,
        output_csv_path=summary_price_csv_path,
    )
    log(
        f"Wrote {summary_price_rows} rows to {summary_price_csv_path} "
        f"(missing_price_days={missing_price_days})"
    )

    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "raw_rows": len(articles),
        "relevant_keys": len(relevant_keys),
        "total_days": len(all_days),
        "completed_days_before_run": len(completed_days),
        "pending_days_at_start": len(pending_days),
        "summaries_written": summaries_written,
        "failures": failures,
        "model": model,
        "parallel_workers": workers,
        "max_articles_per_day": max_articles_per_day,
        "max_chars_per_article": max_chars_per_article,
        "summary_price_csv_path": summary_price_csv_path,
        "summary_price_rows": summary_price_rows,
        "summary_days_missing_price": missing_price_days,
        "dry_run": dry_run,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Report written to {report_path}")
    return report


def main() -> int:
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/default_summarization.yaml"
    )
    log(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    run_pipeline(cfg)
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
