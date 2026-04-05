from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ARTICLES_PATH = "data/news_raw/new_york_times.tsv"
DEFAULT_BERT_CACHE_PATH = "data/news_preprocessed/BERT_label_cache.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print article by article_id and its BERT relevance label."
    )
    parser.add_argument("article_id", help="Raw article_id or content_key (id:...)")
    parser.add_argument("--articles-path", default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--bert-cache-path", default=DEFAULT_BERT_CACHE_PATH)
    return parser.parse_args()


def to_content_key(article_id_or_key: str) -> str:
    value = article_id_or_key.strip()
    if value.startswith("id:") or value.startswith("url:"):
        return value
    return f"id:{value}"


def find_article_row(articles_path: Path, article_id: str) -> dict[str, str] | None:
    with articles_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if (row.get("article_id") or "").strip() == article_id:
                return {k: (v or "") for k, v in row.items()}
    return None


def find_cache_entry(cache_path: Path, content_key: str) -> dict | None:
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("content_key") == content_key:
                return entry
    return None


def main() -> int:
    args = parse_args()
    articles_path = Path(args.articles_path)
    cache_path = Path(args.bert_cache_path)

    if not articles_path.exists():
        raise FileNotFoundError(f"Articles file not found: {articles_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"BERT cache file not found: {cache_path}")

    query = args.article_id.strip()
    content_key = to_content_key(query)
    article_id = content_key.split("id:", 1)[1] if content_key.startswith("id:") else query

    row = find_article_row(articles_path, article_id)
    if row is None:
        print(f"Article not found in TSV for article_id: {article_id}")
    else:
        print("Article:")
        print(f"  article_id: {row.get('article_id', '')}")
        print(f"  article_day: {row.get('article_day', '')}")
        print(f"  published_at: {row.get('published_at', '')}")
        print(f"  headline: {row.get('headline', '')}")
        print(f"  section_name: {row.get('section_name', '')}")
        print(f"  web_url: {row.get('web_url', '')}")
        print(f"  article_text: {row.get('article_text', '')}")

    entry = find_cache_entry(cache_path, content_key)
    if entry is None:
        print(f"BERT label not found for content_key: {content_key}")
        return 0

    label = int(entry.get("label", 0))
    status = "RELEVANT" if label == 1 else "NOT RELEVANT"
    print("BERT relevance:")
    print(f"  content_key: {content_key}")
    print(f"  label: {label} ({status})")
    print(f"  reason: {entry.get('reason', '')}")
    print(f"  model: {entry.get('model', '')}")
    print(f"  confidence: {entry.get('confidence', '')}")
    print(f"  label_created_at_utc: {entry.get('label_created_at_utc', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
