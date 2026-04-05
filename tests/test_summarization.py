"""Tests for daily summarization pipeline helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_generation import summarization


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "article_id": "nyt://article/1",
        "source": "new_york_times",
        "query": "",
        "article_day": "2023-01-10",
        "published_at": "2023-01-10T08:00:00+0000",
        "headline": "Headline",
        "article_text": "Article body text",
        "snippet": "Snippet",
        "abstract": "Abstract",
        "lead_paragraph": "Lead",
        "section_name": "Business",
        "subsection_name": "",
        "web_url": "https://example.com",
        "keywords": "economy | stocks",
    }
    base.update(overrides)
    return base


def test_build_dedup_key_prefers_article_id() -> None:
    key = summarization.build_dedup_key(_row(article_id="nyt://article/xyz"))
    assert key == "id:nyt://article/xyz"


def test_article_to_block_truncates_text() -> None:
    block = summarization._article_to_block(
        _row(article_text="A" * 5000),
        max_chars_per_article=120,
    )
    assert "Headline: Headline" in block
    assert len(block) < 5600


def test_load_completed_days_reads_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "daily_summaries.jsonl"
    output.write_text(
        json.dumps({"article_day": "2023-01-01", "summary": "x"}) + "\n"
        + json.dumps({"article_day": "2023-01-02", "summary": "y"}) + "\n",
        encoding="utf-8",
    )
    completed = summarization.load_completed_days(str(output))
    assert completed == {"2023-01-01", "2023-01-02"}


def test_write_daily_summary_and_price_csv(tmp_path: Path) -> None:
    output_csv = tmp_path / "daily_summary_and_price.csv"
    rows_written, missing = summarization.write_daily_summary_and_price_csv(
        summary_by_day={
            "2023-01-01": "summary 1",
            "2023-01-02": "summary 2",
        },
        sp500_by_day={
            "2023-01-01": "3800.50",
        },
        output_csv_path=str(output_csv),
    )
    assert rows_written == 1
    assert missing == 1
    text = output_csv.read_text(encoding="utf-8")
    assert "date,sp500_close,summary" in text
    assert "2023-01-01,3800.50,summary 1" in text
