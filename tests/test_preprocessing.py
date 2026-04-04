"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# The module lives under "src/data fetch/" (space in dir name),
# so we load it via importlib rather than a normal import.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "data fetch" / "preprocessing.py"
_spec = importlib.util.spec_from_file_location("preprocessing", _MODULE_PATH)
preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing)

# Pull names into the test namespace for convenience.
normalize_text = preprocessing.normalize_text
build_dedup_key = preprocessing.build_dedup_key
deduplicate = preprocessing.deduplicate
sort_rows = preprocessing.sort_rows
_parse_label_response = preprocessing._parse_label_response
load_label_cache = preprocessing.load_label_cache
append_cache_entries = preprocessing.append_cache_entries
read_tsv = preprocessing.read_tsv
write_output_tsv = preprocessing.write_output_tsv
interpolate_missing_days = preprocessing.interpolate_missing_days
OUTPUT_FIELDNAMES = preprocessing.OUTPUT_FIELDNAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**overrides: str) -> dict[str, str]:
    base = {
        "article_id": "",
        "source": "new_york_times",
        "query": "",
        "article_day": "2023-01-10",
        "published_at": "2023-01-10T08:00:00+0000",
        "headline": "Test headline",
        "article_text": "Body text",
        "snippet": "",
        "abstract": "",
        "lead_paragraph": "",
        "section_name": "",
        "subsection_name": "",
        "web_url": "",
        "keywords": "",
        "economy_relevance_label": "",
        "is_interpolated_no_news": "0",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Dedup by article_id
# ---------------------------------------------------------------------------

def test_dedup_by_article_id():
    rows = [
        _row(article_id="nyt://1", headline="A"),
        _row(article_id="nyt://1", headline="A duplicate"),
        _row(article_id="nyt://2", headline="B"),
    ]
    unique, removals = deduplicate(rows)
    assert len(unique) == 2
    assert unique[0]["headline"] == "A"
    assert unique[1]["headline"] == "B"
    assert removals["id"] == 1


# ---------------------------------------------------------------------------
# 2. Dedup fallback by web_url
# ---------------------------------------------------------------------------

def test_dedup_fallback_by_web_url():
    rows = [
        _row(web_url="https://nyt.com/article1", headline="A"),
        _row(web_url="https://nyt.com/article1", headline="A dup"),
    ]
    unique, removals = deduplicate(rows)
    assert len(unique) == 1
    assert removals["url"] == 1


# ---------------------------------------------------------------------------
# 3. Dedup fallback by published_at + headline
# ---------------------------------------------------------------------------

def test_dedup_fallback_by_published_at_headline():
    rows = [
        _row(published_at="2023-01-10T08:00:00+0000", headline="Same"),
        _row(published_at="2023-01-10T08:00:00+0000", headline="Same"),
    ]
    unique, removals = deduplicate(rows)
    assert len(unique) == 1
    assert removals["published_headline"] == 1


# ---------------------------------------------------------------------------
# 4. Sorting: older dates first, invalid dates last
# ---------------------------------------------------------------------------

def test_sort_older_first_invalid_last():
    rows = [
        _row(article_day="2023-03-01", published_at="2023-03-01T00:00:00Z"),
        _row(article_day="", published_at=""),
        _row(article_day="2023-01-01", published_at="2023-01-01T00:00:00Z"),
        _row(article_day="bad-date", published_at=""),
    ]
    sorted_rows, invalid_count = sort_rows(rows)
    assert invalid_count == 2
    assert sorted_rows[0]["article_day"] == "2023-01-01"
    assert sorted_rows[1]["article_day"] == "2023-03-01"
    # invalid dates at end
    assert sorted_rows[2]["article_day"] in ("", "bad-date")
    assert sorted_rows[3]["article_day"] in ("", "bad-date")


# ---------------------------------------------------------------------------
# 5. Label response parser rejects malformed JSON
# ---------------------------------------------------------------------------

def test_parse_label_response_valid():
    raw = '[{"id": 0, "label": 1, "reason": "economy related"}]'
    result = _parse_label_response(raw, 1)
    assert result is not None
    assert result[0]["label"] == 1


def test_parse_label_response_rejects_bad_json():
    assert _parse_label_response("not json", 1) is None


def test_parse_label_response_rejects_wrong_structure():
    assert _parse_label_response('{"id": 0, "label": 1}', 1) is None  # not a list


def test_parse_label_response_rejects_invalid_label():
    raw = '[{"id": 0, "label": 2, "reason": "bad"}]'
    assert _parse_label_response(raw, 1) is None


def test_parse_label_response_strips_markdown_fences():
    raw = '```json\n[{"id": 0, "label": 0, "reason": "lifestyle"}]\n```'
    result = _parse_label_response(raw, 1)
    assert result is not None
    assert result[0]["label"] == 0


# ---------------------------------------------------------------------------
# 6. Cache hit avoids API call
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path: Path):
    cache_file = str(tmp_path / "cache.jsonl")

    entries = [
        {"content_key": "id:nyt://1", "label": 1, "reason": "econ", "model": "gemini-2.0-flash",
         "confidence": "", "label_created_at_utc": "2024-01-01T00:00:00Z"},
    ]
    append_cache_entries(cache_file, entries)

    cache = load_label_cache(cache_file)
    assert "id:nyt://1" in cache
    assert cache["id:nyt://1"]["label"] == 1


def test_load_label_cache_skips_bad_lines(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    cache_file.write_text(
        '{"content_key": "id:1", "label": 0}\n'
        'not json\n'
        '{"missing_key": true}\n',
        encoding="utf-8",
    )
    cache = load_label_cache(str(cache_file))
    assert len(cache) == 1
    assert "id:1" in cache


# ---------------------------------------------------------------------------
# 7. End-to-end: small fixture writes expected output columns
# ---------------------------------------------------------------------------

def test_end_to_end_dry_run(tmp_path: Path):
    # Write a small input TSV
    input_path = tmp_path / "input.tsv"
    header = "\t".join(preprocessing.INPUT_FIELDNAMES)
    row1_vals = [
        "nyt://1", "new_york_times", "", "2023-01-10", "2023-01-10T08:00:00+0000",
        "Headline A", "Body A", "", "", "", "Business", "", "https://nyt.com/a", "economy",
    ]
    row2_vals = [
        "nyt://2", "new_york_times", "", "2023-01-05", "2023-01-05T12:00:00+0000",
        "Headline B", "Body B", "", "", "", "Arts", "", "https://nyt.com/b", "art",
    ]
    # duplicate of row 1
    row3_vals = row1_vals.copy()
    input_path.write_text(
        header + "\n" + "\t".join(row1_vals) + "\n" + "\t".join(row2_vals) + "\n" + "\t".join(row3_vals) + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "output.tsv"
    report_path = tmp_path / "report.json"

    cfg = {
        "run": {"name": "test", "dry_run": True, "limit": 0},
        "paths": {
            "input": str(input_path),
            "output": str(output_path),
            "report": str(report_path),
            "label_cache": str(tmp_path / "cache.jsonl"),
            "env_file": str(tmp_path / ".env"),
        },
        "gemini": {
            "model": "gemini-2.0-flash",
            "batch_size": 25,
            "max_text_chars": 2000,
        },
    }

    report = preprocessing.run_pipeline(cfg)

    # Check report
    assert report["input_rows"] == 3
    assert report["duplicates_removed_total"] == 1
    assert report["rows_after_dedup"] == 2

    # Check output has correct columns and is sorted (2023-01-05 before 2023-01-10)
    import csv
    with output_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        out_rows = list(reader)

    assert len(out_rows) == 2
    assert out_rows[0]["article_day"] == "2023-01-05"
    assert out_rows[1]["article_day"] == "2023-01-10"

    # All output columns present
    for col in OUTPUT_FIELDNAMES:
        assert col in out_rows[0], f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 8. Interpolation adds [no news] row when a day has only label=0 rows
# ---------------------------------------------------------------------------

def test_interpolation_adds_row_for_day_with_only_label_0():
    rows = [
        _row(article_day="2023-01-01", economy_relevance_label="1"),
        _row(article_day="2023-01-02", economy_relevance_label="0"),
        _row(article_day="2023-01-03", economy_relevance_label="1"),
    ]
    result, missing_days, synthetic_added = interpolate_missing_days(rows)

    # 2023-01-02 has only label=0, so it should get a synthetic row
    assert missing_days == 1
    assert synthetic_added == 1

    interpolated = [r for r in result if r.get("is_interpolated_no_news") == "1"]
    assert len(interpolated) == 1
    assert interpolated[0]["article_day"] == "2023-01-02"
    assert interpolated[0]["headline"] == "[no news]"
    assert interpolated[0]["economy_relevance_label"] == "0"


# ---------------------------------------------------------------------------
# 9. Interpolation adds [no news] row when a calendar day has no rows at all
# ---------------------------------------------------------------------------

def test_interpolation_adds_row_for_calendar_gap():
    # 2023-01-02 has no rows at all
    rows = [
        _row(article_day="2023-01-01", economy_relevance_label="1"),
        _row(article_day="2023-01-03", economy_relevance_label="1"),
    ]
    result, missing_days, synthetic_added = interpolate_missing_days(rows)

    assert missing_days == 1
    assert synthetic_added == 1

    interpolated = [r for r in result if r.get("is_interpolated_no_news") == "1"]
    assert len(interpolated) == 1
    assert interpolated[0]["article_day"] == "2023-01-02"
    assert interpolated[0]["headline"] == "[no news]"

    # Result should be sorted: 01, 02 (synthetic), 03
    days = [r["article_day"] for r in result]
    assert days == ["2023-01-01", "2023-01-02", "2023-01-03"]
