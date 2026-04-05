"""Tests for summary-price sequence builder."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_generation import build_summary_price_sequences as seq_builder


def test_build_sequences_uses_next_day_target() -> None:
    rows = [
        {"date": "2023-01-01", "sp500_close": "100", "summary": "s1"},
        {"date": "2023-01-02", "sp500_close": "101", "summary": "s2"},
        {"date": "2023-01-03", "sp500_close": "102", "summary": "s3"},
        {"date": "2023-01-04", "sp500_close": "103", "summary": "s4"},
    ]
    sequences = seq_builder.build_sequences(rows, window_size=2)
    assert len(sequences) == 2

    first = sequences[0]
    assert first["window_start_date"] == "2023-01-01"
    assert first["window_end_date"] == "2023-01-02"
    assert first["target_date"] == "2023-01-03"
    assert first["target_next_close"] == "102"
    assert json.loads(first["window_dates"]) == ["2023-01-01", "2023-01-02"]
    assert json.loads(first["window_summaries_json"]) == ["s1", "s2"]


def test_read_and_write_sequences_roundtrip(tmp_path: Path) -> None:
    input_csv = tmp_path / "daily_summary_and_price.csv"
    output_csv = tmp_path / "summary_price_sequences.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "sp500_close", "summary"])
        writer.writeheader()
        writer.writerow({"date": "2023-01-01", "sp500_close": "100", "summary": "s1"})
        writer.writerow({"date": "2023-01-02", "sp500_close": "101", "summary": "s2"})
        writer.writerow({"date": "2023-01-03", "sp500_close": "102", "summary": "s3"})

    rows = seq_builder.read_daily_rows(str(input_csv))
    sequences = seq_builder.build_sequences(rows, window_size=2)
    seq_builder.write_sequences(str(output_csv), sequences)

    out_text = output_csv.read_text(encoding="utf-8")
    assert "target_next_close" in out_text
    assert "2023-01-03" in out_text
