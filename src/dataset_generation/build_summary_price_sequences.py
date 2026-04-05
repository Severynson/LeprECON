"""Build N-length text+price windows with next-day close target.

Input CSV columns:
- date (YYYY-MM-DD)
- sp500_close
- summary

Output CSV columns:
- window_start_date
- window_end_date
- window_dates
- window_summaries_json
- window_sp500_closes_json
- target_date
- target_next_close
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT_PATH = "data/news_preprocessed/daily_summary_and_price.csv"
DEFAULT_OUTPUT_PATH = "data/news_preprocessed/summary_price_sequences.csv"
DEFAULT_WINDOW_SIZE = 16


def read_daily_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: (v or "") for k, v in row.items()} for row in reader]
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


def build_sequences(rows: list[dict[str, str]], window_size: int) -> list[dict[str, str]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    sequences: list[dict[str, str]] = []
    max_start = len(rows) - window_size - 1
    if max_start < 0:
        return sequences

    for start in range(max_start + 1):
        window = rows[start : start + window_size]
        target = rows[start + window_size]

        sequences.append(
            {
                "window_start_date": window[0]["date"],
                "window_end_date": window[-1]["date"],
                "window_dates": json.dumps([r["date"] for r in window], ensure_ascii=False),
                "window_summaries_json": json.dumps(
                    [r["summary"] for r in window], ensure_ascii=False
                ),
                "window_sp500_closes_json": json.dumps(
                    [r["sp500_close"] for r in window], ensure_ascii=False
                ),
                "target_date": target["date"],
                "target_next_close": target["sp500_close"],
            }
        )

    return sequences


def write_sequences(path: str, rows: list[dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_start_date",
        "window_end_date",
        "window_dates",
        "window_summaries_json",
        "window_sp500_closes_json",
        "target_date",
        "target_next_close",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build N-length summary/price windows with next-day close target."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_daily_rows(args.input_path)
    sequences = build_sequences(rows, args.window_size)
    write_sequences(args.output_path, sequences)
    print(
        f"Built {len(sequences)} sequences from {len(rows)} daily rows "
        f"(window_size={args.window_size})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
