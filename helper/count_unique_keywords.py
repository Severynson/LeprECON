from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT_PATH = "data/news_raw/new_york_times.tsv"
KEYWORDS_DELIMITER = "|"


def count_unique_keywords(input_path: Path) -> int:
    unique_keywords: set[str] = set()

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            raw_keywords = row.get("keywords", "")
            if not raw_keywords:
                continue
            for keyword in raw_keywords.split(KEYWORDS_DELIMITER):
                cleaned = keyword.strip()
                if cleaned:
                    unique_keywords.add(cleaned)

    return len(unique_keywords)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count unique keywords in the 'keywords' column of a TSV file."
        )
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    unique_count = count_unique_keywords(input_path)
    print(unique_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
