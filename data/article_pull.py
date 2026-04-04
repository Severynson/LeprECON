from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


API_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
ARCHIVE_API_URL_TEMPLATE = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
DEFAULT_QUERY = ""
DEFAULT_START_DATE = "2018-10-01"
DEFAULT_OUTPUT_PATH = "data/news_raw/new_york_times.tsv"
DEFAULT_CHECKPOINT_PATH = "data/news_raw/new_york_times.checkpoint.json"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 5.0
MAX_PAGES_PER_DAY = 100

FIELDNAMES = [
    "article_id",
    "source",
    "query",
    "article_day",
    "published_at",
    "headline",
    "article_text",
    "snippet",
    "abstract",
    "lead_paragraph",
    "section_name",
    "subsection_name",
    "web_url",
    "keywords",
]


@dataclass
class PullCheckpoint:
    current_date: str
    next_page: int


def load_env_value(key: str, env_path: str = ".env") -> str | None:
    direct_value = os.environ.get(key)
    if direct_value:
        return direct_value

    env_file = Path(env_path)
    if not env_file.exists():
        return None

    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = pattern.match(line)
        if not match:
            continue

        env_key, env_value = match.groups()
        if env_key != key:
            continue

        value = env_value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value

    return None


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[article_pull {timestamp}] {message}", flush=True)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def build_article_text(doc: dict[str, Any]) -> str:
    candidates = [
        doc.get("lead_paragraph"),
        doc.get("snippet"),
        doc.get("abstract"),
        doc.get("headline", {}).get("main"),
    ]
    for candidate in candidates:
        cleaned = normalize_text(candidate)
        if cleaned:
            return cleaned
    return ""


def extract_keywords(doc: dict[str, Any]) -> str:
    names = [item.get("value", "") for item in doc.get("keywords", [])]
    return " | ".join(filter(None, (normalize_text(name) for name in names)))


def extract_article_row(doc: dict[str, Any], query: str) -> dict[str, str]:
    headline = normalize_text(doc.get("headline", {}).get("main"))
    snippet = normalize_text(doc.get("snippet"))
    abstract = normalize_text(doc.get("abstract"))
    lead_paragraph = normalize_text(doc.get("lead_paragraph"))
    published_at = normalize_text(doc.get("pub_date"))
    article_day = published_at[:10] if published_at else ""

    return {
        "article_id": normalize_text(doc.get("_id")),
        "source": "new_york_times",
        "query": normalize_text(query),
        "article_day": article_day,
        "published_at": published_at,
        "headline": headline,
        "article_text": build_article_text(doc),
        "snippet": snippet,
        "abstract": abstract,
        "lead_paragraph": lead_paragraph,
        "section_name": normalize_text(doc.get("section_name")),
        "subsection_name": normalize_text(doc.get("subsection_name")),
        "web_url": normalize_text(doc.get("web_url")),
        "keywords": extract_keywords(doc),
    }


def format_nyt_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def format_month_cursor(value: date) -> str:
    return value.strftime("%Y-%m")


def iter_dates(start_date: str, end_date: str) -> list[date]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    current = start
    dates: list[date] = []
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def iter_month_starts(start_date: str, end_date: str) -> list[date]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date().replace(day=1)
    end = datetime.strptime(end_date, "%Y-%m-%d").date().replace(day=1)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    months: list[date] = []
    current = start
    while current <= end:
        months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def read_existing_article_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    seen_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            article_id = row.get("article_id", "")
            if article_id:
                seen_ids.add(article_id)
    return seen_ids


def ensure_output_writer(output_path: Path) -> csv.DictWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    handle = output_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
    if not file_exists:
        writer.writeheader()
        handle.flush()
    writer._output_handle = handle  # type: ignore[attr-defined]
    return writer


def close_output_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_output_handle", None)
    if handle is not None:
        handle.close()


def load_checkpoint(checkpoint_path: Path) -> PullCheckpoint | None:
    if not checkpoint_path.exists():
        return None

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return PullCheckpoint(
        current_date=payload["current_date"],
        next_page=int(payload["next_page"]),
    )


def write_checkpoint(checkpoint_path: Path, current_date: str, next_page: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_date": current_date,
        "next_page": next_page,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def search_nyt_articles(
    query: str,
    begin_date: str,
    end_date: str,
    api_key: str,
    *,
    page: int = 0,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    params = {
        "q": query,
        "begin_date": begin_date,
        "end_date": end_date,
        "sort": "oldest",
        "page": page,
        "api-key": api_key,
    }
    http = session or requests.Session()
    for attempt in range(max_retries + 1):
        if attempt > 0:
            log(
                "Retrying NYT request "
                f"query={query!r} begin_date={begin_date} end_date={end_date} page={page} "
                f"attempt={attempt}/{max_retries}"
            )
        response = http.get(API_URL, params=params, timeout=timeout_seconds)
        if response.status_code < 400:
            if attempt > 0:
                log(
                    "NYT request recovered "
                    f"query={query!r} begin_date={begin_date} end_date={end_date} page={page}"
                )
            return response.json()

        should_retry = response.status_code == 429 or response.status_code >= 500
        log(
            "NYT request failed "
            f"status={response.status_code} query={query!r} begin_date={begin_date} "
            f"end_date={end_date} page={page}"
        )
        if not should_retry or attempt == max_retries:
            response.raise_for_status()

        retry_after_header = response.headers.get("Retry-After")
        if retry_after_header:
            try:
                sleep_seconds = max(float(retry_after_header), backoff_seconds)
            except ValueError:
                sleep_seconds = backoff_seconds * (attempt + 1)
        else:
            sleep_seconds = backoff_seconds * (attempt + 1)
        log(
            "Backing off before retry "
            f"sleep_seconds={sleep_seconds:.1f} query={query!r} begin_date={begin_date} "
            f"end_date={end_date} page={page}"
        )
        time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state in search_nyt_articles")


def fetch_nyt_archive_month(
    *,
    year: int,
    month: int,
    api_key: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    url = ARCHIVE_API_URL_TEMPLATE.format(year=year, month=month)
    params = {"api-key": api_key}
    http = session or requests.Session()

    for attempt in range(max_retries + 1):
        if attempt > 0:
            log(
                f"Retrying NYT archive request year={year} month={month} "
                f"attempt={attempt}/{max_retries}"
            )

        response = http.get(url, params=params, timeout=timeout_seconds)
        if response.status_code < 400:
            if attempt > 0:
                log(f"NYT archive request recovered year={year} month={month}")
            return response.json()

        should_retry = response.status_code == 429 or response.status_code >= 500
        log(
            f"NYT archive request failed status={response.status_code} year={year} month={month}"
        )
        if not should_retry or attempt == max_retries:
            response.raise_for_status()

        retry_after_header = response.headers.get("Retry-After")
        if retry_after_header:
            try:
                sleep_seconds = max(float(retry_after_header), backoff_seconds)
            except ValueError:
                sleep_seconds = backoff_seconds * (attempt + 1)
        else:
            sleep_seconds = backoff_seconds * (attempt + 1)
        log(
            f"Backing off archive request sleep_seconds={sleep_seconds:.1f} year={year} month={month}"
        )
        time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry state in fetch_nyt_archive_month")


def extract_response_docs(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    response_block = payload.get("response")
    if not isinstance(response_block, dict):
        log(
            f"Unexpected NYT payload shape: missing response object keys={list(payload.keys())}"
        )
        return [], None

    docs = response_block.get("docs")
    meta = response_block.get("meta", {})
    hits = meta.get("hits") if isinstance(meta, dict) else None

    if docs is None:
        log(
            "NYT payload returned docs=None "
            f"hits={hits} response_keys={list(response_block.keys())}"
        )
        return [], hits

    if not isinstance(docs, list):
        log(
            "Unexpected NYT docs type "
            f"type={type(docs).__name__} hits={hits} response_keys={list(response_block.keys())}"
        )
        return [], hits

    return docs, hits


def is_date_in_range(article_day: str, start_date: str, end_date: str) -> bool:
    if not article_day:
        return False
    return start_date <= article_day <= end_date


def pull_articles(
    *,
    api_key: str,
    query: str,
    start_date: str,
    end_date: str,
    output_path: str,
    checkpoint_path: str,
    page_limit: int = MAX_PAGES_PER_DAY,
    smoke_test: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    mode: str = "archive",
) -> int:
    output = Path(output_path)
    checkpoint = Path(checkpoint_path)
    seen_ids = read_existing_article_ids(output)
    writer = ensure_output_writer(output)

    checkpoint_state = load_checkpoint(checkpoint)
    session = requests.Session()
    rows_written = 0
    resume_started = checkpoint_state is None
    processed_units = 0
    empty_units = 0

    log(
        "Starting pull "
        f"mode={mode} "
        f"query={query!r} start_date={start_date} end_date={end_date} "
        f"output_path={output_path} smoke_test={smoke_test}"
    )
    if checkpoint_state:
        log(
            "Resuming from checkpoint "
            f"current_date={checkpoint_state.current_date} next_page={checkpoint_state.next_page}"
        )

    try:
        if mode == "archive":
            months = iter_month_starts(start_date, end_date)
            if smoke_test:
                months = months[:1]

            for month_start in months:
                month_cursor = format_month_cursor(month_start)
                if not resume_started:
                    checkpoint_cursor = (
                        checkpoint_state.current_date[:7] if checkpoint_state else ""
                    )
                    if checkpoint_cursor != month_cursor:
                        continue
                    resume_started = True

                checkpoint_state = None
                month_rows_written = 0
                processed_units += 1
                log(f"Processing month={month_cursor}")

                payload = fetch_nyt_archive_month(
                    year=month_start.year,
                    month=month_start.month,
                    api_key=api_key,
                    max_retries=max_retries,
                    backoff_seconds=backoff_seconds,
                    session=session,
                )
                docs, hits = extract_response_docs(payload)
                log(
                    f"Fetched month={month_cursor} docs={len(docs)} hits={hits} "
                    f"rows_written_total={rows_written}"
                )

                if not docs:
                    empty_units += 1
                    log(
                        f"No archive articles found for month={month_cursor} empty_months_so_far={empty_units}"
                    )
                    write_checkpoint(checkpoint, current_date=month_cursor, next_page=0)
                    continue

                for doc in docs:
                    row = extract_article_row(doc, query)
                    article_day = row["article_day"]
                    if not is_date_in_range(article_day, start_date, end_date):
                        continue

                    article_id = row["article_id"]
                    if article_id and article_id in seen_ids:
                        continue

                    writer.writerow(row)
                    writer._output_handle.flush()  # type: ignore[attr-defined]
                    if article_id:
                        seen_ids.add(article_id)
                    rows_written += 1
                    month_rows_written += 1

                write_checkpoint(checkpoint, current_date=month_cursor, next_page=0)
                log(
                    f"Finished month={month_cursor} month_rows_written={month_rows_written} "
                    f"processed_months={processed_units} total_rows_written={rows_written}"
                )
        elif mode == "search":
            dates = iter_dates(start_date, end_date)
            if smoke_test:
                dates = dates[:1]

            for day in dates:
                day_iso = day.isoformat()
                if not resume_started:
                    if checkpoint_state and checkpoint_state.current_date != day_iso:
                        continue
                    resume_started = True

                start_page = (
                    checkpoint_state.next_page
                    if checkpoint_state and checkpoint_state.current_date == day_iso
                    else 0
                )
                checkpoint_state = None
                day_rows_written = 0
                processed_units += 1
                log(f"Processing day={day_iso} start_page={start_page}")

                for page in range(start_page, page_limit):
                    nyt_day = format_nyt_date(day)
                    payload = search_nyt_articles(
                        query=query,
                        begin_date=nyt_day,
                        end_date=nyt_day,
                        api_key=api_key,
                        page=page,
                        max_retries=max_retries,
                        backoff_seconds=backoff_seconds,
                        session=session,
                    )
                    docs, hits = extract_response_docs(payload)

                    log(
                        f"Fetched day={day_iso} page={page} docs={len(docs)} hits={hits} "
                        f"rows_written_total={rows_written}"
                    )

                    if not docs:
                        if page == 0:
                            empty_units += 1
                            log(
                                f"No articles found for day={day_iso} query={query!r} "
                                f"empty_days_so_far={empty_units}"
                            )
                        break

                    for doc in docs:
                        row = extract_article_row(doc, query)
                        article_id = row["article_id"]
                        if article_id and article_id in seen_ids:
                            continue

                        writer.writerow(row)
                        writer._output_handle.flush()  # type: ignore[attr-defined]
                        if article_id:
                            seen_ids.add(article_id)
                        rows_written += 1
                        day_rows_written += 1

                    write_checkpoint(
                        checkpoint, current_date=day_iso, next_page=page + 1
                    )
                    log(
                        f"Checkpoint saved day={day_iso} next_page={page + 1} "
                        f"day_rows_written={day_rows_written} total_rows_written={rows_written}"
                    )

                write_checkpoint(checkpoint, current_date=day_iso, next_page=0)
                log(
                    f"Finished day={day_iso} day_rows_written={day_rows_written} "
                    f"processed_days={processed_units} total_rows_written={rows_written}"
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        clear_checkpoint(checkpoint)
        log(
            f"Pull complete processed_units={processed_units} empty_units={empty_units} "
            f"total_rows_written={rows_written}"
        )
    finally:
        close_output_writer(writer)
        session.close()

    return rows_written


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull New York Times articles into a TSV file."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--mode", choices=["archive", "search"], default="archive")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--page-limit", type=int, default=MAX_PAGES_PER_DAY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--env-path", default=".env")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    api_key = load_env_value("NYT_API_KEY", env_path=args.env_path)
    if not api_key:
        raise RuntimeError(
            "NYT_API_KEY was not found in the environment or in the .env file."
        )

    log(f"Loaded NYT_API_KEY from env_path={args.env_path}")
    rows_written = pull_articles(
        api_key=api_key,
        query=args.query,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        page_limit=args.page_limit,
        smoke_test=args.smoke_test,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        mode=args.mode,
    )
    print(f"Wrote {rows_written} article rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
