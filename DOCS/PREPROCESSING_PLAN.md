# Preprocessing Plan For `src/data fetch/preprocessing.py`

## Goal

Build a preprocessing script that:

1. Reads `data/news_raw/new_york_times.tsv`.
2. Drops true accidental duplicates.
3. Sorts rows by `article_day` ascending (oldest first).
4. Calls a cheap Gemini model to classify each article:
   - `label=1`: likely relevant to global economy / markets (economy, politics, business, technology, major events).
   - `label=0`: unlikely to impact global economy / S&P 500 (arts, movies, lifestyle, etc.).
5. Writes a clean labeled output file and metadata report.


## Data Contract

Input columns (current schema):

- `article_id`
- `source`
- `query`
- `article_day`
- `published_at`
- `headline`
- `article_text`
- `snippet`
- `abstract`
- `lead_paragraph`
- `section_name`
- `subsection_name`
- `web_url`
- `keywords`

Output columns:

- all input columns
- `economy_relevance_label` (`0` or `1`)
- `economy_relevance_reason` (short model rationale, optional but useful for audit)
- `economy_relevance_model` (e.g. `gemini-1.5-flash`)
- `economy_relevance_confidence` (optional if returned)
- `label_created_at_utc`


## Duplicate Strategy

Drop only real duplicate pulls, keeping first occurrence.

Dedup key priority per row:

1. `article_id` if non-empty.
2. Else `web_url` if non-empty.
3. Else `published_at + headline` if both non-empty.
4. Else `article_day + headline` if both non-empty.
5. Else treat row as unique.

Implementation details:

- Build `dedup_key` with normalized text (trim + collapse whitespace).
- Keep a `set` of seen keys.
- Count and report how many rows were removed by each key type.


## Sorting Strategy

Sort on:

1. `article_day` ascending.
2. Tie-breaker `published_at` ascending.
3. Tie-breaker `article_id` / `web_url` for stable deterministic output.

Invalid/missing dates:

- Parse `article_day` with `%Y-%m-%d`.
- If invalid/missing, put at end and log count.


## Gemini Labeling Design

### Model and Cost Controls

- Default model: cheap/fast Gemini variant (configurable, e.g. `gemini-1.5-flash`).
- Batch requests in small chunks (e.g. 20-50 rows/request).
- Cache label results by a stable content key to avoid re-charging on reruns.

### Label Input Text

For each row, construct compact text:

- `headline`
- `section_name`, `subsection_name`
- `keywords`
- `article_text` truncated to fixed max chars/tokens

### Prompt Contract

Require strict JSON response for each item:

- `id` (request row id)
- `label` (`0` or `1`)
- `reason` (<= 20 words)

Rules in prompt:

- `1` for macro/market-sensitive topics.
- `0` for topics unlikely to affect global economy.
- Be conservative on weak signals; only output `1` if plausible macro linkage exists.

### Reliability

- Validate JSON schema.
- Retry malformed responses with smaller batch size.
- If still failing, fallback label `0` with `reason="labeling_failed"` and log error.


## Script CLI

Proposed arguments:

- `--input-path` default `data/news_raw/new_york_times.tsv`
- `--output-path` default `data/news_processed/new_york_times_preprocessed.tsv`
- `--report-path` default `data/news_processed/preprocessing_report.json`
- `--cache-path` default `data/news_processed/gemini_label_cache.jsonl`
- `--model` default cheap Gemini model
- `--batch-size` default `25`
- `--max-text-chars` default `2000`
- `--dry-run` (do dedup/sort only, no API calls)
- `--limit` for smoke tests

Environment:

- `GEMINI_API_KEY` required for labeling unless `--dry-run`.


## Pipeline Steps In Code

1. Read TSV rows.
2. Normalize key text fields.
3. Deduplicate.
4. Sort.
5. Load label cache.
6. Build unlabeled queue (skip cached keys).
7. Call Gemini in batches, validate/parse output, update cache.
8. Attach labels to all rows.
9. Interpolate missing dates based on relevant-news coverage:
   - A day is considered to have relevant news only if it has at least one row with `economy_relevance_label=1`.
   - Rows labeled `0` do not count toward relevant coverage.
   - If a date has only label `0` rows (or no rows), treat that date as missing relevant news.
   - Add one synthetic placeholder row for that date, keeping the date fields and setting non-date fields to `[no news]`.
10. Write output TSV.
11. Write JSON report with stats and failures.


## Output Report Metrics

- `input_rows`
- `rows_after_dedup`
- `duplicates_removed_total`
- `duplicates_removed_by_key_type`
- `rows_with_invalid_article_day`
- `rows_labeled_from_cache`
- `rows_labeled_from_api`
- `rows_label_failures`
- `missing_relevant_news_days_interpolated`
- `synthetic_no_news_rows_added`
- `started_at_utc`, `finished_at_utc`


## Missing-Date Interpolation Rules

Apply after labeling and before final write.

Date range:

- Use min/max `article_day` observed in sorted rows (or explicit CLI range if provided later).

Daily logic:

- If day has >=1 row with `economy_relevance_label=1`: keep rows as-is, no interpolation row.
- If day has rows but all are `economy_relevance_label=0`: day is missing relevant news, add one interpolation row.
- If day has no rows at all: also add one interpolation row.

Interpolation row content:

- Keep date column(s): `article_day` set to that day, `published_at` can be `[no news]` unless a stricter date-time convention is chosen later.
- Set `[no news]` to:
  - `headline`
  - `article_text`
  - `snippet`
  - `abstract`
  - `lead_paragraph`
  - `section_name`
  - `subsection_name`
  - `web_url`
  - and all other non-date columns
- `economy_relevance_label` for interpolation row should be `0`.
- Add optional marker column `is_interpolated_no_news=1` (recommended for downstream filtering).


## Testing Plan

Add tests in `tests/test_preprocessing.py`:

1. Dedup by `article_id`.
2. Dedup fallback by `web_url`.
3. Dedup fallback by `published_at + headline`.
4. Sorting places older dates first and invalid dates last.
5. Label response parser rejects malformed JSON.
6. Cache hit avoids API call.
7. End-to-end small fixture writes expected output columns.
8. Interpolation adds `[no news]` row when a day has only label `0` rows.
9. Interpolation adds `[no news]` row when a calendar day has no rows at all.


## Implementation Order

1. Build pure local preprocessing (read, dedup, sort, write, report).
2. Add Gemini client wrapper + JSON schema validation.
3. Add cache and retry logic.
4. Add tests with mocked Gemini responses.
5. Document usage in `DOCS/SCRIPTS.md`.


## Notes From Current Dataset Check

- `article_id` duplicates currently appear to be `0` in your local TSV sample check.
- `web_url` duplicates are non-zero, so fallback dedup by URL is necessary.
