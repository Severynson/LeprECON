# Architecture Plan

## 1. System Purpose

Build a research pipeline that predicts next-day S&P 500 behavior from:

- market time-series features
- daily news-derived semantic embeddings

The system supports both:

- classification: next-day direction
- regression: raw next-day S&P 500 close

The architecture must make it easy to swap:

- news providers
- summarization settings
- text encoders
- daily embedding aggregation methods
- forecasting models
- feature sets

## 2. Chosen High-Level Design

Use a two-stage architecture:

1. Pull and persist raw historical news articles.
2. Clean and summarize every article into a consistent format.
3. Encode each summarized article with FinBERT.
4. Pool token embeddings into one vector per article.
5. Aggregate article embeddings into one daily embedding.
6. Join daily news features with market features.
7. Build rolling windows of length `N`.
8. Train classification and regression models, primarily with a Transformer-based time-series architecture.
9. Evaluate on one final chronological holdout split.

This structure is preferred over an end-to-end monolithic model because it is:

- easier to debug
- easier to cache
- cheaper to rerun
- safer for incremental experimentation
- better for comparing ablations

## 3. Confirmed Architectural Decisions

### 3.1 Prediction targets

Chosen v1 targets:

- classification target: next-day direction
- regression target: raw next-day close

Even though return-based regression is often easier statistically, raw next-day close is the current project choice and should be implemented as the default target through config.

### 3.2 Article-to-day alignment

Chosen v1 rule:

- articles are assigned to the same day they were published

This decision must be implemented consistently in:

- ingestion
- preprocessing
- dataset generation
- label construction

### 3.3 Text encoder

Chosen v1 setup:

- default encoder: FinBERT
- encoder mode: frozen
- architecture requirement: encoder interface must stay flexible so fine-tuning can be added later

### 3.4 Article representation

Each article should produce one embedding vector.

Recommended v1:

- tokenize summarized article text with the selected encoder tokenizer
- use mean pooling over the final hidden state unless overridden by config
- persist article-level embeddings for reuse

### 3.5 Daily aggregation

Recommended v1:

- aggregate same-day article embeddings with a plain mean
- store article count
- store source counts if multiple sources are enabled
- store a `has_news` indicator

For no-news days:

- use a zero vector
- include an explicit `has_news` indicator because it is usually more informative than a silent zero vector alone

### 3.6 Summarization policy

Chosen direction:

- summarize every article
- use summarization to keep input format consistent before encoding
- keep summarization configurable through YAML
- cache raw text, summary text, and summary metadata
- keep summarization outside the model training path

### 3.7 Market feature family

Chosen v1 market features:

- returns
- technical indicators

Later additions may include:

- employment rate
- GDP
- other macroeconomic time-series

The architecture should therefore treat the per-day feature vector as extensible rather than fixed to `[embedding, price]`.

### 3.8 Forecasting model family

Recommended v1 model stack:

- ARIMA baseline
- linear regression or logistic regression baselines on technical indicators only
- the same downstream model families without BERT embeddings as ablation baselines
- primary model: Transformer-based sequence model on technical indicators plus news embeddings
- comparison model: LSTM or GRU on technical indicators plus news embeddings

Project intent:

- Transformer is the main forecasting architecture for both regression and classification
- LSTM is included as a secondary comparison model, not the primary design target

### 3.9 Evaluation protocol

Chosen v1:

- use one final chronological holdout
- no random split across time

Metrics:

- classification: accuracy, F1, ROC-AUC
- regression: MAE, RMSE, R2

### 3.10 Reproducibility and config-driven execution

The implementation should be config-driven from the start.

Required initial config files:

- `configs/default_train.yaml`
- `configs/default_summarization.yaml`
- `configs/default_dataset_generation.yaml`

Those configs should control:

- selected text encoder
- source article files
- output dataset versions
- training feature toggles
- summarization model and retry behavior
- train and test date ranges

## 4. Raw Data Acquisition Architecture

This project needs a dedicated article-pull script separate from summarization and training.

Requirements:

- pull articles once and store them locally
- fetch only the configured historical interval
- v1 pull interval starts in October 2018
- do not pull earlier than the configured start
- do not pull later than the configured end
- save intermediate progress during long pulls
- support recovery if the process stops midway

Recommended output format:

- CSV for primary persisted article storage
- optional secondary cached artifacts later if needed

This separation is important because API access is rate-limited or cost-sensitive and should not be coupled to every downstream experiment.

## 5. Proposed Data Model

### 5.1 Raw article record

Each raw article record should contain:

- `article_id`
- `source`
- `headline`
- `published_at`
- `section`
- `url`
- `raw_text`
- `pull_batch_id`
- `pulled_at`
- `article_day`

### 5.2 Summarized article record

Each summarized article record should contain:

- `article_id`
- `source`
- `article_day`
- `summary_text`
- `summary_model`
- `summary_prompt_version`
- `summary_status`
- `summary_attempts`

### 5.3 Article embedding record

Each embedding record should contain:

- `article_id`
- `article_day`
- `encoder_name`
- `pooling`
- `embedding`

### 5.4 Daily news feature record

One row per day:

- `day`
- `daily_embedding`
- `article_count`
- `source_counts`
- `has_news`

### 5.5 Market feature record

One row per day:

- `day`
- `open`
- `high`
- `low`
- `close`
- `volume`
- returns
- technical indicators
- optional macro features

### 5.6 Model-ready sequence sample

For each sample:

- input: previous `N` daily feature rows
- target: next day's label

Example:

- `X[t-N+1 : t] -> y[t+1]`

## 6. Proposed Module Layout

Recommended code structure:

```text
src/
  config/
    schemas.py
    loaders.py
  data/
    article_pull.py
    nytimes_client.py
    market_data.py
    storage.py
  preprocessing/
    clean_text.py
    summarize.py
    align_to_day.py
  embeddings/
    encoder.py
    pooling.py
    aggregate_daily.py
  features/
    technical_indicators.py
    dataset_builder.py
    sequence_builder.py
  models/
    baselines.py
    transformer.py
    recurrent.py
  training/
    train_classifier.py
    train_regressor.py
    evaluate.py
  utils/
    logging.py
    seeds.py
    paths.py

configs/
  default_train.yaml
  default_summarization.yaml
  default_dataset_generation.yaml
```

## 7. Execution Pipeline

### Stage A: Article pulling

- Pull articles from the configured APIs.
- Respect the configured date range.
- Write incremental checkpoints during long runs.
- Persist raw article data to CSV.

### Stage B: Text preprocessing

- Clean article text.
- Summarize every article into a consistent short format.
- Cache summaries and metadata.

### Stage C: Embedding generation

- Tokenize summarized text with FinBERT tokenizer.
- Run encoder in frozen mode.
- Pool token embeddings into one vector per article.
- Persist article-level embeddings.

### Stage D: Daily aggregation

- Group article embeddings by day.
- Aggregate into one daily embedding.
- Generate `has_news` and article-count features.

### Stage E: Dataset building

- Join daily news features with daily market features.
- Build returns and technical indicators.
- Generate classification and regression targets.
- Build rolling windows.
- Persist processed datasets.

### Stage F: Training and evaluation

- Train classical baselines first.
- Train technical-indicator-only ablations.
- Train Transformer-based market-plus-news models as the primary experiment.
- Train LSTM-based market-plus-news models as comparison experiments.
- Evaluate all models on the same final holdout split.

## 8. V1 Defaults

These defaults should be encoded in YAML config files.

- News source: New York Times
- Primary raw storage: CSV
- Pull start date: October 2018
- Text encoder: FinBERT
- Encoder mode: frozen
- Summarization: enabled for every article
- Article pooling: mean pooling
- Daily aggregation: mean of article embeddings
- No-news handling: zero vector plus `has_news` indicator
- Market feature family: returns plus technical indicators
- Regression target: raw next-day close
- Classification target: next-day direction
- Primary downstream model: Transformer
- Secondary comparison model: LSTM
- Evaluation: one final chronological holdout
- Baselines: ARIMA, technical-indicator-only models, and ablations without BERT embeddings

## 9. Risks And Failure Modes

The architecture should explicitly account for these risks:

- news signal may be too weak for next-day prediction
- summarization may remove useful details
- raw price targets may generalize poorly
- article volume may vary heavily by day
- same-day assignment can create subtle cutoff issues if timestamps are inconsistent
- API limits can make naive repeated pulling expensive
- full summarization or full training runs can waste compute if early validation is skipped

## 10. Remaining Explicit Choices

These details should stay configurable or be confirmed before implementation expands:

1. Should no-news days always use both zero vectors and `has_news`, or should that be switchable?
2. What exact summary format should every article follow: paragraph, bullets, or structured fields?
3. Which technical indicators belong in the first default set?
4. What exact end date should the initial pull range use?
5. Should persisted artifacts stay CSV-only in v1, or should parquet/cache artifacts be added early?

## 11. Minimum Implementation Plan

### Phase 1: Config and ingestion

- implement YAML config loading
- implement dedicated NYT article pull script
- implement resumable article persistence
- implement market data ingestion

### Phase 2: Text pipeline

- implement text cleaning
- implement full-article summarization with checkpointing
- validate summary output on a small sample before full runs
- generate article-level FinBERT embeddings

### Phase 3: Feature pipeline

- aggregate article embeddings into daily embeddings
- compute returns and technical indicators
- join market and news features
- build processed datasets and rolling windows

### Phase 4: Modeling

- train ARIMA and technical-indicator-only baselines
- train ablations without BERT embeddings
- train sequence model with news embeddings

### Phase 5: Evaluation

- evaluate every model on the same final holdout
- compare baseline and text-augmented results
- analyze whether news adds incremental predictive value
