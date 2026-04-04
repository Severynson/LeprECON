# BERT Stock Dynamic Estimator

## Goal

This project explores whether daily macro and financial news can improve short-horizon prediction of the S&P 500 when combined with market time-series features.

The core idea is:

1. Pull and persist historical news articles for a fixed date range.
2. Summarize every article into a consistent short format.
3. Encode summarized articles with FinBERT into dense semantic representations.
4. Aggregate article embeddings into one daily news representation.
5. Combine daily news features with returns and technical indicators.
6. Train downstream models to predict next-day market direction and raw next-day close.

The project is meant to be a reproducible research pipeline, not just a single training script. The main objective is to measure whether news-derived embeddings add signal beyond technical-indicator-only baselines.

## High-Level Idea

The system is split into two stages:

- Language stage: summarize and encode each day's news into a fixed-size vector representation.
- Forecasting stage: feed a rolling sequence of daily market features plus news embeddings into a Transformer-based time-series model, with LSTM kept as a comparison model.

This separation keeps the project modular and makes it easier to compare:

- different news sources
- different summarization settings
- different encoders
- different aggregation strategies
- different forecasting models

## Planned Scope

Initial scope:

- Use New York Times as the first news source.
- Pull and save source articles once, then reuse local files.
- Use FinBERT as the default encoder in frozen mode.
- Summarize every article for consistent input format.
- Start with technical indicators plus news embeddings.
- Predict next-day direction and raw next-day close.
- Use a Transformer-based downstream forecaster as the primary model, and compare it against technical-indicator-only, classical, and LSTM baselines.

Later scope:

- Add more news sources through configuration.
- Add more macro features such as employment rate and GDP.
- Compare frozen embeddings against fine-tuned encoder variants.
- Compare multiple summarization strategies and prompt formats.

## What Success Looks Like

The project should produce:

- A reproducible dataset build pipeline with cached article pulls and preprocessing artifacts.
- A leakage-aware training and evaluation pipeline.
- Baseline models and text-augmented models trained on the same chronological split.
- Clear experiment outputs showing whether news embeddings improve forecasting quality.

## Repo Documents

- [ARCHITECTURE.md](/Users/severynkurach/Desktop/Programming/BERT-stock-dynamic-estimator/ARCHITECTURE.md): original rough draft.
- [ARCHITECTURE_UPDATED.md](/Users/severynkurach/Desktop/Programming/BERT-stock-dynamic-estimator/ARCHITECTURE_UPDATED.md): technical architecture, data contracts, and implementation plan.
- [IMPLEMENTATION_GUIDELINES.md](/Users/severynkurach/Desktop/Programming/BERT-stock-dynamic-estimator/IMPLEMENTATION_GUIDELINES.md): testing policy, rollout rules, and cost-control constraints.
