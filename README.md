# BERT Stock Dynamic Estimator

## Goal

This project explores whether daily macro and financial news can improve short-horizon prediction of the S&P 500 when combined with market time-series features.

The core idea is:

1. Pull and persist historical news articles for a fixed date range.
2. **Filter articles by economy/market relevance** (using Gemini or fine-tuned DistilBERT).
3. Summarize every article into a consistent short format.
4. Encode summarized articles with FinBERT into dense semantic representations.
5. Aggregate article embeddings into one daily news representation.
6. Combine daily news features with returns and technical indicators.
7. Train downstream models to predict next-day market direction and raw next-day close.

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

## Article Classification Strategy

The pipeline offers two methods to classify articles as economically relevant:

### 1. **Gemini API** (Reference/Ground Truth)
Use Google's Gemini LLM for high-quality labels. Best when you have a limited dataset or need baseline labels for training.

```bash
# Run with Gemini (using cached labels or real API calls)
python src/dataset_generation/preprocessing.py configs/default_preprocessing.yaml
```

**Pros:** High accuracy, useful as ground truth  
**Cons:** API cost (~$0.05–$0.10 per 1M input tokens), slow for large datasets

### 2. **Fine-Tuned DistilBERT** (Fast Inference)
Train DistilBERT on Gemini-labeled data, then use it to label remaining articles cheaply and quickly.

```bash
# Step 1: Fine-tune DistilBERT on Gemini labels (one-time)
python src/dataset_generation/fine_tune_distilbert.py configs/default_distilbert_finetune.yaml

# Step 2: Use fine-tuned model to label remaining articles
python src/dataset_generation/label_with_distilbert.py configs/default_distilbert_inference.yaml
```

**Pros:** Fast inference (~10ms/article), no API costs, works on unlabeled datasets  
**Cons:** Quality depends on training set, one-time training required

**Recommended workflow:**
1. Use Gemini to label a representative sample (~10k–50k articles)
2. Fine-tune DistilBERT on Gemini's labels (training takes ~5–15 minutes on CPU)
3. Use DistilBERT to label the rest of your dataset at virtually no cost
4. Cache both sets of labels separately for reproducibility

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
