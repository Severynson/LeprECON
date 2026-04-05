"""Tests for DistilBERT fine-tuning pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add src to path so we can import the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_generation import fine_tune_distilbert as fine_tune

# Imports
normalize_text = fine_tune.normalize_text
build_dedup_key = fine_tune.build_dedup_key
build_feature_text = fine_tune.build_feature_text
balance_dataset = fine_tune.balance_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _article(**overrides: str) -> dict[str, str]:
    base = {
        "article_id": "nyt://article/1",
        "source": "new_york_times",
        "query": "",
        "article_day": "2023-01-10",
        "published_at": "2023-01-10T08:00:00+0000",
        "headline": "Headline",
        "article_text": "Article body text",
        "snippet": "Short snippet",
        "abstract": "Abstract",
        "lead_paragraph": "Lead para",
        "section_name": "Business",
        "subsection_name": "",
        "web_url": "https://example.com",
        "keywords": "keyword1, keyword2",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Feature text construction
# ---------------------------------------------------------------------------

def test_build_feature_text_concatenates_fields():
    article = _article()
    text = build_feature_text(article, max_chars=100)
    assert "Headline" in text
    assert "Article body text" in text
    assert "Business" in text
    assert "[SEP]" in text


def test_build_feature_text_skips_empty_fields():
    article = _article(snippet="", abstract="", lead_paragraph="")
    text = build_feature_text(article, max_chars=100)
    assert "Headline" in text
    assert "Article body text" in text
    assert "Business" in text
    # Should have fewer [SEP] tokens
    assert text.count("[SEP]") <= 2


def test_build_feature_text_truncates_article_text():
    article = _article(article_text="A" * 5000)
    text = build_feature_text(article, max_chars=100)
    # article_text should be truncated
    assert len(text) < 5000


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------

def test_balance_dataset_undersamples_majority():
    # 20 label=0, 5 label=1 → balanced to 5 each
    texts = ["text_0"] * 20 + ["text_1"] * 5
    labels = [0] * 20 + [1] * 5

    balanced_texts, balanced_labels = balance_dataset(texts, labels, seed=42)

    count_0 = sum(1 for l in balanced_labels if l == 0)
    count_1 = sum(1 for l in balanced_labels if l == 1)

    assert count_0 == count_1 == 5


def test_balance_dataset_preserves_minority_all():
    # 20 label=0, 5 label=1 → balanced to 5 each, all label=1 preserved
    texts = ["text_0"] * 20 + ["text_1"] * 5
    labels = [0] * 20 + [1] * 5

    balanced_texts, balanced_labels = balance_dataset(texts, labels, seed=42)

    count_1_before = 5
    count_1_after = sum(1 for l in balanced_labels if l == 1)

    assert count_1_after == count_1_before


def test_balance_dataset_already_balanced():
    texts = ["text_0"] * 10 + ["text_1"] * 10
    labels = [0] * 10 + [1] * 10

    balanced_texts, balanced_labels = balance_dataset(texts, labels, seed=42)

    assert len(balanced_texts) == 20
    assert sum(1 for l in balanced_labels if l == 0) == 10
    assert sum(1 for l in balanced_labels if l == 1) == 10


# ---------------------------------------------------------------------------
# Content key matching
# ---------------------------------------------------------------------------

def test_build_dedup_key_uses_article_id():
    article = _article(article_id="nyt://article/xyz")
    key = build_dedup_key(article)
    assert key == "id:nyt://article/xyz"


def test_build_dedup_key_fallback_to_url():
    article = _article(article_id="", web_url="https://example.com/page")
    key = build_dedup_key(article)
    assert key == "url:https://example.com/page"


def test_build_dedup_key_fallback_to_published_headline():
    article = _article(
        article_id="",
        web_url="",
        published_at="2023-01-10T08:00:00Z",
        headline="Headline",
    )
    key = build_dedup_key(article)
    assert key.startswith("published_headline:")
    assert "2023-01-10T08:00:00Z" in key
    assert "Headline" in key
