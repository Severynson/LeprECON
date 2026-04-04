"""DistilBERT wrapper for binary economy-relevance classification."""

from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
)


MODEL_ID = "distilbert-base-uncased"


def load_model(
    num_labels: int = 2,
    pretrained: str = MODEL_ID,
    device: str | torch.device = "cpu",
    use_fast_tokenizer: bool = False,
    local_files_only: bool = False,
) -> tuple[DistilBertForSequenceClassification, AutoTokenizer]:
    """Load DistilBERT model and tokenizer.

    Args:
        num_labels: Number of classification labels (default 2 for binary).
        pretrained: HuggingFace model identifier.
        device: torch device.
        use_fast_tokenizer: Whether to use Rust "fast" tokenizer implementation.
        local_files_only: If True, do not attempt any network downloads.

    Returns:
        (model, tokenizer) tuple.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=num_labels,
        local_files_only=local_files_only,
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained,
        use_fast=use_fast_tokenizer,
        local_files_only=local_files_only,
    )
    return model, tokenizer


def encode_texts(
    texts: list[str],
    model: DistilBertForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str | torch.device = "cpu",
    max_length: int = 512,
) -> torch.Tensor:
    """Encode texts and return model logits (no grad).

    Args:
        texts: List of input texts.
        model: Loaded DistilBERT model.
        tokenizer: Model tokenizer.
        device: torch device.
        max_length: Max sequence length for tokenizer.

    Returns:
        Logits tensor of shape (batch_size, num_labels).
    """
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
    return outputs.logits
