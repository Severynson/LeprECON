from typing import List, Union

import torch
from transformers import BertTokenizer, BertForSequenceClassification


MODEL_NAME = "ProsusAI/finbert"


def load_finbert(
    use_head: bool = False,
    device: str | None = None,
) -> tuple[BertForSequenceClassification, BertTokenizer, torch.device]:
    """Load ProsusAI/finbert and its tokenizer.

    Args:
        use_head: If True the returned model keeps its classification head
            active (forward returns logits).  If False (default) the model
            is patched so that forward returns the 768-dim pooled hidden
            state **before** the classifier, which is the feature vector
            suitable for downstream storage and time-series construction.
        device: Force a specific device string (e.g. "cpu", "cuda:0").
            When None the function picks CUDA if available, else CPU.

    Returns:
        (model, tokenizer, device)
    """
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    if not use_head:
        # Replace the classifier head with an identity so that the
        # pooled hidden state flows through unchanged.
        model.classifier = torch.nn.Identity()

    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def encode_texts(
    texts: Union[str, List[str]],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode one or more texts into feature vectors (or logits if head is kept).

    When the model was loaded with ``use_head=False`` (default), each text
    produces a 768-dimensional feature vector — the same representation on
    which the original classification head was trained.

    Args:
        texts: A single string or a list of strings.
        model: FinBERT model returned by :func:`load_finbert`.
        tokenizer: Tokenizer returned by :func:`load_finbert`.
        device: Device returned by :func:`load_finbert`.
        max_length: Maximum token length for the tokenizer.

    Returns:
        Tensor of shape ``(n_texts, hidden_dim)`` where *hidden_dim* is 768
        when the head is detached, or 3 (positive/negative/neutral logits)
        when the head is kept.
    """
    if isinstance(texts, str):
        texts = [texts]

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)

    # .logits holds either the real logits (use_head=True) or the
    # identity-passed pooled hidden state (use_head=False).
    return outputs.logits
