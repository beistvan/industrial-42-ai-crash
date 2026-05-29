"""Model loading helpers shared by evaluation and submission scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ml.ngram_baseline import NGramBaseline
from src.ml.transformer_model import TransformerProcessModel


def load_sequence_model(path: Path, *, device: str | None = None) -> Any:
    """Load either an n-gram pickle or a Transformer checkpoint.

    Supported formats:
      - .pkl/.pickle: NGramBaseline
      - anything else: TransformerProcessModel checkpoint written by
        scripts/train_transformer.py
    """
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return NGramBaseline.load(path)
    return TransformerProcessModel.load(path, device=device)
