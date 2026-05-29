"""Scoring metrics matching the official Infineon eval protocol.

Task 1 (next-step):    Top-1 / Top-3 / Top-5 accuracy, MRR.
Task 2 (completion):   Exact match, token accuracy, normalized edit distance.
Task 3 (anomaly):      Binary precision / recall / F1 (computed downstream
                       from rule_validator outputs).
"""
from __future__ import annotations

from typing import Sequence


def top_k_accuracy(predictions: Sequence[Sequence[str]], gold: Sequence[str], k: int) -> float:
    """Fraction of items where the gold token is in the top-k predictions."""
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold must have the same length")
    if not gold:
        return 0.0
    hits = sum(1 for preds, g in zip(predictions, gold) if g in list(preds)[:k])
    return hits / len(gold)


def mean_reciprocal_rank(predictions: Sequence[Sequence[str]], gold: Sequence[str]) -> float:
    """MRR: 1/rank of the gold token, or 0 if absent."""
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold must have the same length")
    if not gold:
        return 0.0
    total = 0.0
    for preds, g in zip(predictions, gold):
        try:
            rank = list(preds).index(g) + 1
            total += 1.0 / rank
        except ValueError:
            pass
    return total / len(gold)


def exact_match(prediction: Sequence[str], gold: Sequence[str]) -> bool:
    return list(prediction) == list(gold)


def token_accuracy(prediction: Sequence[str], gold: Sequence[str]) -> float:
    """Position-aligned token accuracy on the overlapping prefix."""
    if not gold:
        return 0.0
    n = min(len(prediction), len(gold))
    if n == 0:
        return 0.0
    hits = sum(1 for i in range(n) if prediction[i] == gold[i])
    return hits / len(gold)


def normalized_edit_distance(prediction: Sequence[str], gold: Sequence[str]) -> float:
    """Levenshtein distance over tokens, normalized by max(len(pred), len(gold))."""
    a, b = list(prediction), list(gold)
    if not a and not b:
        return 0.0
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 1.0
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[n] / max(m, n)
