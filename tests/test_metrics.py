"""Tests for the metrics module."""
from __future__ import annotations

import pytest

from src.eval.metrics import (
    exact_match,
    mean_reciprocal_rank,
    normalized_edit_distance,
    token_accuracy,
    top_k_accuracy,
)


@pytest.mark.unit
def test_top_k_accuracy():
    preds = [["A", "B", "C"], ["X", "Y", "Z"], ["B", "A", "C"]]
    gold = ["A", "Y", "C"]
    assert top_k_accuracy(preds, gold, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(preds, gold, 2) == pytest.approx(2 / 3)
    assert top_k_accuracy(preds, gold, 3) == pytest.approx(1.0)


@pytest.mark.unit
def test_mean_reciprocal_rank():
    preds = [["A", "B", "C"], ["X", "Y", "Z"], ["A", "B", "C"]]
    gold = ["A", "Z", "MISSING"]
    expected = (1.0 + 1 / 3 + 0.0) / 3
    assert mean_reciprocal_rank(preds, gold) == pytest.approx(expected)


@pytest.mark.unit
def test_exact_match_and_token_accuracy():
    assert exact_match(["A", "B"], ["A", "B"]) is True
    assert exact_match(["A", "B"], ["A", "C"]) is False
    assert token_accuracy(["A", "B", "C"], ["A", "X", "C"]) == pytest.approx(2 / 3)
    assert token_accuracy([], ["A"]) == 0.0


@pytest.mark.unit
def test_normalized_edit_distance():
    assert normalized_edit_distance(["A", "B"], ["A", "B"]) == 0.0
    # one substitution out of two -> 0.5
    assert normalized_edit_distance(["A", "B"], ["A", "C"]) == pytest.approx(0.5)
    # full mismatch
    assert normalized_edit_distance(["A", "B"], ["C", "D"]) == 1.0
    # empty vs non-empty
    assert normalized_edit_distance([], ["A"]) == 1.0
