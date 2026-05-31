"""Tests for Task 3 anomaly scoring helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.eval.anomaly_scoring import (
    compose_anomaly_score,
    injection_label_matches_prediction,
    log_prob_to_validity_score,
    pick_primary_rule,
)


@dataclass
class _V:
    rule: str
    step_index: int


@pytest.mark.unit
def test_pick_primary_rule_earliest_step():
    violations = [
        _V("RULE_ETCH_NO_MASK", 40),
        _V("RULE_SHIP_BEFORE_TEST", 2),
    ]
    assert pick_primary_rule(violations) == "RULE_SHIP_BEFORE_TEST"


@pytest.mark.unit
def test_pick_primary_rule_tiebreak_by_official_order():
    violations = [
        _V("RULE_CMP_NO_DEP", 10),
        _V("RULE_DEP_NO_CLEAN", 10),
    ]
    assert pick_primary_rule(violations) == "RULE_DEP_NO_CLEAN"


@pytest.mark.unit
def test_injection_label_named_rule():
    assert injection_label_matches_prediction(
        "RULE_ETCH_NO_MASK_drop_develop",
        ["RULE_ETCH_NO_MASK", "RULE_DEP_NO_CLEAN"],
        "RULE_ETCH_NO_MASK",
    )


@pytest.mark.unit
def test_injection_label_shuffle_middle():
    assert injection_label_matches_prediction(
        "shuffle_middle",
        ["RULE_LITHO_LEVEL_SKIP"],
        "RULE_LITHO_LEVEL_SKIP",
    )
    assert not injection_label_matches_prediction("shuffle_middle", [], None)


@pytest.mark.unit
def test_log_prob_to_validity_score_monotone():
    low = log_prob_to_validity_score(-200.0, 100)
    high = log_prob_to_validity_score(-20.0, 100)
    assert 0.0 <= low < high <= 1.0


@pytest.mark.unit
def test_compose_anomaly_score_invalid_caps_low():
    score = compose_anomaly_score(valid=False, rules=["R1", "R2"], lm_score=0.9)
    assert score <= 0.8
