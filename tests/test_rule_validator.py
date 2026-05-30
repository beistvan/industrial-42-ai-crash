"""Tests for the rule validator wrapper."""
from __future__ import annotations

import pytest

from src.data import load_family
from src.eval.rule_validator import classify_sequence, is_valid_sequence, violation_rules


@pytest.mark.integration
def test_real_training_sequences_are_valid():
    """All 1,000 MOSFET training sequences should pass the validator."""
    seqs = load_family("mosfet")
    for sid, steps in seqs.items():
        if not is_valid_sequence(steps):
            pytest.fail(f"{sid} unexpectedly invalid: {violation_rules(steps)[:3]}")


@pytest.mark.unit
def test_obviously_broken_sequence_is_invalid():
    # SHIP LOT before WAFER SORT TEST is a documented rule violation.
    bad = ["RECEIVE WAFER LOT", "SHIP LOT", "WAFER SORT TEST"]
    assert is_valid_sequence(bad) is False
    rules = violation_rules(bad)
    assert any(r.startswith("RULE_") for r in rules)


@pytest.mark.unit
def test_classify_sequence_shape():
    res = classify_sequence(["RECEIVE WAFER LOT", "SHIP LOT", "WAFER SORT TEST"])
    assert set(res.keys()) == {"valid", "rules", "primary_rule"}
    assert res["valid"] is False
    assert res["primary_rule"] is not None


@pytest.mark.unit
def test_primary_rule_is_earliest_step_index():
    """Rules should be returned earliest-firing first, regardless of the order
    the official validator emits them (which can be checker-order, not
    sequence-order)."""
    from src.eval.rule_validator import _validate

    bad = ["RECEIVE WAFER LOT", "SHIP LOT", "WAFER SORT TEST"]
    violations = list(_validate(bad))
    if not violations:
        pytest.skip("Validator emitted no violations for the canonical bad sequence.")
    earliest_step = min(getattr(v, "step_index", 10**9) for v in violations)
    earliest_rules = {
        v.rule for v in violations
        if getattr(v, "step_index", 10**9) == earliest_step
    }
    res = classify_sequence(bad)
    assert res["primary_rule"] in earliest_rules
