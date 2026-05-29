"""Thin wrapper around the official Infineon rule validator.

The official `validate_sequence(steps) -> list[Violation]` lives in
`data/raw/infineon/training_data/generate_sequences.py`. We import it via
file path so that:

  * Task 3 (anomaly detection) uses the *exact* same rule set the organizers
    will use, with zero risk of divergence from a re-implementation.
  * If the upstream file is refreshed mid-hackathon, the wrapper picks up the
    change automatically.
"""
from __future__ import annotations

from typing import Sequence

from src.data.infineon_loader import _load_official_module


def _validate(steps: Sequence[str]):
    mod = _load_official_module()
    return mod.validate_sequence(list(steps))


def is_valid_sequence(steps: Sequence[str]) -> bool:
    """True iff the sequence has zero rule violations."""
    return not _validate(steps)


def violation_rules(steps: Sequence[str]) -> list[str]:
    """Distinct rule names triggered by the sequence (order of first hit)."""
    seen: list[str] = []
    for v in _validate(steps):
        if v.rule not in seen:
            seen.append(v.rule)
    return seen


def classify_sequence(steps: Sequence[str]) -> dict:
    """Return a structured classification result suitable for Task 3 outputs."""
    rules = violation_rules(steps)
    return {
        "valid": not rules,
        "rules": rules,
        "primary_rule": rules[0] if rules else None,
    }
