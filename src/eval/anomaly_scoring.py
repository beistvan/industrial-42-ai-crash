"""Task 3 helpers: rule attribution and LM-based validity SCORE."""
from __future__ import annotations

import math
from typing import Any, Sequence

from src.data.infineon_loader import Family

# Order rules are checked in generate_sequences.validate_sequence (tie-break).
OFFICIAL_RULE_ORDER: tuple[str, ...] = (
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
)

_RULE_RANK = {name: i for i, name in enumerate(OFFICIAL_RULE_ORDER)}


def pick_primary_rule(violations: Sequence[Any]) -> str | None:
    """Return the rule ID to emit as PREDICTED_RULE.

    Chooses the violation with the smallest ``step_index``, breaking ties using
    ``OFFICIAL_RULE_ORDER`` so multi-violation sequences match the organizer's
    rule taxonomy when several fire at the same step.
    """
    if not violations:
        return None
    best_step = min(getattr(v, "step_index", 10**9) for v in violations)
    candidates = [
        v.rule for v in violations
        if getattr(v, "step_index", 10**9) == best_step
    ]
    candidates.sort(key=lambda r: (_RULE_RANK.get(r, len(_RULE_RANK)), r))
    return candidates[0]


def injection_label_matches_prediction(
    injected_label: str,
    predicted_rules: Sequence[str],
    primary_rule: str | None,
) -> bool:
    """Dev-eval matching between synthetic injection labels and validator output."""
    if not injected_label:
        return False
    if injected_label == "shuffle_middle":
        # Synthetic multi-rule mutation — any attributed rule counts.
        return bool(predicted_rules)
    for rule in predicted_rules:
        if injected_label.startswith(rule):
            return True
    return primary_rule is not None and injected_label.startswith(primary_rule)


def step_log_prob(model: Any, family: Family, prefix: list[str], token: str) -> float:
    """Log-probability of ``token`` as the next step after ``prefix``."""
    if hasattr(model, "step_log_prob"):
        return float(model.step_log_prob(family, prefix, token))
    for tok, lp in model.predict_topk_with_scores(family, prefix, k=500):
        if tok == token:
            return float(lp)
    return float("-inf")


def sequence_log_prob(model: Any, family: Family, steps: list[str]) -> float:
    """Teacher-forced sum of log-probs for each step after the first."""
    if len(steps) <= 1:
        return 0.0
    if hasattr(model, "sequence_log_prob"):
        return float(model.sequence_log_prob(family, steps))
    total = 0.0
    for i in range(1, len(steps)):
        total += step_log_prob(model, family, steps[:i], steps[i])
    return total


def log_prob_to_validity_score(total_log_prob: float, n_steps: int) -> float:
    """Map cumulative log-prob to P(valid) in [0, 1] for judge SCORE / AUC."""
    if n_steps <= 1:
        return 0.5
    avg = total_log_prob / (n_steps - 1)
    return max(0.0, min(1.0, math.exp(avg)))


def compose_anomaly_score(
    *,
    valid: bool,
    rules: Sequence[str],
    lm_score: float,
    max_rules: int = 10,
) -> float:
    """Combine LM plausibility with rule-validator penalty for invalid rows."""
    if valid:
        return lm_score
    rule_penalty = max(0.0, 1.0 - min(len(rules), max_rules) / max_rules)
    return min(lm_score, rule_penalty)
