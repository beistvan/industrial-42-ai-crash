"""Beam-search completion shared by NGramBaseline and TransformerProcessModel.

Both models expose `predict_topk_with_scores(family, prefix, k)` which returns
`[(token, log_prob), ...]`. This module is a model-agnostic beam expander
that composes with the official rule validator.

Algorithm:

    beam_t = [(seq_i, cum_log_prob_i, finished_i)]   length = beam_width
    while not all_finished and steps < max_steps:
        candidates = []
        for (seq, cum_lp, finished) in beam_t:
            if finished: candidates += [(seq, cum_lp, True)]
            else:
                for (tok, lp) in topk_with_scores(seq, k=candidate_pool):
                    if rule_constrained and adds_new_violation(seq, tok):
                        continue
                    candidates += [(seq + [tok], cum_lp + lp, tok == stop_token)]
        beam_t = top-K candidates by length-normalized cum_lp
    return best beam by length-normalized cum_lp

Length normalization (`cum_lp / len(seq_minus_prefix)`) prevents the beam from
preferring short completions that bail before SHIP LOT.
"""
from __future__ import annotations

from typing import Callable

from src.data.infineon_loader import Family

ScorerFn = Callable[[Family, list[str], int], list[tuple[str, float]]]


def _adds_new_violation(seq: list[str], tok: str, base_violations: set[str]) -> bool:
    """True iff appending `tok` to `seq` introduces a rule not already in `base_violations`."""
    from src.eval.rule_validator import violation_rules
    trial = set(violation_rules(seq + [tok]))
    return bool(trial - base_violations)


def _beam_complete(
    scorer: ScorerFn,
    family: Family,
    prefix: list[str],
    *,
    max_steps: int,
    stop_token: str,
    rule_constrained: bool,
    candidate_pool: int,
    beam_width: int,
    length_normalize: bool,
) -> list[str]:
    """Run beam search using the model's `predict_topk_with_scores` callable."""
    if beam_width <= 1:
        # Cheap fallback to greedy — caller probably should not have invoked us.
        return list(prefix)

    from src.eval.rule_validator import violation_rules

    initial_violations: set[str] = set(violation_rules(prefix)) if rule_constrained else set()
    prefix_len = len(prefix)

    # Each beam entry: (sequence, cumulative_log_prob, finished, base_violations).
    beam: list[tuple[list[str], float, bool, set[str]]] = [
        (list(prefix), 0.0, False, initial_violations),
    ]

    def normalized(cum_lp: float, seq: list[str]) -> float:
        if not length_normalize:
            return cum_lp
        # Penalize by generated length only — keep prefix out so different
        # prefixes are comparable.
        generated = max(1, len(seq) - prefix_len)
        return cum_lp / generated

    for _ in range(max_steps):
        if all(finished for _, _, finished, _ in beam):
            break

        candidates: list[tuple[list[str], float, bool, set[str]]] = []
        for seq, cum_lp, finished, base_v in beam:
            if finished:
                candidates.append((seq, cum_lp, True, base_v))
                continue

            scored = scorer(family, seq, max(candidate_pool, beam_width))
            for tok, log_prob in scored:
                if rule_constrained and _adds_new_violation(seq, tok, base_v):
                    continue
                new_seq = seq + [tok]
                new_finished = tok == stop_token
                new_base_v = (
                    set(violation_rules(new_seq)) if rule_constrained else base_v
                )
                candidates.append((new_seq, cum_lp + log_prob, new_finished, new_base_v))

            # Safety net: if every candidate got filtered out (all violate),
            # still extend with the model's top-1 to make progress. This mirrors
            # the rule-constrained-greedy fallback.
            if rule_constrained and not any(
                c[0][-1] != seq[-1] or c[1] != cum_lp for c in candidates if c[0][:-1] == seq
            ) and scored:
                tok, log_prob = scored[0]
                new_seq = seq + [tok]
                new_finished = tok == stop_token
                candidates.append(
                    (new_seq, cum_lp + log_prob, new_finished, set(violation_rules(new_seq)))
                )

        if not candidates:
            break

        candidates.sort(key=lambda c: normalized(c[1], c[0]), reverse=True)
        beam = candidates[:beam_width]

    # Pick best by length-normalized score.
    best = max(beam, key=lambda c: normalized(c[1], c[0]))
    return best[0]
