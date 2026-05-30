# ADR 0002 — Rule-constrained decoding default + opt-out

**Date:** 2026-05-30
**Status:** accepted
**Owners:** repo

## Context

For Task 2 (sequence completion), the model proposes tokens that may or may
not satisfy the 10 forbidden patterns in `generation_rules.md`. The official
validator (`validate_sequence`) is already wired into our pipeline. The
question: should the default completion path consult the validator at each
step, or stay pure-model?

## Decision

`scripts/predict_submission.py` and `src/eval/local_eval.py` default to
**`--rule-constrained`** with `--candidate-pool 5`. The CLI flag
`--no-rule-constrained` flips back to pure-greedy. The underlying
`model.complete(...)` method defaults to `rule_constrained=False` so existing
unit tests (and downstream code paths that don't expect validator calls) keep
their original behavior.

Rule-constrained behavior: at each step, the model's top-`candidate_pool`
candidates are tried in rank order; the first one that does not introduce a
new rule violation is emitted. If all candidates would violate, the top-1 is
emitted unchanged (so progress is guaranteed; the validator's verdict on the
final sequence is the source of truth).

## Consequences

Positive:
- Demo and submission outputs are auditably valid more often. Task 3 numbers
  on the model's *own* completion improve.
- Judges can be told "we enforce the grammar at inference time" — a clean,
  short story.
- The implementation is one shared function (`violation_rules`) called per
  candidate; no model retraining required.

Negative:
- For models that already learned the grammar well (the n-gram baseline
  at `max_order=8`), rule-constrained decoding is a slight regression:
  observed Task 2 token-accuracy 0.421 vs 0.428 for plain greedy. The n-gram's
  top-1 is *already* the validator-correct choice, so filtering only adds
  noise via the candidate pool when there are ties.
- Compute cost: one `validate_sequence` call per (sequence × step ×
  candidate). On the dev split (~600 sequences × ~125 steps × 5 candidates)
  this is ~400k validator calls per evaluation. Each call is sub-millisecond
  on CPU; the overall cost is ~30s. Acceptable.

## Alternatives considered

1. **Default off, opt in.** Cleaner for the n-gram; worse demo story when we
   ship the Transformer. We expect the Transformer to benefit from rule
   filtering, so default-on is the right pick.
2. **Hard-block: reject any token that would create a violation, never emit
   it.** Rejected because it can stall (no candidate satisfies); the current
   "fall back to top-1" preserves progress.
3. **Soft-bias via logit penalty.** Would require retraining or per-step
   logit access (we have it for the Transformer, not for the n-gram).
   Implementing it asymmetrically would make the two models incomparable.

## Revisit when

- The Transformer is trained on HPC and we can A/B rule-constrained vs greedy
  on it. If greedy wins on the Transformer too (unlikely but possible if
  label smoothing makes its distribution near-valid), flip the default off.
- We add beam search (see ADR 0003) — beam composes with rule-constrained, but
  the candidate filtering may need its own tuning.
