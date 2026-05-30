# ADR 0003 — Beam search is opt-in, not default

**Date:** 2026-05-30
**Status:** accepted
**Owners:** repo

## Context

Task 2 (60% / 80% prefix → SHIP LOT) is graded by exact match, token accuracy,
and normalized edit distance over the completion-only portion. Greedy decoding
compounds errors — a single misstep cascades. Beam search is the textbook
fix: keep `K` hypotheses alive, score by cumulative log-prob, pick the best
at the end. We scaffolded it now so it's available the moment a trained
Transformer lands from Leonardo.

## Decision

`NGramBaseline.complete(...)` and `TransformerProcessModel.complete(...)` both
accept `beam_width: int = 1` and `length_normalize: bool = True`. Default
behavior is unchanged (greedy). `beam_width > 1` dispatches to the shared
`src/ml/beam_search.py::_beam_complete` helper. Composes with
`rule_constrained=True` — rule-filtering runs per beam expansion.

Both models expose a new `predict_topk_with_scores(family, prefix, k)` that
returns `[(token, log_prob), ...]`. For the n-gram this is `log(count/total)`
from the matched suffix (or unconditional backoff). For the Transformer this
is `log_softmax(logits)[token_id]`, with specials masked out.

## Consequences

Positive:
- Both submission script and evaluator accept `--beam-width N`. Tomorrow's
  HPC sweep can A/B Transformer + beam{1,4,8} without code changes.
- Length-normalized scoring is on by default (`length_normalize=True`), so the
  beam doesn't bias toward short completions that exit early.
- Implementation is model-agnostic: any future model that implements
  `predict_topk_with_scores` gets beam search for free.

Negative:
- For the **n-gram baseline**, beam-4 *underperforms* greedy on the dev split:
  observed Task 2 token-accuracy 0.408 (beam-4 + rule-constrained) vs 0.428
  (greedy + rule-constrained). The n-gram's distribution is sharp; beam
  search with length normalization explores plausible-looking alternative
  paths that are actually worse. This is why the default is `beam_width=1`.
- Each beam expansion runs `validate_sequence` on `candidate_pool` × `beam_width`
  candidates when rule-constrained — quadratic in `beam_width`. Keep
  `beam_width ≤ 8` for HPC budget.

## Alternatives considered

1. **No length normalization.** Tested informally; beam consistently terminated
   too early via a high-prob short path that didn't reach SHIP LOT. Length
   normalization on by default was the obvious choice.
2. **Diverse beam search.** Useful when beams collapse to near-duplicates.
   Not implemented — for a 120-token vocabulary with sharp distributions,
   diversity penalty is probably noise. Add if we see beam-4 = beam-8.
3. **Sampling (nucleus/temperature) instead of beam.** Wrong objective for
   exact-match grading. Could be useful for the demo, not for the score.
4. **Constrained beam with the rule validator as a strict per-step filter
   inside the beam expansion** (don't admit a candidate if it would violate).
   This is what we implemented when `rule_constrained=True` + `beam_width>1`.

## Revisit when

- Transformer trained on HPC is available. Re-run A/B: greedy / beam-4 /
  beam-8, with and without rule_constrained. The expectation is that beam will
  help the Transformer where it hurt the n-gram (smoother distribution from
  label smoothing + larger effective context).
- Task 2 exact-match rate is still under 0.05 after beam search on the
  Transformer. That signals we need diverse beam or a structured-prediction
  decoder; both are larger changes.
