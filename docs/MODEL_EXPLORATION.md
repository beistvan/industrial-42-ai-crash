# Model Exploration — What to Try, What to Skip

Companion to `docs/TASK_BOARD.md` and `docs/LEONARDO_BATTLE_PLAN.md`. This is
the *should we try X?* discussion table — useful when you and your teammates
have 30 minutes between sweep stages and want to argue about priorities.

## Where we stand (as of 2026-05-30)

| Model | Top-1 | MRR | Top-5 | Task 2 tok-acc | Task 2 NED | Train time |
|---|---:|---:|---:|---:|---:|---:|
| n-gram, max_order=8, greedy | 0.687 | 0.807 | 0.928 | 0.428 | 0.221 | ~1s |
| n-gram, rule-constrained | 0.687 | 0.807 | 0.928 | 0.421 | 0.224 | ~1s |
| n-gram, beam-4 + rule-constrained | 0.687 | 0.807 | 0.928 | 0.408 | 0.228 | ~20s |
| Transformer, small, 1 epoch CPU (throwaway) | 0.08 | 0.10 | 0.15 | 0.03 | 0.99 | ~10min |
| Transformer, medium, 50ep on HPC | **unknown** | **unknown** | — | — | — | ~5h |

The Transformer must beat the n-gram's MRR (0.807) on the same dev split to
be worth submitting. Otherwise we ship the n-gram.

## Ideas, ranked by expected payoff

| Idea | Effort | Expected gain | Status |
|---|---|---|---|
| **Beam search on the Transformer's `complete()`** | implemented | Task 2 exact-match 0.005 → maybe 0.05–0.10 | opt-in via `beam_width=N` |
| **N-gram + Transformer ensemble at inference** | 2h | Task 1 Top-1 +1–3 pts | not started |
| **Family-token dropout at train time** | 30 min | Task 4 OOD: large unknown | not started |
| **Longer `max_len` (256 instead of 180)** | 30 min cfg | Marginal | not needed for current routes |
| **Kneser-Ney smoothing on the n-gram** | 1h | Task 1 MRR +1–2 pts | not started |
| **Synthetic data ratio sweep (1:1, 1:5, 1:10)** | 0 extra (in `leonardo_v1.yaml`) | Unknown | wired |
| **MLM pre-training on synthetic, then real fine-tune** | 6h | Marginal at this data scale | not recommended |
| **State-space models (Mamba)** | 8h+ | Unknown | not recommended for hackathon |
| **HF pretrained (distilgpt2, etc.)** | 30h | Likely negative | see ADR 0001 |

## Why each one matters (or doesn't)

### Beam search — implemented
The model already produces good per-step distributions; greedy compounds
errors. Beam keeps `K` hypotheses alive. Composes with rule-constrained
decoding. Empirically *hurts* the n-gram (sharp distribution, length
normalization explores worse paths) but expected to help the Transformer
where label smoothing softens the distribution. See ADR 0003.

### N-gram + Transformer ensemble
The n-gram is already strong on Task 1. Throwing it away when we ship the
Transformer leaves points on the table. Simple approach:

```python
def predict_topk_ensemble(family, prefix, k=5):
    suffix_order = ngram.matched_suffix_order(family, prefix)
    if suffix_order >= 5:               # n-gram has strong evidence
        return ngram.predict_topk(family, prefix, k)
    return merge_by_avg_rank(
        ngram.predict_topk(family, prefix, k),
        transformer.predict_topk(family, prefix, k),
        k,
    )
```

~50 LOC + a small helper to expose `matched_suffix_order` on the n-gram. Low
risk, +1–3 pts on Top-1 expected. Add if time after the demo page.

### Family-token dropout
Hackathon scoring includes a hidden 4th product family (Task 4 OOD). A
family-conditioned model trained always with the correct family token will
collapse on the unknown family. Randomly replacing `<fam:X>` with `<pad>`
during training (~10% of steps) teaches the model to also work without family
conditioning. 30 minutes of training-loop edit. Cheap insurance.

### Longer `max_len`
All real routes are < 180 steps; longer max_len only helps if we train on
long synthetic sequences. Skip unless we add multi-route concatenation
(unlikely in scope).

### Kneser-Ney smoothing
The current n-gram backoff is the crude "longest matching suffix wins". KN
smoothing redistributes probability mass more carefully (continuation
probability), giving better top-k for low-suffix-coverage prefixes. A drop-in
replacement in `ngram_baseline.py` — ~1h with tests. Worth doing IF the
Transformer disappoints and we ship the n-gram.

### Synthetic data ratio
Already wired into `configs/sweeps/leonardo_v1.yaml` as
`m_real_only` / `m_real_extras_1x` / `m_real_extras_2x`. The sweep will tell
us whether more synthetic helps; no manual decision needed.

### MLM pre-training + fine-tune
Two-stage training is overkill at 3,000-sequence data scale. Single-stage
training with label smoothing achieves the same regularization.

### State-space models (Mamba)
Architecture has merit for long sequences. Our sequences are 100–151 tokens —
well within Transformer attention's sweet spot. The debugging overhead is
not worth it.

### HF pretrained
See ADR 0001 for the full reasoning. Short version: vocabulary mismatch
(multi-word step tokens vs BPE subwords) and wrong priors (English vs fab
process logic).

## What we are NOT exploring (and why)

- **Pixel-level / image-based representations** (e.g. brain-tumor U-Net style).
  Not applicable — process steps are symbolic tokens, not images.
- **Multi-agent LLM workflows.** No. See `README.md::What NOT to do`.
- **RAG over generation_rules.md.** The validator already enforces the rules
  deterministically; adding a retrieval layer adds latency and surface area
  for bugs.
- **Active learning loops.** No label oracle available during the hackathon.

## Discussion prompts for tomorrow

1. If the medium Transformer comes back at MRR 0.85 (well above n-gram 0.81),
   do we still spend 2h on the ensemble? *Author's view: no — diminishing
   returns; spend the time on the demo page.*
2. If the medium Transformer comes back at MRR 0.75 (below n-gram), what's
   our plan? *Author's view: ship n-gram, finalize submission, then spend
   remaining time on the demo page to differentiate on UX rather than score.*
3. Should we add family-token dropout to one of the finalist runs as an
   ablation? *Author's view: yes if a slot is open, since Task 4 OOD is
   un-scoreable until the final reveal.*
