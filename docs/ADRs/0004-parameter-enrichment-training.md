# ADR 0004: Parameter-enrichment training (Wave 5)

## Status

Accepted — scaffolded; run only if Wave 3 (modern arch) and Wave 4 (Task-2 prefix
training) do not beat current dev bars.

## Context

Track 1 stretch goal: *“inclusion of optional process parameters”*. Official
training CSVs are STEP-only; reference files add DESCRIPTION and fab-level
PARAMETERS per canonical step name (`reference/*_parameters.csv`).

Waves 1–4 train on STEP tokens only. Wave 3/4 target architecture and Task-2
training recipe. If those plateau (~MRR 0.873 / tok 0.455), Wave 5 adds
parameter metadata as **training-only auxiliary tokens**.

## Decision

1. **After-step auxiliary tokens** — after each STEP with known parameters,
   append one deterministic token `<p:{sha1(family|step)[:10]}>`.
2. **Training only** — dev eval, Streamlit demo, and judge submission stay
   STEP-only (same as organizer eval inputs).
3. **Inference masking** — block param-token ids in `predict_topk` / beam search
   so the model cannot emit aux tokens in CSV outputs.
4. **Vocab** — param tokens are added to the embedding table via the normal
   `Vocabulary.from_sequences` path (enriched train+dev for id stability).
5. **max_len** — Wave 5 configs use `max_len: 256` (enriched streams are longer).

## Alternatives considered

| Option | Why not (for now) |
|---|---|
| Concatenate param text into STEP string | Explodes vocab; breaks step identity |
| Separate param encoder / cross-attention | Too much scope for hackathon fallback wave |
| Train on reference long-description CSVs only | Different corpus size; variants are the scored train set |

## Consequences

- Checkpoints carry `param_enrichment: after_step` in metadata.
- Leaderboard runs prefixed `p5_*`.
- If Wave 5 wins, regenerate submission with `make regenerate-submission` as usual.
- If params do not help, negative result is still valid Track 1 stretch evidence.

## Trigger

```bash
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_params.yaml --stage finalists --row 0
```

Bars (dev holdout, as of Wave 2):

- Task 1 MRR ≥ **0.8731** (`f_drop15_100_mrr`)
- Task 2 tok-acc ≥ **0.4545** (`g_drop15_nosched_t2`)
