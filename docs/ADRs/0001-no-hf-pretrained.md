# ADR 0001 — No HuggingFace pretrained model

**Date:** 2026-05-30
**Status:** accepted
**Owners:** repo

## Context

The Infineon track gives ~3,000 real fab sequences (MOSFET / IGBT / IC) over a
~120-token vocabulary of multi-word step strings (`RECEIVE WAFER LOT`,
`WAFER SORT TEST`, `METAL CMP`, …). A natural question at hackathon kickoff:
why not fine-tune `distilgpt2` (82M) or a similar small pretrained LM rather
than train a 13M-param decoder Transformer from scratch?

## Decision

We do not use any HuggingFace pretrained model. We train a compact, from-scratch
decoder-only Transformer (`src/ml/transformer_model.py`, configs in
`configs/transformer_{small,medium}.yaml`).

## Consequences

Positive:
- Vocabulary is the natural unit of prediction. Each Infineon step token = one
  embedding row. No subword reconstruction needed at inference.
- Training is cheap (~5h for medium on a single A100), fits the Slurm budget.
- The model can be ensembled with the n-gram baseline because both operate on
  the same vocabulary.

Negative:
- We forfeit any benefit from pretrained-LM priors. For a domain this far from
  English, this is a near-zero loss; for adjacent tasks (e.g. natural-language
  process descriptions) it would matter.
- Smaller model → likely lower absolute Top-1 on truly novel patterns than a
  much larger pretrained model would deliver. We have not measured this
  rigorously; see "Alternatives considered".

## Alternatives considered

1. **`distilgpt2` (82M) BPE-tokenized fine-tune.** Tokenizer splits each step
   string into 3–5 BPE pieces. Decoder must then emit valid step strings
   piece-by-piece — error-prone, requires post-hoc projection back onto the
   120-token vocabulary. Pretrained English priors are wrong for this domain.
   Compute cost ~6× ours, would not finish in Slurm window.
2. **`distilgpt2` with vocabulary replacement.** Adding 120 new tokens forces
   the embedding layer to be retrained, throwing away the very priors that
   justified picking a pretrained model. Reduces to "13M-param model + 70M
   dead weights".
3. **TinyLlama 1.1B fine-tune.** Same issues, ~10× the compute.
4. **State-space model (Mamba).** Could work, but no time to debug, and the
   sequence lengths (≤180 tokens) don't stress Transformer attention enough to
   matter.
5. **TimesNet / PatchTST.** These are for continuous time-series, not symbolic
   discrete sequences. Not a fit.

See also: `docs/FINETUNE_OPTION_REVIEW.md` (longer-form review of the
distilgpt2 option, written before this decision).

## Revisit when

- We have a held-out OOD family (Task 4) and the from-scratch Transformer
  collapses on it. A pretrained model *might* generalize better via its
  English vocabulary priors — although the vocabulary-mismatch problem above
  still applies, so this is unlikely to be a clean win.
- We have >24h of compute budget and a clean way to align the BPE tokenizer
  with the step vocabulary.
