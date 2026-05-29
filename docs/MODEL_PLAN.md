# Model Plan — Industrial: Models that learn how processes unfold

Tracks our actual modeling stack for the Infineon track. For the end-to-end
pipeline see [`PIPELINE.md`](./PIPELINE.md); for the strategic rationale
see [`implementation-plan-en.md`](./implementation-plan-en.md) and
[`FINETUNE_OPTION_REVIEW.md`](./FINETUNE_OPTION_REVIEW.md).

## Spine

1. **N-gram baseline (done).** Family-conditioned suffix-backoff predictor
   in `src/ml/ngram_baseline.py`. Implements `predict_topk(family, prefix,
   k)` and `complete(family, prefix, …)`. Strong on Task 1 (Top-5 ≈ 0.99,
   MRR ≈ 0.83), weak on Task 2 exact completion (greedy roll-out).
2. **Small decoder-only transformer (next).** Placeholder at
   `src/ml/transformer_model.py`. Vocabulary ≈ 130 tokens (PAD/BOS/EOS/UNK
   + family + ~120 step tokens), so embeddings stay tiny. Suggested
   configurations (per `implementation-plan-en.md` §5):
   - `small`: d_model 128, n_layers 4, n_heads 4, max_len 180, batch 128
   - `medium`: d_model 256, n_layers 6, n_heads 8, max_len 180, batch 256
   Training objective: next-token prediction with family token prefix
   (`<BOS> <FAMILY_*> step_1 … step_n`).
3. **Rule validator (done).** `src/eval/rule_validator.py` wraps the
   official `validate_sequence` from
   `data/raw/infineon/training_data/generate_sequences.py`. Used as the
   primary Task 3 classifier. Re-evaluates 10 documented rule types and
   attributes the violation.

## Swap-in interface

Both the n-gram and the transformer satisfy:

```python
class NextStepModel(Protocol):
    def predict_topk(self, family: Family, prefix: list[str], k: int = 5) -> list[str]: ...
    def complete(self, family: Family, prefix: list[str], *,
                 max_steps: int = 200, stop_token: str = "SHIP LOT") -> list[str]: ...
```

`src/eval/run_eval.evaluate_all(model, eval_dir)` evaluates anything that
satisfies this protocol against the dev split → Task 1/2/3 metrics in one
shot. Same call for the n-gram, the transformer, and any HF fine-tune.

## Training path

1. Train locally on the existing 1,000 sequences/family.
2. Verify dashboard shows improvement vs n-gram baseline.
3. Generate extra synthetic data with the official CLI (`generate_sequences.py
   --family … --count 10000`) and retrain.
4. Once local pipeline is green, move to Leonardo (see
   [`LEONARDO_ONBOARDING.md`](./LEONARDO_ONBOARDING.md)) for the medium
   model + scaling experiments.
5. Save checkpoint + metrics for every useful run; the dashboard reads
   `artifacts/runs/*.json` automatically.

## GPU usage principle

Do not debug CUDA on the cluster. The from-scratch transformer is small
enough to verify on CPU / MPS first. Use Leonardo A100s only for:
- the medium configuration,
- the scaling study (small / medium × 3k / 30k sequences),
- the optional `distilgpt2` contrast experiment.

## Artifacts

- `models/ngram_baseline.pkl` — fitted n-gram (done).
- `models/transformer_*.pt` — transformer checkpoints (next).
- `artifacts/ngram_metrics.json` — latest metrics (done).
- `artifacts/runs/*.json` — per-run snapshots powering the dashboard
  (done).
- `artifacts/submission/{nextstep,completion,anomaly}.csv` — official
  submission files, produced by `scripts/predict_submission.py`.

## Out of scope

See [`FINETUNE_OPTION_REVIEW.md`](./FINETUNE_OPTION_REVIEW.md) — we are
**not** leading with a large pretrained LLM. The 120-token vocab is
unrelated to natural language, the brief rewards an open transparent stack,
and Task 4 OOD penalizes memorization-prone large models.
