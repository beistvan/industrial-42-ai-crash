# Task Board — Industrial: Models that learn how processes unfold

Use this file as the only task board. Phase 0 (pre-hack scaffolding) is done
and intentionally removed — see git history if you need it.

## Done

- Real Infineon `training_data/` vendored (`data/raw/infineon/`).
- Deterministic train/dev split (`scripts/make_dev_split.py`).
- N-gram suffix-backoff baseline + persistence
  (`src/ml/ngram_baseline.py`, `models/ngram_baseline.pkl`).
- Task 1/2/3 evaluator (`src/eval/run_eval.py`) and rule-validator wrapper
  (`src/eval/rule_validator.py`).
- Metrics + per-run snapshots
  (`artifacts/ngram_metrics.json`, `artifacts/runs/*.json`).
- Streamlit demo + run-history dashboard
  (`src/app/main.py`, `src/app/eval_dashboard.py`).
- `make smoke` runs the real-data pipeline end-to-end + pytest (20 tests).

## In progress / next

| Task | Status | Pointer |
| --- | --- | --- |
| Independent evaluator cross-check | placeholder | `src/eval/local_eval.py` |
| Small from-scratch transformer | placeholder | `src/ml/transformer_model.py`, `scripts/train_transformer.py` |
| Beam search for Task 2 completion | not started | `src/ml/ngram_baseline.complete` |
| Submission writer | scaffolded | `scripts/predict_submission.py` |
| Leonardo HPC access | doc'd | `docs/LEONARDO_ONBOARDING.md` |
| Optional `distilgpt2` contrast | optional | `docs/FINETUNE_OPTION_REVIEW.md` |

## Improvement levers (only after the next model lands)

- Beam search + grammar-constrained decoding for Task 2 exact match.
- Per-family `max_order` tuning + Kneser–Ney smoothing on the n-gram.
- Transformer ensembling with the n-gram for low-suffix-coverage prefixes.
- Augment with `generate_sequences.py --count 10000` for scaling rows.
- Calibrated confidence on top-k (entropy / margin).

## Do not do

- Do not improve the n-gram with tricks the transformer can't match —
  optimize for the metric, not the model.
- Do not touch the other two track repos after selection.
- Do not start cluster training before the local pipeline is green.
