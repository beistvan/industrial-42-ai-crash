# Task Board — Industrial: Models that learn how processes unfold

Use this file as the only task board. Phase 0 (pre-hack scaffolding) is done
and intentionally removed — see git history if you need it.

## Done

- Real Infineon `training_data/` vendored (`data/raw/infineon/`).
- Deterministic train/dev split, seed-pinned
  (`scripts/make_dev_split.py`, md5-verified reproducible).
- N-gram suffix-backoff baseline + persistence
  (`src/ml/ngram_baseline.py`, `models/ngram_baseline.pkl`).
- Task 1/2/3 evaluator and rule-validator wrapper
  (`src/eval/run_eval.py`, `src/eval/rule_validator.py`).
- Metrics + per-run snapshots
  (`artifacts/ngram_metrics.json`, `artifacts/runs/*.json`).
- Streamlit demo + run-history dashboard
  (`src/app/main.py`, `src/app/eval_dashboard.py`).
- `make smoke` end-to-end pipeline + pytest (33 tests passing).
- Local generated-data augmentation
  (`scripts/generate_extra_sequences.py`, 250 sequences/family default).
- Compact decoder-only Transformer, CPU smoke path green
  (`src/ml/transformer_model.py`, `scripts/train_transformer.py`).
- Submission writer supports both n-gram and Transformer
  (`scripts/predict_submission.py`).
- Rule-constrained completion (opt-in, default-on in CLIs)
  (`src/ml/{ngram_baseline,transformer_model}.py::complete`).
- Task 3 primary-rule = earliest step_index
  (`src/eval/rule_validator.py`).
- **HPC battle harness ready**: train script supports AMP, scheduler,
  label_smoothing, best-by-metric checkpointing, eval cadence; sweep YAML
  with 12 shortlist configs; Slurm array template + dep-chained leaderboard
  job; runbook
  (`scripts/train_transformer.py`, `configs/sweeps/leonardo_v1.yaml`,
  `scripts/sweep_transformer.py`, `scripts/summarize_runs.py`,
  `scripts/leonardo/{sweep_array,train_one,run_battle}.{slurm,sh}`,
  `docs/LEONARDO_BATTLE_PLAN.md`).

## Open — must-have before submission

| Task | Status | Pointer |
| --- | --- | --- |
| Pick winning Transformer checkpoint via Leonardo sweep | blocked on HPC access | follow `docs/LEONARDO_BATTLE_PLAN.md` |
| Produce final `extras/results_submission/{nextstep,completion,anomaly}.csv` against organizer-distributed eval inputs | blocked on org eval drop | `scripts/predict_submission.py` |
| Submit via Tally (demo video + GitHub link + slide deck) | Sun 10:00 deadline | Z10 deck p.70 |

## Open — should-have

| Task | Status | Pointer |
| --- | --- | --- |
| **Shock-injection demo page in Streamlit** (see `docs/DEMO_SPEC.md`) | not started, ~3h | new tab in `src/app/main.py` |
| 3-min pitch script aligned with current numbers + chosen checkpoint | needs update after sweep | `docs/PITCH_SCRIPT.md` |
| Demo video recording (2-min) | not started | required for submission |
| Slide deck (3-min pitch) | not started | required for submission |

## Open — nice-to-have

| Task | Status | Pointer |
| --- | --- | --- |
| Beam search for Task 2 (compose with rule-constrained) | not started | `src/ml/*::complete` |
| Per-family `max_order` tuning + Kneser–Ney on the n-gram | not started | `src/ml/ngram_baseline.py` |
| Transformer ensembling with n-gram for low-suffix-coverage prefixes | not started | inference wrapper |
| Family-token dropout at train time (Task 4 OOD insurance) | not started | `scripts/train_transformer.py` |
| Calibrated confidence on top-k (entropy / margin) | not started | new metric in `src/eval/metrics.py` |

## Improvement levers (only after the next model lands)

- Compare best Transformer vs n-gram on the official eval; ensemble if
  complementary errors.
- Augment with `generate_sequences.py --count 10000` if synthetic-only
  beats real on dev (signals more data helps).

## Do not do

- Do not improve the n-gram with tricks the transformer can't match —
  optimize for the metric, not the model.
- Do not touch the other two track repos after selection.
- Do not start cluster training before the local pipeline is green.
- Do not commit credentials, SSH keys, or Leonardo internal links.
