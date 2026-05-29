# Merge summary: Part 2 implementation + Part 3 dev-eval split

This merged repo keeps the current Part 2 implementation as the source of truth:

- `src/ml/ngram_baseline.py`
- `scripts/train_ngram.py`
- `scripts/predict_submission.py`
- `src/eval/run_eval.py`
- `src/eval/rule_validator.py`

The dev-eval work from Part 3 was integrated around that implementation instead
of replacing it.

## What was added/changed

- Added stable deterministic split generation in `src/data/splits.py`.
- Updated `scripts/make_dev_split.py` with:
  - `--force` overwrite protection,
  - default seed `42`,
  - `split_summary.json`,
  - 600 Task 1/2 partial-sequence dev examples,
  - 300 valid + 300 injected-invalid Task 3 examples.
- Added CLI wrapper `src/eval/local_eval.py` for evaluating a saved n-gram model.
- Added docs: `docs/DEV_EVAL_SPLIT.md`.
- Added test coverage: `tests/test_infineon_dev_split.py`.
- Added Makefile targets:
  - `make dev-split`
  - `make train-ngram`
  - `make local-eval`
  - `make predict-dev`

## Generated local files included in this ZIP

```text
data/processed/splits/train_ids.json
data/processed/splits/dev_ids.json
data/processed/splits/split_summary.json

data/processed/dev_eval/eval_input_valid_dev.csv
data/processed/dev_eval/eval_input_valid_dev_gold.csv
data/processed/dev_eval/eval_input_anomaly_dev.csv
data/processed/dev_eval/eval_input_anomaly_dev_gold.csv
```

A baseline run was also smoke-tested and produced:

```text
models/ngram_baseline.pkl
artifacts/ngram_metrics.json
artifacts/local_eval_metrics.json
extras/results/nextstep.csv
extras/results/completion.csv
extras/results/anomaly.csv
```

## Verified commands

```bash
python scripts/make_dev_split.py --force
python scripts/train_ngram.py
python src/eval/local_eval.py
python scripts/predict_submission.py \
  --model models/ngram_baseline.pkl \
  --eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
  --eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
  --out-dir extras/results
python -m pytest -q
```

Final test result: `22 passed`.
