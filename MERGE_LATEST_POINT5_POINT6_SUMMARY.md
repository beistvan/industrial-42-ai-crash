# Latest merge summary — Point 5 + local Point 6

Base: latest uploaded `industrial-models-that-learn-how-processes-unfold.zip`.
Merged in: previous Transformer / local augmentation integration.

## Preserved from latest repo

- Updated README operating path and current n-gram baseline status.
- Streamlit eval dashboard at `src/app/eval_dashboard.py`.
- Current n-gram baseline, dev split, run-eval, rule-validator, smoke-test, and docs updates.
- Existing real Infineon data and processed dev-eval files.

## Added / restored from previous Point 5 work

- Compact decoder-only Transformer implementation:
  - `src/ml/transformer_model.py`
  - `scripts/train_transformer.py`
  - `configs/transformer_small.yaml`
  - `configs/transformer_medium.yaml`
  - `tests/test_transformer_model.py`
- Unified model loading:
  - `src/ml/loading.py`
  - `.pkl` n-gram and `.pt` Transformer checkpoints both work through `load_sequence_model()`.
- Updated inference/eval tools to support both model types:
  - `scripts/predict_submission.py`
  - `src/eval/local_eval.py`
- Updated Streamlit demo:
  - `make run-demo` can select n-gram `.pkl` or Transformer `.pt` checkpoints.
  - Task 1 / 2 / 3 demo stays on one page.
  - N-gram explanations show suffix-backoff evidence; Transformer explanations show decoder context.

## Added / restored from local Point 6 work

- Local generated-data helper:
  - `scripts/generate_extra_sequences.py`
- Extra-data training path:
  - `scripts/train_ngram.py --extra-data-dir ...`
  - `scripts/train_transformer.py --extra-data-dir ...`
- Make targets:
  - `make train-transformer-smoke`
  - `make train-transformer-small-local`
  - `make generate-extra-local`
  - `make train-ngram-extra`
  - `make train-transformer-small-extra-local`
  - `make run-dashboard`

## Verification run

Executed successfully after merge:

```bash
python -m py_compile src/data/infineon_loader.py src/data/__init__.py src/ml/loading.py \
  src/ml/transformer_model.py src/ml/__init__.py scripts/train_transformer.py \
  scripts/train_ngram.py scripts/predict_submission.py src/eval/local_eval.py \
  src/app/main.py scripts/generate_extra_sequences.py scripts/smoke_test.py

python -m pytest -q
python scripts/smoke_test.py
make train-transformer-smoke
python src/eval/local_eval.py --model models/ngram_baseline.pkl --out artifacts/local_eval_check.json
python scripts/predict_submission.py --model models/ngram_baseline.pkl \
  --eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
  --eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
  --out-dir extras/results_check
python scripts/generate_extra_sequences.py --count-per-family 2 --seed 101 --out-dir /tmp/inf_extra_merge_test --force
python scripts/train_ngram.py --extra-data-dir /tmp/inf_extra_merge_test --metrics-path artifacts/ngram_extra_test_metrics.json
```

Final packaged repo was cleaned of temporary smoke/test artifacts before zipping.
