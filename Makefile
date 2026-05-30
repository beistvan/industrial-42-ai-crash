PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --force-reinstall --no-cache-dir --no-deps scikit-learn==1.6.1

# CPU-only local setup for step 5 Transformer smoke/small runs.
# Use this on laptops/WSL. On Leonardo/GPU, install the CUDA-enabled torch build instead.
setup-cpu:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch
	$(PYTHON) scripts/check_environment.py --require-torch

check-env:
	$(PYTHON) scripts/check_environment.py

check-torch:
	$(PYTHON) scripts/check_environment.py --require-torch

check-app:
	$(PYTHON) scripts/check_environment.py --require-streamlit

generate-mock-data:
	$(PYTHON) scripts/generate_mock_data.py

train-baseline:
	$(PYTHON) src/ml/baseline.py

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

# Final handoff test: fail if anything is skipped, especially Transformer tests.
test-strict: check-torch
	$(PYTHON) scripts/run_tests_strict.py

lint:
	$(PYTHON) -m ruff check src tests scripts || true

run-demo: check-app
	$(PYTHON) -m streamlit run src/app/main.py

run-dashboard: check-app
	$(PYTHON) -m streamlit run src/app/eval_dashboard.py

# Full local sanity path: real dev split, n-gram baseline, full tests, smoke report.
# Transformer tests no longer skip: without PyTorch they fail with an install hint.
smoke: dev-split train-ngram test
	$(PYTHON) scripts/smoke_test.py

# Real Infineon local dev split and n-gram baseline.
dev-split:
	$(PYTHON) scripts/make_dev_split.py --force

train-ngram:
	$(PYTHON) scripts/train_ngram.py

train-ngram-extra:
	$(PYTHON) scripts/train_ngram.py --extra-data-dir data/generated/infineon --metrics-path artifacts/ngram_extra_metrics.json

local-eval:
	$(PYTHON) src/eval/local_eval.py

predict-dev:
	$(PYTHON) scripts/predict_submission.py \
		--model models/ngram_baseline.pkl \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir extras/results

# Step 5: compact decoder-only Transformer. Smoke target is CPU-safe but requires torch.
train-transformer-smoke: check-torch
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/train_transformer.py --limit-train-sequences 2 --limit-dev-sequences 2 --epochs 1 --skip-eval --d-model 16 --n-layers 1 --n-heads 2 --dim-feedforward 32 --batch-size 2 --model-path models/transformer_smoke.pt --metrics-path artifacts/transformer_smoke_metrics.json --device cpu

train-transformer-small: check-torch
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --model-path models/transformer_small.pt --metrics-path artifacts/transformer_metrics.json

train-transformer-small-local: check-torch
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --device cpu --epochs 1 --model-path models/transformer_small_local.pt --metrics-path artifacts/transformer_small_local_metrics.json

train-transformer-medium: check-torch
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_medium.yaml --model-path models/transformer_medium.pt --metrics-path artifacts/transformer_medium_metrics.json

local-eval-transformer: check-torch
	$(PYTHON) src/eval/local_eval.py --model models/transformer_small.pt --out artifacts/transformer_local_eval_metrics.json

TRANSFORMER_MODEL ?= models/transformer_small.pt
TRANSFORMER_OUT   ?= extras/results_transformer
predict-dev-transformer: check-torch
	$(PYTHON) scripts/predict_submission.py \
		--model $(TRANSFORMER_MODEL) \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir $(TRANSFORMER_OUT)

# Step 6 local-only data augmentation. GPU/Leonardo scaling can use the same files later.
generate-extra-local:
	$(PYTHON) scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

train-transformer-small-extra-local: check-torch
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --device cpu --epochs 1 --extra-data-dir data/generated/infineon --model-path models/transformer_small_extra_local.pt --metrics-path artifacts/transformer_small_extra_local_metrics.json

# Fast local path for step 6 when you only want to verify wiring before a full CPU/GPU run.
train-transformer-extra-smoke: check-torch
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/train_transformer.py --limit-train-sequences 4 --limit-dev-sequences 2 --limit-extra-sequences 2 --epochs 1 --skip-eval --extra-data-dir data/generated/infineon --d-model 16 --n-layers 1 --n-heads 2 --dim-feedforward 32 --batch-size 2 --model-path models/transformer_extra_smoke.pt --metrics-path artifacts/transformer_extra_smoke_metrics.json --device cpu
