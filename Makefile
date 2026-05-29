PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --force-reinstall --no-cache-dir --no-deps scikit-learn==1.6.1

generate-mock-data:
	$(PYTHON) scripts/generate_mock_data.py

train-baseline:
	$(PYTHON) src/ml/baseline.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts || true

run-demo:
	$(PYTHON) -m streamlit run src/app/main.py

run-dashboard:
	$(PYTHON) -m streamlit run src/app/eval_dashboard.py

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

# Step 5: compact decoder-only Transformer. Smoke target is CPU-safe.
train-transformer-smoke:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/train_transformer.py --limit-train-sequences 2 --limit-dev-sequences 2 --epochs 1 --skip-eval --d-model 16 --n-layers 1 --n-heads 2 --dim-feedforward 32 --batch-size 2 --model-path models/transformer_smoke.pt --metrics-path artifacts/transformer_smoke_metrics.json --device cpu

train-transformer-small:
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --model-path models/transformer_small.pt --metrics-path artifacts/transformer_metrics.json

train-transformer-small-local:
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --device cpu --epochs 1 --model-path models/transformer_small_local.pt --metrics-path artifacts/transformer_small_local_metrics.json

train-transformer-medium:
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_medium.yaml --model-path models/transformer_medium.pt --metrics-path artifacts/transformer_medium_metrics.json

local-eval-transformer:
	$(PYTHON) src/eval/local_eval.py --model models/transformer_small.pt --out artifacts/transformer_local_eval_metrics.json

predict-dev-transformer:
	$(PYTHON) scripts/predict_submission.py \
		--model models/transformer_small.pt \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir extras/results_transformer

# Step 6 local-only data augmentation. GPU/Leonardo scaling can use the same files later.
generate-extra-local:
	$(PYTHON) scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

train-transformer-small-extra-local:
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --device cpu --epochs 1 --extra-data-dir data/generated/infineon --model-path models/transformer_small_extra_local.pt --metrics-path artifacts/transformer_small_extra_local_metrics.json
