PYTHON ?= $(shell if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)
STREAMLIT_PORT ?= 8501

.PHONY: setup setup-cpu check-env check-torch check-app test test-strict lint \
	run-dashboard dashboard smoke dev-split train-ngram generate-extra-local \
	local-eval predict-dev train-transformer-smoke predict-dev-transformer \
	leonardo-leaderboard-final regenerate-submission leonardo-status slides-pdf

# --- Environment ---

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --force-reinstall --no-cache-dir --no-deps scikit-learn==1.6.1

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

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

test-strict: check-torch
	$(PYTHON) scripts/run_tests_strict.py

lint:
	$(PYTHON) -m ruff check src tests scripts || true

# --- Dashboard ---

run-dashboard: check-app
	$(PYTHON) -m streamlit run src/app/dashboard.py --server.port $(STREAMLIT_PORT)

dashboard: run-dashboard

# --- Local pipeline (CPU smoke) ---

smoke: dev-split train-ngram test
	$(PYTHON) scripts/smoke_test.py

dev-split:
	$(PYTHON) scripts/make_dev_split.py --force

train-ngram:
	$(PYTHON) scripts/train_ngram.py

generate-extra-local:
	$(PYTHON) scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

local-eval:
	$(PYTHON) src/eval/local_eval.py

predict-dev:
	$(PYTHON) scripts/predict_submission.py \
		--model models/ngram_baseline.pkl \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir result/dev

train-transformer-smoke: check-torch
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/train_transformer.py \
		--limit-train-sequences 2 --limit-dev-sequences 2 --epochs 1 --skip-eval \
		--d-model 16 --n-layers 1 --n-heads 2 --dim-feedforward 32 --batch-size 2 \
		--model-path models/transformer_smoke.pt \
		--metrics-path artifacts/transformer_smoke_metrics.json --device cpu

TRANSFORMER_MODEL ?= models/sweeps/h_mod_nosched_mrr.pt.best
TRANSFORMER_OUT   ?= result/dev

predict-dev-transformer: check-torch
	$(PYTHON) scripts/predict_submission.py \
		--model $(TRANSFORMER_MODEL) \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir $(TRANSFORMER_OUT)

# --- Submission (final picks frozen) ---

leonardo-leaderboard-final:
	$(PYTHON) scripts/summarize_runs.py \
		--metrics-dir artifacts/sweeps \
		--out artifacts/sweeps/LEADERBOARD_FINAL.md \
		--csv artifacts/sweeps/LEADERBOARD_FINAL.csv
	@echo "Wrote artifacts/sweeps/LEADERBOARD_FINAL.{md,csv}"

regenerate-submission:
	USE_SLURM=1 bash scripts/regenerate_submission.sh

leonardo-status:
	bash scripts/leonardo/status_now.sh

slides-pdf:
	bash scripts/export_slides.sh
