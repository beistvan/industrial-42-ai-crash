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


# Leonardo / Slurm convenience targets. Run these on Leonardo after login, git clone, and env setup.
leonardo-setup:
	bash scripts/leonardo/setup_env.sh

leonardo-gpu-smoke:
	sbatch scripts/leonardo/01_gpu_smoke.slurm

leonardo-train-small:
	sbatch scripts/leonardo/02_train_small.slurm

leonardo-train-small-extra:
	sbatch scripts/leonardo/03_train_small_extra.slurm

leonardo-train-medium-extra:
	sbatch scripts/leonardo/04_train_medium_extra.slurm

leonardo-predict-transformer:
	sbatch scripts/leonardo/05_predict_transformer.slurm

leonardo-pack:
	bash scripts/leonardo/pack_for_upload.sh

leonardo-generate-extra-10k:
	sbatch scripts/leonardo/00_generate_extra_10k.slurm

# 12h-plan Wave 1: 6 parallel 100/150-epoch finalists. Concurrency 12.
# Run on Leonardo after `bash scripts/leonardo/copy_to_leonardo.sh` from your laptop.
leonardo-wave1:
	@SLURM_ACCOUNT=$${SLURM_ACCOUNT:-EUHPC_D30_031}; \
	N=$$(python3 scripts/sweep_transformer.py --sweep configs/sweeps/leonardo_final.yaml --stage finalists --dry-run | grep -c '^\['); \
	echo "Submitting Wave 1: $$N finalists, concurrency 12, account=$$SLURM_ACCOUNT"; \
	sbatch --account=$$SLURM_ACCOUNT --array=0-$$((N-1))%12 \
	    scripts/leonardo/sweep_array.slurm configs/sweeps/leonardo_final.yaml finalists

# 12h-plan Wave 2: fine grid around Wave 1 winners. EDIT leonardo_fine.yaml first.
leonardo-wave2:
	@SLURM_ACCOUNT=$${SLURM_ACCOUNT:-EUHPC_D30_031}; \
	N=$$(python3 scripts/sweep_transformer.py --sweep configs/sweeps/leonardo_fine.yaml --stage finalists --dry-run | grep -c '^\['); \
	echo "Submitting Wave 2: $$N rows, concurrency 12, account=$$SLURM_ACCOUNT"; \
	sbatch --account=$$SLURM_ACCOUNT --array=0-$$((N-1))%12 \
	    scripts/leonardo/sweep_array.slurm configs/sweeps/leonardo_fine.yaml finalists

# Rebuild final leaderboard (after Wave 1 or Wave 2 completes)
leonardo-leaderboard-final:
	python3 scripts/summarize_runs.py \
	    --metrics-dir artifacts/sweeps \
	    --out artifacts/sweeps/LEADERBOARD_FINAL.md \
	    --csv artifacts/sweeps/LEADERBOARD_FINAL.csv
	@echo "Wrote artifacts/sweeps/LEADERBOARD_FINAL.{md,csv}"

# Generate +500/family synthetic data for Wave 1's f_extras_500_100_mrr row
leonardo-generate-extras-500:
	python3 scripts/generate_extra_sequences.py \
	    --count-per-family 500 --seed 201 \
	    --out-dir data/generated/infineon_500 --force
