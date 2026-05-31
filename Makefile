PYTHON ?= $(shell if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)
STREAMLIT_PORT ?= 8501

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

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

# Final handoff test: fail if anything is skipped, especially Transformer tests.
test-strict: check-torch
	$(PYTHON) scripts/run_tests_strict.py

lint:
	$(PYTHON) -m ruff check src tests scripts || true

run-demo: check-app
	$(PYTHON) -m streamlit run src/app/main.py --server.port $(STREAMLIT_PORT)

run-dashboard: check-app
	$(PYTHON) -m streamlit run src/app/eval_dashboard.py --server.port $(STREAMLIT_PORT)

run-sweep-dashboard: check-app
	$(PYTHON) -m streamlit run src/app/sweep_dashboard.py --server.port $(STREAMLIT_PORT)

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
		--out-dir result/dev

# Step 5: compact decoder-only Transformer. Smoke target is CPU-safe but requires torch.
train-transformer-smoke: check-torch
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) scripts/train_transformer.py --limit-train-sequences 2 --limit-dev-sequences 2 --epochs 1 --skip-eval --d-model 16 --n-layers 1 --n-heads 2 --dim-feedforward 32 --batch-size 2 --model-path models/transformer_smoke.pt --metrics-path artifacts/transformer_smoke_metrics.json --device cpu

train-transformer-small: check-torch
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_small.yaml --model-path models/transformer_small.pt --metrics-path artifacts/transformer_metrics.json

train-transformer-medium: check-torch
	$(PYTHON) scripts/train_transformer.py --config configs/transformer_medium.yaml --model-path models/transformer_medium.pt --metrics-path artifacts/transformer_medium_metrics.json

local-eval-transformer: check-torch
	$(PYTHON) src/eval/local_eval.py --model models/transformer_small.pt --out artifacts/transformer_local_eval_metrics.json

TRANSFORMER_MODEL ?= models/sweeps/f_drop15_100_mrr.pt.best
TRANSFORMER_OUT   ?= result/dev
predict-dev-transformer: check-torch
	$(PYTHON) scripts/predict_submission.py \
		--model $(TRANSFORMER_MODEL) \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir $(TRANSFORMER_OUT)

# Step 6 local-only data augmentation. GPU/Leonardo scaling can use the same files later.
generate-extra-local:
	$(PYTHON) scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

# Leonardo / Slurm convenience targets. Run these on Leonardo after login, git clone, and env setup.
# Tune parallel GPU usage: export SWEEP_CONCURRENCY=32  (default 32 array tasks at once)
# Optional reservation:     export SLURM_RESERVATION=s_tra_ncc
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

# Generic sweep submit (SWEEP_CONCURRENCY default 32)
leonardo-wave1:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_final.yaml finalists

leonardo-wave2:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_fine.yaml finalists

leonardo-wave3:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_modern.yaml finalists

leonardo-wave3-tune:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_modern_tune.yaml finalists

leonardo-wave4:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_task2.yaml finalists

leonardo-wave5:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_params.yaml finalists

leonardo-wave6:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_scale.yaml finalists

leonardo-wave-lite:
	bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_lite.yaml finalists

leonardo-wave5-if-needed:
	bash scripts/leonardo/wave5_if_needed.sh

# Submit Waves 3+4+6 in parallel (~24 jobs), start auto-leaderboard watcher
leonardo-queue-parallel:
	bash scripts/leonardo/queue_parallel_waves.sh

leonardo-queue-all:
	bash scripts/leonardo/queue_parallel_waves.sh --with-wave5

# Background: rebuild LEADERBOARD_FINAL when metrics JSONs change; finalize when idle
leonardo-watch-pipeline:
	nohup bash scripts/leonardo/pipeline_watch.sh >> logs/pipeline-watch.out 2>&1 & \
	echo "pipeline_watch started — tail -f logs/pipeline_watch.log"

# Chain Wave 2 → 3 → 4 after Wave 2 is already submitted.
leonardo-wave-pipeline:
	@test -n "$$WAVE2_JOB" || (echo "Set WAVE2_JOB to the Wave 2 array id (e.g. 43145135)" && exit 1); \
	nohup bash scripts/leonardo/wave_orchestrator.sh $$WAVE2_JOB >> logs/wave_orchestrator-nohup.out 2>&1 & \
	echo "Pipeline watcher started (Wave2=$$WAVE2_JOB). tail -f logs/wave_orchestrator.log"

# Rebuild final leaderboard (after Wave 1 or Wave 2 completes)
leonardo-leaderboard-final:
	python3 scripts/summarize_runs.py \
	    --metrics-dir artifacts/sweeps \
	    --out artifacts/sweeps/LEADERBOARD_FINAL.md \
	    --csv artifacts/sweeps/LEADERBOARD_FINAL.csv
	@echo "Wrote artifacts/sweeps/LEADERBOARD_FINAL.{md,csv}"

# One-shot pipeline snapshot (login node)
leonardo-status:
	bash scripts/leonardo/status_now.sh

# Pick best T1/T2 from LEADERBOARD_FINAL.csv and write result/submission/*.csv
# Auto-uses Slurm when CUDA is unavailable (login node). Force local: FORCE_LOCAL=1
regenerate-submission:
	USE_SLURM=1 bash scripts/regenerate_submission.sh

# Pitch deck PDF for Tally (styled PPTX if LibreOffice; else Marp from SLIDES.md)
slides-pdf:
	bash scripts/export_slides.sh

# Generate +500/family synthetic data for Wave 1's f_extras_500_100_mrr row
leonardo-generate-extras-500:
	python3 scripts/generate_extra_sequences.py \
	    --count-per-family 500 --seed 201 \
	    --out-dir data/generated/infineon_500 --force
