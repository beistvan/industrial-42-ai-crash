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

smoke: setup
	$(PYTHON) scripts/generate_mock_data.py
	$(PYTHON) src/ml/baseline.py
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/smoke_test.py

dev-split:
	$(PYTHON) scripts/make_dev_split.py --force

train-ngram:
	$(PYTHON) scripts/train_ngram.py

local-eval:
	$(PYTHON) src/eval/local_eval.py

predict-dev:
	$(PYTHON) scripts/predict_submission.py \
		--model models/ngram_baseline.pkl \
		--eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
		--eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
		--out-dir extras/results
