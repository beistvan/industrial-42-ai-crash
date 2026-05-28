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
