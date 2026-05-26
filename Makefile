PYTHON ?= python

setup:
	$(PYTHON) -m pip install -r requirements.txt

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

smoke: generate-mock-data train-baseline test
	$(PYTHON) scripts/smoke_test.py
