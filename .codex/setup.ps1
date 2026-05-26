$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip
if (Test-Path requirements.txt) { pip install -r requirements.txt }
python scripts/smoke_test.py
