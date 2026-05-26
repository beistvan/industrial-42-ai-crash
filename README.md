# Industrial: Models that learn how processes unfold

Train a sequence model that learns industrial process trajectories and predicts the next states, bottlenecks, or anomalies.

## Track
Infineon — industrial process sequences, transformers/hybrids

## 36-hour target
Build a reliable MVP with:
- data loader;
- baseline model;
- improved model or orchestration layer;
- evaluation report;
- demo UI;
- pitch-ready story;
- one-command run path.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
make test
make run-demo
```

## Core idea
Transformer encoder/decoder, TimesNet/PatchTST, or hybrid temporal model with anomaly head

## Demo UI
Streamlit process replay: timeline, predicted next state, anomaly score, bottleneck root-cause candidates

## Dataset plan
NASA turbofan, SECOM manufacturing, semiconductor process logs if provided, synthetic event logs

## Main evaluation
Next-step prediction accuracy/MAE, anomaly F1/AUC, early warning lead time, robust performance under noise

## Files to read first
- `CLAUDE.md`
- `AGENTS.md`
- `docs/PRE_HACK_CHECKLIST.md`
- `docs/TASK_BOARD.md`
- `docs/DEMO_SPEC.md`
- `docs/PITCH_SCRIPT.md`


## Pre-hack mock-data smoke test

No sponsor data is available before the real case reveal, so this repo includes a synthetic mock-data generator.

Run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
make smoke
```

Expected output: a generated file in `data/raw/`, a baseline metrics file at `artifacts/metrics.json`, and passing pytest tests.
