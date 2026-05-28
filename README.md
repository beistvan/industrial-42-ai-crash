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

---

# Industrial: Models that learn how processes unfold

**START HERE. This README is the single operating path for this track.**

Goal: predict process state, bottleneck risk, or anomaly from process telemetry.

## Authoritative files only

Use these files for hackathon execution and ignore older scattered instruction files unless a maintainer explicitly asks for them:

1. `README.md` — this start page and one-command path.
2. `docs/PRE_HACK_CHECKLIST.md` — readiness checklist before the track reveal.
3. `docs/TASK_BOARD.md` — what to do now, tomorrow, and during demo prep.
4. `docs/DEMO_SPEC.md` — what the Streamlit app must show.
5. `docs/JUDGE_CARD.md` — pitch/judging story.
6. `artifacts/metrics.json` — latest baseline evidence.

Do **not** spend time reconciling `AGENTS.md`, `CLAUDE.md`, `MEMORY.md`, `RULES.md`, `SKILLS.md`, old orchestration packs, or extra planning docs during the hackathon. Treat them as background/reference only.

## Current baseline state

- Rows: `2400`
- Accuracy: `0.9983`
- F1 anomaly: `0.9934`
- Baseline: RandomForest anomaly classifier on synthetic process-sequence telemetry

## One-command path

```bash
make smoke
make test
make run-demo
```

Expected result:

- mock data exists in `data/raw/`;
- baseline evidence exists in `artifacts/metrics.json`;
- tests pass;
- Streamlit demo opens at `http://localhost:8501`.

## Demo app

```bash
make run-demo
```

The demo currently shows: process scenario input -> normal/watch/anomaly decision -> expected next step -> confidence/explanation -> metric evidence -> data preview.

## Tomorrow after case reveal

1. Save the real brief in `docs/REAL_BRIEF.md`.
2. Put real data in `data/raw/`.
3. Inspect columns, target, missingness, row count, and examples.
4. Map real data into the existing baseline pipeline.
5. Run `make smoke` and fix the baseline on real data.
6. Only after baseline works, improve model/policy/prompting.
7. Keep the demo wired at all times.

Real-data mapping priorities:

- run/process identifier
- step/order/time column
- sensor or state variables
- anomaly/failure/bottleneck target if available
- next-step or quality outcome if available

## Stop rule

Once this track is selected, stop touching the other two track repos. If this track is not selected, leave it frozen.
