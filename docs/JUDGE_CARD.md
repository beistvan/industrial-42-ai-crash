# Judge Card — Industrial / Infineon

## Problem
Given partial semiconductor fabrication sequences (MOSFET / IGBT / IC, ~120
step vocabulary), predict the next step, complete the sequence, and detect
rule-violating sequences. A hidden 4th product family scores OOD
generalization post-submission.

## Baseline shipped
Family-conditioned suffix-backoff **n-gram** (`src/ml/ngram_baseline.py`),
`max_order=8`, trained on 900 sequences/family from the official Infineon
`training_data/`. Anomaly detection uses the official `validate_sequence`
rule validator unmodified.

## Current evidence (`artifacts/ngram_metrics.json`, dev split = 100/family)

| Task | Metric | Value |
| --- | --- | ---: |
| 1 — next-step | Top-1 | **0.693** |
| 1 — next-step | Top-5 | **0.988** |
| 1 — next-step | MRR | **0.838** |
| 2 — completion | Token accuracy | 0.405 |
| 2 — completion | Norm. edit distance | 0.224 |
| 3 — anomaly | F1 (invalid) | **1.000** |
| 3 — anomaly | Rule attribution | 0.667 |

## What the demo proves
- Real data → real metrics: no mock data, no RandomForest.
- Reproducible with one command (`make smoke` → `make run-demo`).
- Decisions are explainable: matched suffix order, top-k probabilities,
  per-violation rule + step index.
- Same evaluation harness for n-gram, transformer, and any HF contrast — see
  the `NextStepModel` protocol in `src/eval/run_eval.py`.

## Honest limitations
- Greedy completion has 0% exact match → beam + transformer next.
- Rule attribution 0.667 reflects multi-rule sequences; not a model bug.

## Pitch line
"We trained a model that actually learns process logic, evaluated it on the
official dev split with the official rule validator, and shipped a demo that
explains every prediction it makes."
