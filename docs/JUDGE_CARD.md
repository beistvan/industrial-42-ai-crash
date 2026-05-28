# Judge Card — Industrial / Infineon

## One-liner
Process Sequence Model helps users learn process evolution from event/sensor sequences with a measurable pipeline, a baseline comparison, and a working demo.

## What we will build if this track is chosen
sequence encoder + next-state prediction + anomaly/trajectory explanation.

## User-facing value
process timeline viewer with predicted next steps, drift warning, anomaly heatmap, and what-changed explanation.

## Data assumption
hackathon provided process logs/telemetry; fallback: synthetic multi-stage manufacturing traces with drift and defects.

## Baseline
rolling statistics + GRU/1D-CNN baseline.

## Advanced model path
Transformer encoder / TCN / hybrid state-space fallback with masked-step prediction.

## Evaluation metrics
next-step accuracy, masked reconstruction loss, anomaly AUROC, early-warning lead time, inference latency.

## Why judges should care
Judge sees a real trained sequence model that predicts/understands process dynamics instead of a static classifier.

## Cut line
If time is short, keep baseline + eval + demo. Cut the advanced model before cutting the demo.

---

# Judge card — Industrial: Models that learn how processes unfold

Use this page for the final story. Keep it short.

## Problem

predict process state, bottleneck risk, or anomaly from process telemetry.

## Data assumption

Likely data type: process sequence / sensor / event-log / manufacturing telemetry data.

## Baseline

RandomForest anomaly classifier on synthetic process-sequence telemetry.

Current evidence:

- Rows: `2400`
- Accuracy: `0.9983`
- F1 anomaly: `0.9934`
- Baseline: RandomForest anomaly classifier on synthetic process-sequence telemetry

## What the demo proves

- Data-to-decision pipeline is wired.
- Metrics are visible and reproducible.
- The decision is explainable, not a black box.
- The app remains usable when real data replaces mock data.

## What judges should care about

- early warning usefulness
- anomaly/bottleneck detection quality
- process explanation trace
- robust real-data baseline

## Tomorrow upgrade path

Map real process logs to run_id + step/time + telemetry features, then run the baseline before adding sequence models.

## Pitch line

“We came with a tested scaffold, replaced synthetic data with the official case data, established a reproducible baseline, then improved the decision layer where it mattered most.”
