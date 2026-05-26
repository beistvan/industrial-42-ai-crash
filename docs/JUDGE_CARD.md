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
