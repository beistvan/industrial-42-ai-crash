# MEMORY.md — Industrial: Models that learn how processes unfold

## Stable context
Repo: `industrial-models-that-learn-how-processes-unfold`
Track: Infineon — industrial process sequences, transformers/hybrids
Product: Train a sequence model that learns industrial process trajectories and predicts the next states, bottlenecks, or anomalies.

## Current decisions
Dataset: TBD
Baseline: TBD
Improved model: TBD
Frontend: Streamlit unless changed
Main metric: Next-step prediction accuracy/MAE, anomaly F1/AUC, early warning lead time, robust performance under noise

## Event-specific updates
Add brief details here after kickoff.

## Risk log
- Data not available: fallback dataset/synthetic generator.
- Model too slow: use baseline + smaller model.
- UI breaks: keep CLI demo and screenshots.
