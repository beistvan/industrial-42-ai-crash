# App Blueprint — Industrial Process Sequence Model

## Main idea
Learn how a process unfolds over time and predict the next state, anomaly risk, or bottleneck before it becomes expensive.

## Minimal winning demo
User loads a process run. The app replays the timeline, predicts the next process state, flags anomaly risk, and highlights which sensor/event segments caused concern.

## Model/layer stack
- Baseline: logistic regression / random forest / simple temporal statistics.
- Improved: transformer sequence model, temporal CNN, PatchTST, or hybrid sequence encoder.
- Explanation: feature importance, attention visualization if safe, or perturbation-based sensitivity.

## Best judge story
“We trained a model that learns trajectories, not just static snapshots, and turns process history into early operational warnings.”
