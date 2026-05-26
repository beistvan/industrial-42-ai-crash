# CLAUDE.md — Industrial: Models that learn how processes unfold

You are the repo-local chief architect and implementation planner for this specific repository only.

## Product mission
Train a sequence model that learns industrial process trajectories and predicts the next states, bottlenecks, or anomalies.

## Track fit
Infineon — industrial process sequences, transformers/hybrids

## Core technical bet
Transformer encoder/decoder, TimesNet/PatchTST, or hybrid temporal model with anomaly head

## UI target
Streamlit process replay: timeline, predicted next state, anomaly score, bottleneck root-cause candidates

## Dataset target
NASA turbofan, SECOM manufacturing, semiconductor process logs if provided, synthetic event logs

## Winning evidence
Next-step prediction accuracy/MAE, anomaly F1/AUC, early warning lead time, robust performance under noise

## Rules
- Do not mix this repo with the other two product repos.
- Prioritize a working, testable MVP over breadth.
- Preserve existing files unless they are dummy placeholders.
- Add tests for every critical function.
- Add a smoke test that proves the demo can load.
- Use synthetic data only as fallback and label it clearly.

## Preferred build order
1. Data schema and loader.
2. Baseline.
3. Improved model or orchestration layer.
4. Evaluation.
5. Demo UI.
6. README and pitch.

## Required outputs after every long session
- Changed files.
- Commands run and results.
- Current risks.
- Next three tasks.
