# Model Plan — Industrial: Models that learn how processes unfold

## Baseline
A simple baseline that runs quickly and gives a reference point.

## Improved approach
Transformer encoder/decoder, TimesNet/PatchTST, or hybrid temporal model with anomaly head

## Training path
1. Train locally on tiny sample.
2. Run smoke test.
3. Scale batch size on A100 only after the pipeline works.
4. Save checkpoint and metrics after every useful run.

## GPU usage principle
Use A100s to run several controlled experiments or a larger batch/model only after the baseline is reproducible. Do not spend the first night debugging CUDA.

## Artifacts
- `models/baseline.pkl` or equivalent.
- `checkpoints/best_model.pt` if deep learning.
- `runs/metrics.json`.
- `artifacts/evaluation_report.md`.
