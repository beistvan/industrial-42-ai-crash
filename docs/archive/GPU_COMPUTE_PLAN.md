# GPU Compute Plan — Industrial: Models that learn how processes unfold

## Principle
Use the supercomputer only after the code runs locally on a tiny sample.

## Suggested steps
1. Local CPU smoke test.
2. Single GPU small batch.
3. Larger batch or hyperparameter sweep.
4. Save best checkpoint.
5. Stop training when demo/pitch risk is higher than model-quality gain.

## A100-friendly libraries
- PyTorch for custom training.
- Hugging Face Transformers/Datasets for LLM or sequence models.
- PyTorch Lightning or plain PyTorch for clean loops.
- scikit-learn for baselines.
- Optuna only if time allows.

## Avoid
- distributed training unless provided templates make it easy;
- custom CUDA;
- complex MLOps platforms;
- model sizes that cannot be explained or demoed.
