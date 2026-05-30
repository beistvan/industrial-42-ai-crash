# Leonardo GPU Results — Infineon Industrial Track

## Local baseline

| Run | Data | Model | Task 1 Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---|---:|---:|---:|---:|
| baseline | 3k seq | n-gram suffix backoff | 0.6867 | 0.9283 | 0.9283 | 0.8072 |
| baseline+extra | 3k + 750 seq | n-gram suffix backoff | 0.6933 | 0.9483 | 0.9500 | 0.8207 |
| small | 3k seq | decoder-only Transformer | 0.7017 | 0.9983 | 1.0000 | 0.8479 |
| small+extra | 3k + 750 seq | decoder-only Transformer | 0.7267 | 0.9983 | 1.0000 | 0.8612 |

## Selected model

Lead neural checkpoint: `models/transformer_small_extra.pt.best`

The model is not committed to Git because checkpoint files are ignored. Prediction CSV outputs are stored in:

- `extras/results_transformer/`
- `extras/results_transformer_small_extra/`

## Interpretation

The Transformer improves next-step prediction over the n-gram baseline, especially Top-5 and MRR. Task 3 anomaly detection remains rule-based through the official Infineon validator.
