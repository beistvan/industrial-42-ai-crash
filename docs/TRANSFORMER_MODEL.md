# Compact decoder-only Transformer for the Infineon industrial track

This repo now supports a small family-conditioned causal Transformer in addition
to the suffix-backoff n-gram baseline.

## Why this model

The official Infineon case is a sequence-learning problem over a small process
vocabulary. A large LLM is unnecessary. We use a compact decoder-only Transformer
that receives:

```text
<bos>, <fam:mosfet|igbt|ic>, STEP_1, STEP_2, ..., STEP_N
```

and is trained with a next-token objective:

```text
predict token[t + 1] from tokens[:t]
```

The model exposes the same interface as `NGramBaseline`:

```python
predict_topk(family, prefix, k=5)
complete(family, prefix, max_steps=200)
```

That keeps the existing local evaluator, submission writer, and Streamlit app
compatible with either model.

## Configs

Two starting configs are included:

- `configs/transformer_small.yaml`
  - `d_model: 128`
  - `n_layers: 4`
  - `n_heads: 4`
  - `dropout: 0.1`
  - `max_len: 180`
  - `batch_size: 128`

- `configs/transformer_medium.yaml`
  - `d_model: 256`
  - `n_layers: 6`
  - `n_heads: 8`
  - `dropout: 0.1`
  - `max_len: 180`
  - `batch_size: 256`

`max_len=180` covers the current official training routes, whose longest route
is below this context length after adding the family token.

## Usage

Create or refresh the local dev split first:

```bash
python scripts/make_dev_split.py --force
```

Fast smoke test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_transformer.py \
  --limit-train-sequences 2 \
  --limit-dev-sequences 2 \
  --epochs 1 \
  --skip-eval \
  --d-model 16 \
  --n-layers 1 \
  --n-heads 2 \
  --dim-feedforward 32 \
  --batch-size 2 \
  --model-path models/transformer_smoke.pt \
  --device cpu
```

Small full run:

```bash
python scripts/train_transformer.py \
  --config configs/transformer_small.yaml \
  --model-path models/transformer_small.pt \
  --metrics-path artifacts/transformer_metrics.json
```

GPU/Leonardo-style run:

```bash
python scripts/train_transformer.py \
  --config configs/transformer_medium.yaml \
  --device cuda \
  --model-path models/transformer_medium.pt \
  --metrics-path artifacts/transformer_medium_metrics.json
```

Evaluate any saved model:

```bash
python src/eval/local_eval.py --model models/transformer_small.pt
```

Write dev submission files from the Transformer:

```bash
python scripts/predict_submission.py \
  --model models/transformer_small.pt \
  --eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
  --eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
  --out-dir extras/results_transformer
```

## Pitch line

> We compare a memorization-prone n-gram baseline against a trained
> family-conditioned Transformer, then constrain anomaly judgment with explicit
> process rules. This lets us separate learned sequence distribution from hard
> manufacturing invariants.


## Local generated-data option

After running:

```bash
make generate-extra-local
```

you can augment the Transformer training split with generated valid routes:

```bash
python scripts/train_transformer.py \
  --config configs/transformer_small.yaml \
  --device cpu \
  --epochs 1 \
  --extra-data-dir data/generated/infineon \
  --model-path models/transformer_small_extra_local.pt \
  --metrics-path artifacts/transformer_small_extra_local_metrics.json
```

The same `--extra-data-dir` flag is supported by `scripts/train_ngram.py`.
