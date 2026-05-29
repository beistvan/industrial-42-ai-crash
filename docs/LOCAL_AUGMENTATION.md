# Step 6 — local generated-data augmentation

This is the GPU-free part of the data-scaling plan. It uses the official
Infineon generator already vendored under `data/raw/infineon/training_data/` and
writes generated valid routes under `data/generated/infineon/`.

## Generate extra valid sequences locally

```bash
make generate-extra-local
```

Default output:

```text
data/generated/infineon/
  MOSFET_extra_250_seed101.csv
  IGBT_extra_250_seed102.csv
  IC_extra_250_seed103.csv
  manifest.json
```

For a smaller/larger run:

```bash
python scripts/generate_extra_sequences.py --count-per-family 50 --seed 101 --force
python scripts/generate_extra_sequences.py --count-per-family 1000 --seed 101 --force
```

## Train with generated data

N-gram with extra data:

```bash
make train-ngram-extra
```

Transformer local one-epoch CPU run with extra data:

```bash
make train-transformer-small-extra-local
```

The generated sequences are **only added to training**. The dev split and dev
truth stay fixed, so metrics remain comparable across baseline, Transformer,
and extra-data runs.

## Why this matters

The pitch can show a scaling table:

| Run | Training data | Model | Device |
| --- | --- | --- | --- |
| baseline | official train split | n-gram | CPU |
| transformer local | official train split | small Transformer | CPU |
| augmented local | official + generated extras | small Transformer | CPU |
| Leonardo | official + larger generated extras | small/medium Transformer | GPU |

This keeps the local pipeline reproducible while preparing the exact same code
path for Leonardo GPU training later.
