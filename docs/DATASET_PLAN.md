# Dataset Plan — Industrial: Models that learn how processes unfold

The official Infineon training data is already vendored in this repo. For
the full data contract and refresh recipe see
[`DATA_SPEC.md`](./DATA_SPEC.md); for an upstream README see
[`../data/raw/infineon/SOURCE.md`](../data/raw/infineon/SOURCE.md).

## Source

Official Industrial AI track from
[`Lumos-Data/zero_one_hack_01`](https://github.com/Lumos-Data/zero_one_hack_01/tree/main/tracks/industrial-infineon),
vendored locally under `data/raw/infineon/` (PR #1).

## Layout in the repo

```
data/raw/infineon/
├── SOURCE.md                              ← provenance + refresh recipe
├── reference/                             ← 9 canonical / enriched CSVs
└── training_data/                         ← 3 _variants.csv (1,000 seq/fam)
    ├── generate_sequences.py              ← official generator + validator
    ├── generation_rules.md                ← grammar + 10 rules + eval protocol
    └── *.csv
```

## Data contract used by `src/`

- `Sequences = dict[str, list[str]]` (sequence id → ordered step tokens).
- Long-format `_variants.csv` parsed via the official
  `read_csv_sequences` (wrapped in `src/data/infineon_loader.py`).
- `Vocabulary` reserves PAD/BOS/EOS/UNK + per-family tokens
  (`<fam:mosfet>`, `<fam:igbt>`, `<fam:ic>`) at low ids; remaining ~120
  step tokens are sorted deterministically.

## Splits

`scripts/make_dev_split.py` persists deterministic splits under
`data/processed/splits/` so every model (n-gram, transformer, any HF
fine-tune) and every evaluator reads the same dev set. Default split:
900 train / 100 dev per family.

## Synthetic augmentation

The official `generate_sequences.py` CLI can produce unlimited extra valid
sequences. Use this for the scaling experiments (see `MODEL_PLAN.md`):

```bash
python data/raw/infineon/training_data/generate_sequences.py \
  --family mosfet --count 10000 --seed 101 \
  --output data/raw/infineon/training_data/MOSFET_extra_10k.csv
```

## Eval inputs (organizer-distributed, not in this checkout)

When kickoff drops the official eval CSVs, place them at:

- `data/raw/infineon/eval/eval_input_valid.csv` (Tasks 1 + 2)
- `data/raw/infineon/eval/eval_input_anomaly.csv` (Task 3)

`scripts/predict_submission.py` reads these and writes
`extras/results/{nextstep,completion,anomaly}.csv` for submission.

## Anti-leakage / reproducibility

- `make_dev_split.py` is deterministic for a given `--seed`.
- Train and dev id lists are persisted as JSON for cross-evaluator
  agreement.
- Vocabulary is sorted; no run-to-run drift.
- The rule validator is the official one; we never re-implement it.

## What this replaces

This file used to suggest fallback datasets (NASA turbofan, SECOM
manufacturing). That fallback is no longer relevant — the official
Infineon corpus is in the repo and the synthetic generator gives
unlimited augmentation.
