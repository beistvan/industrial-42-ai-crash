# Data Spec — Infineon Industrial AI track

This is the data spec for the Infineon track. Source-of-truth for the
process grammar is
[`data/raw/infineon/training_data/generation_rules.md`](../data/raw/infineon/training_data/generation_rules.md).

## Source

Vendored from the official track repo:
https://github.com/Lumos-Data/zero_one_hack_01/tree/main/tracks/industrial-infineon

See [`data/raw/infineon/SOURCE.md`](../data/raw/infineon/SOURCE.md) for the
exact layout and refresh instructions.

## What the task is

Symbolic sequence modeling over fabrication process steps. Each "sample" is an
ordered sequence of step tokens drawn from a fixed ~120-token vocabulary. The
model must learn process *logic* (ordering, dependencies, forbidden patterns)
— not sensor dynamics.

## Files

### Bulk training data — `data/raw/infineon/training_data/`

| File | Format | Shape |
| --- | --- | --- |
| `MOSFET_variants.csv` | long (`SEQUENCE_ID, STEP`) | 1,000 sequences, ~125k rows |
| `IGBT_variants.csv`   | long (`SEQUENCE_ID, STEP`) | 1,000 sequences, ~148k rows |
| `IC_variants.csv`     | long (`SEQUENCE_ID, STEP`) | 1,000 sequences, ~115k rows |

Long format means one row per `(sequence, step)`:

```csv
SEQUENCE_ID,STEP
seq_0001,RECEIVE WAFER LOT
seq_0001,LOT IDENTIFICATION
seq_0001,INITIAL WAFER INSPECTION
...
seq_0002,RECEIVE WAFER LOT
...
```

### Reference recipes — `data/raw/infineon/reference/`

One canonical sequence per family (single-column `STEP`), plus enriched
variants that add `DESCRIPTION` and `REALISTIC FAB-LEVEL PARAMETERS`. Use
these for human-readable inspection and for future text/parameter-aware
modeling variants. The scored eval only needs the step tokens.

### Tooling

- `data/raw/infineon/training_data/generate_sequences.py` — generate or
  validate sequences against the process grammar.
- `data/raw/infineon/training_data/generation_rules.md` — authoritative
  grammar reference: vocabulary (§1), block notation per family (§2),
  **10 forbidden patterns** (§3), variation axes (§4), eval protocol (§5).

## Core invariants

- Every valid sequence starts with `RECEIVE WAFER LOT` and ends with
  `SHIP LOT`.
- Sequence lengths: IC ~107, MOSFET ~126, IGBT ~151 steps.
- Vocabulary is shared across families (~120 distinct tokens); families differ
  in which optional blocks appear and how many cycles.
- Anomalies in Task 3 are **rule violations** (one of 10 documented patterns),
  not numeric outliers.

## Internal data contract for code in `src/`

The loader (`src/data/loader.py`, to be written) returns:

```python
Sequences = dict[str, list[str]]   # sequence_id -> ordered list of step tokens
```

The vocabulary helper builds a deterministic token↔id map from the union of
tokens across all three loaded families. `<bos>` (id 0) and `<eos>` (id 1) are
reserved for sequence-model training; the dataset tokens never collide with
them.

## Eval inputs (not yet checked in)

Organizers distribute these at kickoff. Drop them at:

- `data/raw/infineon/eval/eval_input_valid.csv` — Tasks 1 & 2 (600 partial
  sequences truncated at 60% / 80%).
- `data/raw/infineon/eval/eval_input_anomaly.csv` — Task 3 (987 mixed valid /
  rule-violating sequences, unlabeled).

The scoring script `eval_metrics.py` is also organizer-provided and runs
without external dependencies.

