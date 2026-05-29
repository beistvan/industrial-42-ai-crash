# Local Infineon dev/eval split

This repo creates a local validation set from the official Infineon training CSV
files. It is independent from the n-gram and transformer implementations, so it
is safe to run while model files are being edited.

## Generate

```bash
python scripts/make_dev_split.py --force
# or
make dev-split
```

Default split:

- train: 900 sequences per family
- dev: 100 sequences per family
- seed: 42
- families: `mosfet`, `igbt`, `ic`

## Outputs

```text
data/processed/splits/train_ids.json
data/processed/splits/dev_ids.json
data/processed/splits/split_summary.json

data/processed/dev_eval/eval_input_valid_dev.csv
data/processed/dev_eval/eval_input_valid_dev_gold.csv
data/processed/dev_eval/eval_input_anomaly_dev.csv
data/processed/dev_eval/eval_input_anomaly_dev_gold.csv
```

`eval_input_valid_dev.csv` contains the Task 1/2 local inputs. Each dev sequence
is truncated at 60% and 80%, so the file contains:

```text
100 sequences × 3 families × 2 cut points = 600 partial sequences
```

`eval_input_anomaly_dev.csv` contains 300 valid and 300 injected-invalid
sequences. Invalid examples are only accepted if the official Infineon validator
marks the mutated sequence invalid.

## CSV schemas

### Task 1/2 input and gold

Both files use long sequence format:

```text
SEQUENCE_ID,FAMILY,STEP
```

The input file contains truncated partial sequences. The gold file contains the
matching full sequences. The evaluator gets the next-step target by comparing the
partial length with the full sequence.

### Task 3 input

```text
SEQUENCE_ID,FAMILY,STEP
```

The sequence is valid iff `src.eval.rule_validator.classify_sequence()` returns
zero official rule violations.

### Task 3 gold

```text
SEQUENCE_ID,FAMILY,VALID,INJECTED_LABEL,DETECTED_RULES
```

`VALID=1` means valid; `VALID=0` means anomaly/invalid. `DETECTED_RULES` is the
official validator's rule list for local attribution scoring.

## Baseline run

```bash
python scripts/make_dev_split.py --force
python scripts/train_ngram.py --max-order 12
python src/eval/local_eval.py
```

The main baseline training script already evaluates after fitting and writes:

```text
artifacts/ngram_metrics.json
models/ngram_baseline.pkl
```

`src/eval/local_eval.py` is a convenience wrapper that reloads the saved baseline
and writes:

```text
artifacts/local_eval_metrics.json
```

## Submission-like prediction files

For a local dry run:

```bash
python scripts/predict_submission.py \
  --model models/ngram_baseline.pkl \
  --eval-valid data/processed/dev_eval/eval_input_valid_dev.csv \
  --eval-anomaly data/processed/dev_eval/eval_input_anomaly_dev.csv \
  --out-dir extras/results
```

This writes:

```text
extras/results/nextstep.csv
extras/results/completion.csv
extras/results/anomaly.csv
```
