# Team handoff — 42 AI Crash / Infineon track

## Submission models (hybrid — 2 checkpoints, 3 CSV files)

| Task | File | Model checkpoint | Role |
|---|---|---|---|
| Task 1 — next-step | `extras/results_submission/nextstep.csv` | `f_drop15_100_mrr.pt.best` | Best dev MRR |
| Task 2 — completion | `extras/results_submission/completion.csv` | `f_extras_1x_100_t2.pt.best` | Best dev token accuracy |
| Task 3 — anomaly | `extras/results_submission/anomaly.csv` | Same run as Task 1 | Rule validator + continuous SCORE |

Paths:

```
models/sweeps/f_drop15_100_mrr.pt.best      # Task 1 + Task 3
models/sweeps/f_extras_1x_100_t2.pt.best    # Task 2 only
```

---

## Headline numbers (local dev holdout — *not* official organizer eval)

Eval set: `data/processed/dev_eval/` (600 Task-1/2 items, injected anomalies for Task-3).
Full table: `artifacts/sweeps/LEADERBOARD_FINAL.csv`

### Baseline vs final

| Model | Task-1 MRR | Task-1 Top-1 | Task-2 token acc | Task-3 F1 |
|---|---:|---:|---:|---:|
| N-gram baseline | 0.807 | 0.687 | 0.421 | 1.00 |
| Submission T1 (`f_drop15_100_mrr`) | **0.873** | **0.748** | 0.437 | 1.00 |
| Submission T2 (`f_extras_1x_100_t2`) | 0.870 | 0.743 | **0.451** | 1.00 |

### Lift vs n-gram

| Metric | Baseline → submission | Δ |
|---|---|---|
| Task-1 MRR | 0.807 → 0.873 | **+8.2% relative** |
| Task-1 Top-1 | 68.7% → 74.8% | **+6.1 pp** |
| Task-2 token acc | 0.421 → 0.451 | **+3.0 pp** |

Task-1 Top-5: **100%** on both Transformer runs.

---

## What the model is

Architecture: **family-conditioned decoder-only Transformer** (from scratch, no LLM).

| Hyperparam | Value |
|---|---|
| d_model | 256 |
| n_layers | 6 |
| n_heads | 8 |
| dim_feedforward | 1024 |
| max_len | 180 |
| dropout | 0.15 (T1) / 0.10 (T2, medium default) |
| ~params | ~4M |

**Training data**: 3,000 real Infineon sequences (MOSFET/IGBT/IC) + 250 generated valid routes per family (`data/generated/infineon/`).

**Decoding**: rule-constrained top-5 greedy for Task 2 (`--rule-constrained --candidate-pool 5`).

**Task 3**: official Infineon rule validator (`classify_sequence`), not a neural anomaly head.

---

## How each submission model was built

### Task 1 — `f_drop15_100_mrr`

Wave-1 finalist: 100 epochs, dropout 0.15, extras 1×, cosine scheduler, AMP, label smoothing 0.1.
Best checkpoint saved at **epoch 84 by dev_mrr**.

Config row in `configs/sweeps/leonardo_final.yaml`:

```yaml
- run_name: f_drop15_100_mrr
  dropout: 0.15
```

### Task 2 — `f_extras_1x_100_t2`

Same recipe as above but dropout 0.10 (medium default), 100 epochs, checkpoint saved by `dev_token_acc` (best at **epoch 65, tok_acc 0.451**).

Config row:

```yaml
- run_name: f_extras_1x_100_t2
  save_best_by: dev_token_acc
```

---

## Reproduce from code

### Local setup (CPU smoke / n-gram)

```bash
git clone -b wave1-submission https://github.com/beistvan/industrial-42-ai-crash.git
cd industrial-42-ai-crash
python -m venv .venv && source .venv/bin/activate
make setup-cpu
make dev-split && make train-ngram && make smoke
```

### Train submission models (Leonardo GPU)

```bash
# 1) Dev split + synthetic extras (once)
python scripts/make_dev_split.py --force
python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

# 2) Wave 1 (6 parallel jobs) — or train single runs:
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_final.yaml --stage finalists --row 2   # f_drop15_100_mrr
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_final.yaml --stage finalists --row 1   # f_extras_1x_100_t2
# On Leonardo: make leonardo-wave1
```

Base architecture: `configs/transformer_medium.yaml`
Sweep matrix: `configs/sweeps/leonardo_final.yaml`

### Generate official submission CSVs

```bash
MODEL_T1=models/sweeps/f_drop15_100_mrr.pt.best
MODEL_T2=models/sweeps/f_extras_1x_100_t2.pt.best

python scripts/predict_submission.py \
  --model "$MODEL_T1" \
  --eval-valid EVAL_DATA/eval_input_valid.csv \
  --eval-anomaly EVAL_DATA/eval_input_anomaly.csv \
  --out-dir extras/results_submission \
  --rule-constrained --beam-width 5 --candidate-pool 5 --device cuda

python scripts/predict_submission.py \
  --model "$MODEL_T2" \
  --eval-valid EVAL_DATA/eval_input_valid.csv \
  --out-dir extras/results_submission_t2 \
  --rule-constrained --beam-width 5 --candidate-pool 5 --device cuda

cp extras/results_submission_t2/completion.csv extras/results_submission/completion.csv
```

### Score a checkpoint locally (dev only)

```bash
python -m src.eval.local_eval \
  --model models/sweeps/f_drop15_100_mrr.pt.best \
  --device cuda \
  --out artifacts/local_eval_f_drop15_100_mrr.json
```

---

## Key repo files for teammate

| File | Purpose |
|---|---|
| `REPORT.md` | Jury-facing write-up |
| `RESULTS_GPU_SUMMARY.md` | GPU sweep summary |
| `artifacts/sweeps/LEADERBOARD_FINAL.csv` | All run numbers |
| `extras/results_submission/*.csv` | Official submission outputs |
| `EVAL_DATA/` | Organizer eval inputs |
| `scripts/predict_submission.py` | Submission writer |
| `docs/LEONARDO_GPU_RUNBOOK.md` | Leonardo reproduction runbook |
| `configs/sweeps/leonardo_final.yaml` | Wave-1 finalist sweep matrix |
| `configs/sweeps/leonardo_fine.yaml` | Wave-2 fine grid (scaffolded, not run) |
| `SLIDES.md` | 10-slide pitch (Marp-renderable) |
