# Team handoff — 42 AI Crash / Infineon track

## Submission models (hybrid — 2 checkpoints, 3 CSV files)

| Task | File | Model checkpoint | Role |
|---|---|---|---|
| Task 1 — next-step | `result/submission/nextstep.csv` | Best dev MRR run (currently `h_mod_nosched_mrr.pt.best`) | Transformer top-5 ranks |
| Task 2 — completion | `result/submission/completion.csv` | Best dev token-acc run (currently `g_drop15_nosched_t2.pt.best`) | Beam completion tail |
| Task 3 — anomaly | `result/submission/anomaly.csv` | **Same checkpoint as Task 1** | Rule validator + LM SCORE |

Task 3 has **no separate sweep or checkpoint**. Detection and rule ID come from
`validate_sequence` (fixed). The continuous `SCORE` column uses the Task-1 model’s
teacher-forced log-probability (`src/eval/anomaly_scoring.py`). When a new T1 leader
appears, rerun predict for T1 (which also writes `anomaly.csv`). A new T2 leader only
changes `completion.csv`.

Paths (current hybrid):

```
models/sweeps/h_mod_nosched_mrr.pt.best       # Task 1 + Task 3 SCORE
models/sweeps/g_drop15_nosched_t2.pt.best     # Task 2 only
```

Regenerate from current leaderboard (Leonardo login node — submits GPU predict jobs via Slurm):

```bash
make leonardo-leaderboard-final   # if new metrics JSONs exist
make regenerate-submission        # auto Slurm when no local CUDA
```

On a machine with a local GPU: `FORCE_LOCAL=1 make regenerate-submission`.

Do **not** load full Transformer checkpoints on the Leonardo login node for scoring —
they OOM (exit 137). Use `make regenerate-submission` (Slurm) or `sbatch scripts/leonardo/05_predict_transformer.slurm`.

### When a new T1 or T2 leader appears

1. Confirm `models/sweeps/<run>.pt.best` and metrics JSON exist.
2. `make leonardo-leaderboard-final` — check the run is top for T1 MRR and/or T2 token acc.
3. `make regenerate-submission` — writes all three CSVs (T1 job includes anomaly).
4. Optionally refresh `src/app/track_context.py` `SUBMISSION` if you want static UI copy to match.

No changes needed to `rule_validator`, `anomaly_scoring`, or Task-3 tests when models change.

---

## Headline numbers (local dev holdout — *not* official organizer eval)

Eval set: `data/processed/dev_eval/` (600 Task-1/2 items, injected anomalies for Task-3).
Full table: `artifacts/sweeps/LEADERBOARD_FINAL.csv` (27 runs).

### Baseline → trained → submission

| Stage | Model | Task-1 MRR | Task-1 Top-1 | Task-2 token acc | Task-3 F1 |
|---|---|---:|---:|---:|---:|
| Level 1 baseline (n-gram) | `ngram_baseline.pkl` | 0.807 | 0.687 | 0.421 | 1.00 |
| Level 2 trained (Wave 1 T1) | `f_drop15_100_mrr` | 0.873 | 0.748 | 0.437 | 1.00 |
| Level 2 optimized (Wave 2 T2) | `g_drop15_nosched_t2` | 0.867 | 0.738 | **0.455** | 1.00 |
| **Submission hybrid** | `h_mod_nosched_mrr` + T2 above | **0.874** | **0.75** | **0.455** | 1.00 |

### Lift vs n-gram

| Metric | Baseline → submission | Δ |
|---|---|---|
| Task-1 MRR | 0.807 → 0.874 | **+8.3% relative** |
| Task-1 Top-1 | 68.7% → 75.0% | **+6.3 pp** |
| Task-2 token acc | 0.421 → 0.455 | **+3.4 pp** |

Task-1 Top-5: **100%** on strong Transformer runs.

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
| dropout | 0.15 (T1) / 0.15, no scheduler (T2 Wave 2) |
| ~params | ~4M |

**Training data**: 3,000 real Infineon sequences (MOSFET/IGBT/IC) + 250 generated valid routes per family (`data/generated/infineon/`).

**Decoding**: rule-constrained beam search for Task 2 (`--rule-constrained --beam-width 5 --candidate-pool 5`).

**Task 3**: official Infineon rule validator (`classify_sequence`), not a neural anomaly head.

**Reference parameters**: `reference/*_parameters.csv` shown in the Streamlit demo via `step_metadata.py`; training uses STEP tokens only (Waves 1–4).

---

## How each submission model was built

### Task 1 — `h_mod_nosched_mrr` (Wave 3)

100 epochs, modern arch (RoPE + RMSNorm + SwiGLU), dropout 0.15, no scheduler,
extras 1×, AMP, label smoothing 0.1. Best checkpoint at **epoch 85 by dev_mrr**.

Config: `configs/sweeps/leonardo_modern.yaml` → row `h_mod_nosched_mrr` (row 4).

Supersedes Wave 1 leader `f_drop15_100_mrr` (+0.05pp MRR).

### Task 2 — `g_drop15_nosched_t2` (Wave 2)

Fine grid around Wave 1 winner: dropout 0.15, **no LR scheduler**, save by `dev_token_acc`.
Best at **epoch 80, tok_acc 0.4545** (+0.3 pp over Wave 1 T2 specialist `f_extras_1x_100_t2`).

Config: `configs/sweeps/leonardo_fine.yaml` → row `g_drop15_nosched_t2`.

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

# 2) Task 1 leader (Wave 3)
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_modern.yaml --stage finalists --row 4

# 3) Task 2 leader (Wave 2)
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_fine.yaml --stage finalists --row 4

# On Leonardo: bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_modern.yaml finalists --row 4
make leonardo-leaderboard-final
```

### Generate official submission CSVs

```bash
make regenerate-submission
# or manually:
bash scripts/regenerate_submission.sh
```

### Score a checkpoint locally (dev only)

```bash
python -m src.eval.local_eval \
  --model models/sweeps/h_mod_nosched_mrr.pt.best \
  --device cuda \
  --out artifacts/local_eval_h_mod_nosched_mrr.json
```

---

## Pipeline status (submission final)

| Wave | Status | Best pick |
|---|---|---|
| Wave 1 | done | `f_drop15_100_mrr` (T1-era), `f_extras_1x_100_t2` (T1-era T2) |
| Wave 2 | done | `g_drop15_nosched_t2` (T2) |
| Wave 3 | done | **`h_mod_nosched_mrr` (T1, MRR 0.874)** |
| Wave 4 | done | no gain vs Wave 2 T2 |
| Wave 5 | skipped | not needed |
| Wave 6 | skipped | optional experiment |

### Optional HPC appendix (historical)

```bash
make leonardo-leaderboard-final
make regenerate-submission
make run-dashboard
```

See `docs/LEONARDO_GPU_RUNBOOK.md` for full Leonardo reproduction.

---

## Key repo files for teammate

| File | Purpose |
|---|---|
| `REPORT.md` | Jury-facing write-up + Track 1 compliance |
| `artifacts/sweeps/LEADERBOARD_FINAL.csv` | All run numbers (27 rows) |
| `result/submission/*.csv` | Official submission outputs |
| `scripts/regenerate_submission.sh` | Re-pick T1/T2 from leaderboard + predict |
| `EVAL_DATA/` | Organizer eval inputs |
| `docs/LEONARDO_GPU_RUNBOOK.md` | Leonardo reproduction runbook |
| `configs/sweeps/leonardo_fine.yaml` | Wave-2 fine grid (completed) |
| `configs/sweeps/leonardo_modern.yaml` | Wave-3 modern architecture |
| `configs/sweeps/leonardo_task2.yaml` | Wave-4 Task-2 prefix training |
| `SLIDES.md` | 10-slide pitch (Marp-renderable) |
