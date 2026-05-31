# Scripts — submission runbook

Last updated: **2026-05-31** (submission **final**)

## Submission snapshot

| Item | Value |
|---|---|
| **T1 + Task 3** | `h_mod_nosched_mrr` — dev MRR **0.8735**, Top-1 **0.75** |
| **T2** | `g_drop15_nosched_t2` — tok **0.4545** |
| **Task 3** | hybrid — rule validator + T1 LM `SCORE` (no separate checkpoint) |
| **CSVs** | `result/submission/{nextstep,completion,anomaly}.csv` (601/601/988) |
| **Leaderboard** | 27 runs — `artifacts/sweeps/LEADERBOARD_FINAL.csv` |

**Verify:** `bash scripts/leonardo/status_now.sh`

---

## Script map

### Core (CPU or GPU)

| Script | Purpose |
|---|---|
| `train_transformer.py` | Train one checkpoint (vanilla or modern arch) |
| `train_ngram.py` | N-gram baseline |
| `sweep_transformer.py` | Expand a sweep YAML row into CLI args |
| `predict_submission.py` | Write judge CSVs (nextstep / completion / anomaly) |
| `summarize_runs.py` | Build `LEADERBOARD_FINAL.{csv,md}` from metrics JSONs |
| `sweep_picks.py` | Pick best T1/T2 runs (uses checkpoint `best_value`) |
| `regenerate_submission.sh` | Pick T1/T2 + predict — **Slurm on login node**, local if CUDA |
| `make_dev_split.py` | Build dev holdout |
| `generate_extra_sequences.py` | Synthetic augmentation |
| `check_environment.py` | Torch / deps smoke check |
| `rehearsal_train.py` | Pre-flight tiny train+eval before Slurm sweeps |
| `validate_artifacts.py` | Schema checks on dev CSVs and metrics JSON |
| `run_eval_matrix.py` | 4-arm eval matrix (baseline → hybrid) |
| `smoke_test.py` | End-to-end CPU smoke after `make smoke` |

### Leonardo / Slurm (`scripts/leonardo/`)

| Script | Purpose |
|---|---|
| **`status_now.sh`** | One-shot queue + picks + CSV check |
| `05_predict_transformer.slurm` | GPU predict job (used by regenerate_submission) |
| `submit_sweep.sh` / `sweep_array.slurm` | Submit sweep YAML as Slurm array |
| `wave_orchestrator.sh`, `queue_parallel_waves.sh`, `wait_wave*.sh` | **Historical** — used during hackathon sweeps |

### Makefile shortcuts

```bash
make dev-split && make train-ngram && make validate-artifacts
make rehearsal-train                # optional before new sweeps
make eval-matrix                    # after checkpoints exist
make leonardo-leaderboard-final     # refresh LEADERBOARD_FINAL.csv
make regenerate-submission          # Slurm predict → result/submission/
make dashboard                      # unified Streamlit UI
bash scripts/leonardo/status_now.sh
```

Winning recipes: [`configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md)  
Pipeline workflow: [`docs/ENGINEERING_PRACTICES.md`](../docs/ENGINEERING_PRACTICES.md)

---

## Regenerate submission (if picks change)

```bash
make leonardo-leaderboard-final
make regenerate-submission
bash scripts/leonardo/status_now.sh
```

Task 3 (`anomaly.csv`) follows T1 automatically — no separate retrain.

---

## Logs

| Log | Content |
|---|---|
| `logs/slurm-sweep-<job>_<row>.out` | Per-run training |
| `logs/inf-predict-tr-<job>.out` | Submission predict |
