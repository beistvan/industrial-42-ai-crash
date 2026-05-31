# Scripts — current situation & runbook

Last updated: **2026-05-31 ~04:00 CEST** (Leonardo login node)

## Quick status

| Item | Value |
|---|---|
| **T1 + Task 3** | `h_mod_nosched_mrr` — dev MRR **0.8735** (Wave 3 modern, nosched) |
| **T2** | `g_drop15_nosched_t2` — tok **0.4545** (Wave 2, still unbeaten) |
| **Submission CSVs** | `result/submission/{nextstep,completion,anomaly}.csv` — **regenerated** (Slurm predict ~04:00) |
| **GPU jobs** | ~13 `zoh-sweep` tasks (Wave 4 T2, Wave 3b tune, Wave 5 params, lite T2) |
| **Watcher** | `wait_wave34_and_finalize.sh` — auto-rebuild when queue drains |

**One-shot snapshot:** `bash scripts/leonardo/status_now.sh`

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

### Leonardo / Slurm (`scripts/leonardo/`)

| Script | Purpose |
|---|---|
| **`status_now.sh`** | **One-shot queue + picks + CSV check** ← run this often |
| `monitor_waves.sh` | Background poller (every 3 min) → `logs/monitor-waves.out` |
| `submit_sweep.sh` | Submit a sweep YAML as Slurm array |
| `sweep_array.slurm` | One GPU task per array row |
| `05_predict_transformer.slurm` | GPU predict job (used by regenerate_submission) |
| `wait_wave34_and_finalize.sh` | Wait for sweeps → leaderboard → regenerate |
| `wait_wave2_and_finalize.sh` | Wave 2 finalize (legacy) |
| `wave_orchestrator.sh` | Full Wave 2→3→4 pipeline |
| `pipeline_watch.sh` | Auto-rebuild leaderboard every 5 min while jobs run |
| `queue_parallel_waves.sh` | Submit Waves 3+4+6 in parallel |
| `wave5_if_needed.sh` | Gate Wave 5 on plateau |
| `fetch_results.sh` | Pull artifacts from Leonardo |
| `setup_env.sh` / `load_env.sh` | HPC venv + modules |

### Makefile shortcuts

```bash
make leonardo-leaderboard-final   # refresh LEADERBOARD_FINAL.csv
make regenerate-submission        # Slurm predict → result/submission/
make leonardo-wave3-tune          # h_mod_nosched_drop12 + h_mod_nosched_120
# Streamlit dashboard (single entry point — src/app/dashboard.py)
make run-dashboard              # default port 8501
STREAMLIT_PORT=8502 make run-dashboard
bash scripts/leonardo/status_now.sh
```

---

## Active GPU work (2026-05-31)

| Job array | YAML | Runs | Goal |
|---|---|---|---|
| 43148356 | `leonardo_task2.yaml` | Wave 4 T2 specialists | Beat T2 0.455 |
| 43158553 | `leonardo_modern_tune.yaml` | `h_mod_nosched_drop12`, `h_mod_nosched_120` | Push T1 past 0.8735 |
| 43160050 | `leonardo_params.yaml` | Wave 5 param enrichment | Fallback if plateau |
| 43162525 | `leonardo_lite.yaml` | `h_mod_nosched_t2` etc. | Modern T2 with nosched |

Wave 3 core (`h_mod_*`) is **done**. Best T1: `h_mod_nosched_mrr`.

---

## When a new leader appears

```bash
make leonardo-leaderboard-final
make regenerate-submission          # login node → Slurm GPU predict
bash scripts/leonardo/status_now.sh # verify CSVs
```

Task 3 (`anomaly.csv`) follows T1 automatically — no separate retrain.

---

## Logs

| Log | Content |
|---|---|
| `logs/slurm-sweep-<job>_<row>.out` | Per-run training |
| `logs/inf-predict-tr-<job>.out` | Submission predict |
| `logs/wave34_finalize.log` | Auto-finalize watcher |
| `logs/monitor-waves.out` | Background monitor |
| `logs/submitted_jobs.txt` | Slurm job IDs |
