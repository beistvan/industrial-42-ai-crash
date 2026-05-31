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

**Verify:** check `result/submission/*.csv` row counts and `artifacts/sweeps/LEADERBOARD_FINAL.csv`.

---

## Script map

### Core

| Script | Purpose |
|---|---|
| `train_transformer.py` | Train one checkpoint (vanilla or modern arch) |
| `train_ngram.py` | N-gram baseline |
| `sweep_transformer.py` | Expand a sweep YAML row into CLI args |
| `predict_submission.py` | Write judge CSVs (nextstep / completion / anomaly) |
| `summarize_runs.py` | Build `LEADERBOARD_FINAL.{csv,md}` from metrics JSONs |
| `sweep_picks.py` | Pick best T1/T2 runs (uses checkpoint `best_value`) |
| `regenerate_submission.sh` | Pick T1/T2 + predict on GPU (requires CUDA) |
| `make_dev_split.py` | Build dev holdout |
| `generate_extra_sequences.py` | Synthetic augmentation |
| `check_environment.py` | Torch / deps smoke check |
| `rehearsal_train.py` | Pre-flight tiny train+eval before long sweeps |
| `validate_artifacts.py` | Schema checks on dev CSVs and metrics JSON |
| `run_eval_matrix.py` | 4-arm eval matrix (baseline → hybrid) |
| `smoke_test.py` | End-to-end CPU smoke after `make smoke` |

### Makefile shortcuts

```bash
make dev-split && make train-ngram && make validate-artifacts
make rehearsal-train                # optional before new training runs
make eval-matrix                    # after checkpoints exist
make leaderboard-final            # refresh LEADERBOARD_FINAL.csv
make regenerate-submission        # GPU predict → result/submission/
make dashboard                      # unified Streamlit UI
```

Winning recipes: [`configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md)  
Pipeline workflow: [`docs/ENGINEERING_PRACTICES.md`](../docs/ENGINEERING_PRACTICES.md)

---

## Regenerate submission (if picks change)

```bash
make leaderboard-final
make regenerate-submission
```

Task 3 (`anomaly.csv`) follows T1 automatically — no separate retrain.

---

## Logs

Training and predict logs (if any) go under `logs/` (gitignored).
