# Engineering practices — pipeline workflow

How this repo trains, validates, and submits models for the Infineon Industrial AI
track. Describes the workflow as implemented in code — not a wish list.

**Track:** Zero One Industrial AI (Infineon) — fab process sequence prediction
(~4M-param from-scratch Transformer, 10-rule grammar for anomaly detection).

---

## Design principles

1. **Deterministic authority where the grammar is exact.** Task 3 detection and
   rule ID come from the organizer `validate_sequence()` — not from a learned
   classifier. The T1 checkpoint supplies a continuous LM **SCORE** only.

2. **Learned models where statistics matter.** Tasks 1 and 2 use a compact
   decoder-only Transformer on the domain vocabulary (~140 step tokens per family).

3. **Hybrid specialists, not one checkpoint for everything.** The config that
   maximizes dev MRR is not the config that maximizes dev token-accuracy or NED.
   We train both and use each on the task it wins.

4. **Bounded inputs, structured outputs, honest eval.** Dev holdout is fixed and
   persisted; every run writes metrics JSON; submission CSVs match organizer format.

5. **No silent failures.** Missing checkpoints, empty arms, and skipped evals
   are recorded explicitly (`status: unavailable`, `status: skipped`) so
   leaderboards and regen scripts never degrade quietly.

We train from scratch — no HuggingFace pretrained LLMs. Rationale:
`docs/ADRs/0001-no-hf-pretrained.md`.

---

## Pipeline overview

```
make dev-split
    → data/processed/splits/ + dev_eval CSVs

make train-ngram
    → models/ngram_baseline.pkl + artifacts/ngram_metrics.json

make validate-artifacts          # schema checks on CSVs + JSON

make rehearsal-train             # tiny Transformer smoke before Slurm
    → models/rehearsal.pt + artifacts/rehearsal_metrics.json

submit_sweep.sh (Slurm array)    # one YAML row = one GPU job
    → models/sweeps/*.pt.best + artifacts/sweeps/*.json

make leonardo-leaderboard-final
    → LEADERBOARD_FINAL.{csv,md}

make eval-matrix                 # 4-arm before/after on dev holdout
    → artifacts/eval_matrix_latest.json

make regenerate-submission
    → result/submission/{nextstep,completion,anomaly}.csv
```

Any model implementing `predict_topk()` and `complete()` drops into
`src/eval/run_eval.evaluate_all()` without changing task logic.

---

## Core practices

| Practice | Command / artifact | Purpose |
|---|---|---|
| **Persisted dev split** | `make dev-split` | Same holdout for n-gram, Transformer, and all evaluators |
| **Artifact validation** | `make validate-artifacts` | Required keys on dev CSVs, sweep metrics JSON, split ids |
| **Training rehearsal** | `make rehearsal-train` · `REHEARSE=1 submit_sweep.sh` | End-to-end train+eval on tiny model before GPU array |
| **Sweep matrix** | `configs/sweeps/leonardo_*.yaml` + `sweep_array.slurm` | Reproducible hyperparameter rows; one job per row |
| **4-arm eval matrix** | `make eval-matrix` · dashboard **Eval matrix** tab | Pitch-ready before/after comparison |
| **Graceful skip** | `local_eval --allow-missing` · matrix `status: unavailable` | Explicit stubs when a checkpoint is absent |
| **Checkpoint status** | `summarize_runs.py` → `checkpoint_status` column | Leaderboard flags missing `.pt.best` files |
| **Business-aligned metric** | NED headline for Task 2 | Lower normalized edit distance = closer fab recipe |
| **Floor baselines** | N-gram dev numbers, T3 validator ceiling | Documented limits in `REPORT.md` and `FLOOR_BASELINES` |

### 4-arm eval matrix

Structured comparison on the dev holdout for demos and the dashboard:

| Arm | Stack | Role |
|---|---|---|
| **A** | N-gram suffix-backoff (order 12) | Level 1 baseline floor |
| **B** | `h_mod_nosched_mrr` | T1 specialist (Wave 3 modern) |
| **C** | `g_drop15_nosched_t2` | T2 specialist + rule-constrained beam |
| **D** | T1 + T2 hybrid | **Shipped submission** |

Implementation: `src/eval/eval_matrix.py`, `scripts/run_eval_matrix.py`.

### Schema validation

Lightweight checks (no external schema dependency) on:

- Long-format dev CSVs: `SEQUENCE_ID`, `FAMILY`, `STEP`
- Anomaly gold CSVs: `SEQUENCE_ID`, `FAMILY`, `VALID`
- Sweep metrics JSON: `run_name`, task1/2/3 overall metrics, optional `history`
- Local eval payloads: full metrics or explicit `status: skipped`

Implementation: `src/eval/schema_validation.py`, `scripts/validate_artifacts.py`.
Wired into `make smoke` via `scripts/smoke_test.py`.

### Separation of concerns (rules vs model)

| Layer | Component | Role |
|---|---|---|
| Deterministic authority | `validate_sequence()` | Task 3 IS_VALID + PREDICTED_RULE |
| Learned prediction | Transformer checkpoints | Task 1 next-step, Task 2 completion |
| Continuous score | T1 LM log-prob | Task 3 SCORE column in `anomaly.csv` |
| Eval comparison | 4-arm matrix | Baseline → specialists → hybrid |

Do not ask the learned model to do what the grammar already solves exactly.

---

## Operational checklist (before burning GPU hours)

```bash
# 1. Data + baseline integrity
make dev-split
make train-ngram
make validate-artifacts

# 2. Pre-flight training loop
make rehearsal-train              # GPU ~5 min
# make rehearsal-train-cpu        # CPU ~15 min fallback

# 3. Submit sweep (optional rehearsal gate)
REHEARSE=1 bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_modern.yaml finalists

# 4. After checkpoints exist
make eval-matrix
make leonardo-leaderboard-final
make regenerate-submission

# 5. Full smoke
make smoke                        # dev-split + n-gram + pytest + artifact check
```

**Dashboard:** `make run-dashboard` → Overview, **Eval matrix**, Leaderboard,
Training & config, Live demo.

---

## Leonardo-specific guards

- **Slurm array concurrency:** `SWEEP_CONCURRENCY` caps simultaneous jobs
  (`scripts/leonardo/submit_sweep.sh`).
- **Login-node OOM:** do not load full Transformer checkpoints on the login node
  for scoring — use `make regenerate-submission` (Slurm GPU predict). See
  `HANDOFF.md`.
- **Wave discipline:** shortlist → finalists → fine grid; pick submission rows
  from `configs/sweeps/WINNING_RECIPES.md`.

---

## Related docs

- [`PIPELINE.md`](PIPELINE.md) — module map and end-to-end loop
- [`SUBMISSION.md`](SUBMISSION.md) — jury deliverables and reproduce commands
- [`LEONARDO_GPU_RUNBOOK.md`](LEONARDO_GPU_RUNBOOK.md) — cluster operations
- [`ADRs/0001-no-hf-pretrained.md`](ADRs/0001-no-hf-pretrained.md) — model choice
- [`../configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md) — submission rows
