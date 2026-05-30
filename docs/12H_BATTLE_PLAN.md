# 12-Hour Battle Plan — Unlimited GPU Window

Replaces the original "ship the shortlist" plan. Same official submission
target (`EVAL_DATA/` + judge-format CSVs), but with multi-wave parallel
search and a **hybrid submission** (best Task-1 model + best Task-2 model).

Anchor: `m_real_extras_1x` (current dev MRR leader at 50ep, still climbing).
Everything else is trying to beat it.

```
Hour 0────1────2────3────4────5────6────7────8────9────10───11───12
     [Wave1: 6×100ep parallel][Wave2: fine grid][Wave3: 150ep winner]
     [gen 500/family extras────────────────]
                              [batch local_eval all ckpts]
                                    [official predict + hybrid merge]
                                              [REPORT demo slides repo]
```

---

## Hour 0–0.5 — Setup (do once)

On your laptop:
```bash
bash scripts/leonardo/copy_to_leonardo.sh
```

Then SSH in and verify:
```bash
ssh a08trd0x@login01-ext.leonardo.cineca.it
cd $SCRATCH/industrial
source .venv/bin/activate
module load cuda/12.1 2>/dev/null || true

python scripts/check_environment.py --require-torch
ls EVAL_DATA/eval_input_*.csv
python scripts/predict_submission.py --help | head -5
```

Generate +500/family extras in the background (~5 min CPU):
```bash
make leonardo-generate-extras-500   # writes data/generated/infineon_500/
```

## Hour 0.5–2 — Wave 1: 6 parallel finalists (~30–60 min wall)

```bash
export SLURM_ACCOUNT=EUHPC_D30_031
make leonardo-wave1
squeue --me
```

This submits all 6 rows of `configs/sweeps/leonardo_final.yaml`
with array concurrency 12. Key changes vs shortlist:
- **100–150 epochs** (top runs were still climbing at 50)
- **AMP back on** (~2× throughput, safe at batch 256)
- **`eval_task2_every: 5`** so we can pick a Task-2 specialist
- **Two saves-by-mode**: `dev_mrr` and `dev_token_acc` per recipe

Watch progress:
```bash
tail -f logs/slurm-sweep-*.out
```

When the array finishes (or any time):
```bash
make leonardo-leaderboard-final
column -t -s, artifacts/sweeps/LEADERBOARD_FINAL.csv | head -20
```

## Hour 2–4 — Wave 2: fine grid around Wave 1 winners

Edit `configs/sweeps/leonardo_fine.yaml` if Wave 1 surfaced unexpected
winners (e.g. tighten dropout range, adjust lr). Then:

```bash
make leonardo-wave2
```

Optional Wave 2b (only if Wave 2 results are flat) — uncomment the
`g_large_extras` row in `leonardo_fine.yaml` and resubmit. 1–2 large
jobs only; they're ~3× slower.

## Hour 4–5 — Score everything, pick winners

```bash
mkdir -p artifacts/local_eval
for ckpt in models/sweeps/f_*.pt.best models/sweeps/g_*.pt.best; do
  name=$(basename "$ckpt" .pt.best)
  python -m src.eval.local_eval \
    --model "$ckpt" --device cuda \
    --out "artifacts/local_eval/${name}.json" &
done
wait

# Combined picture
make leonardo-leaderboard-final
```

Decision table:

| Role | Pick by | Default fallback |
|---|---|---|
| Task 1 (`nextstep.csv`) | highest dev MRR | `m_real_extras_1x` |
| Task 2 (`completion.csv`) | highest dev `token_accuracy` | `f_no_sched_100_t2` |
| Task 3 (`anomaly.csv`) | model-independent (rule validator + LM-style SCORE) | any |

## Hour 5–6 — Official submission (hybrid allowed)

```bash
MODEL_T1=models/sweeps/f_extras_1x_100_mrr.pt.best    # replace with winner
MODEL_T2=models/sweeps/f_no_sched_100_t2.pt.best       # replace with winner

# Task 1 + 3 from MRR model
python -u scripts/predict_submission.py \
  --model "$MODEL_T1" \
  --eval-valid EVAL_DATA/eval_input_valid.csv \
  --eval-anomaly EVAL_DATA/eval_input_anomaly.csv \
  --out-dir extras/results_submission \
  --rule-constrained --beam-width 5 --candidate-pool 5 --device cuda

# Hybrid: overwrite completion with T2 specialist (only if it wins on dev)
python -u scripts/predict_submission.py \
  --model "$MODEL_T2" \
  --eval-valid EVAL_DATA/eval_input_valid.csv \
  --out-dir extras/results_submission_t2 \
  --rule-constrained --beam-width 5 --candidate-pool 5 --device cuda

cp extras/results_submission_t2/completion.csv extras/results_submission/completion.csv
```

Sanity check:
```bash
wc -l extras/results_submission/{nextstep,completion,anomaly}.csv
# Expected: 601 nextstep, 601 completion, 988 anomaly
head -2 extras/results_submission/*.csv
```

Pull back to laptop:
```bash
# From your laptop:
rsync -avz a08trd0x@login01-ext.leonardo.cineca.it:'$SCRATCH/industrial/extras/results_submission/' \
  ./extras/results_submission/
```

**Document the hybrid choice in REPORT.md.** Jury values visible technical decisions.

## Hour 6–9 — Wave 3 (optional)

Only if Wave 2 produced a clear winner that improves on Wave 1. One single
200-epoch job on that exact recipe — no array. Regenerate submission only
if dev metrics beat the Hour-5 pick. **Stop training by Hour 9.**

## Hour 9–12 — Deliverables (non-negotiable)

- `REPORT.md` — baseline → sweep → finalists → hybrid choice with metric table
- `RESULTS_GPU_SUMMARY.md` — Wave 1/2/3 numbers (not the old `transformer_small_extra`)
- `README.md` — predict + eval commands the jury can copy-paste
- 2-min demo: n-gram vs transformer on the same `EVAL_DATA` example
- 10-slide PDF max
- `LICENSE` present, repo public
- Commit submission CSVs + eval JSONs + leaderboard

Submit via Tally by your team's deadline.

---

## Notes & gotchas

- **Anomaly SCORE**: our `predict_submission.py` emits a continuous SCORE
  (10 discrete levels by rule-violation count). Friend's earlier submission
  used binary `1.0/0.0` which gives a degenerate AUC. Always use ours.
- **`note:` field** in YAML is ignored by `sweep_transformer.py` — purely
  for human readability.
- **`f_extras_500_100_mrr`** in Wave 1 requires `data/generated/infineon_500/`.
  Run `make leonardo-generate-extras-500` first (or skip this row).
- **Don't chase +0.0003 MRR** between `m_drop15` and `m_real_extras_1x` —
  use the hybrid Task-1/Task-2 split instead.
- **Concurrency 12** assumes the `s_tra_ncc` reservation has ≥12 GPUs.
  If not, drop to `%8` or `%4` in the `--array` flag.
