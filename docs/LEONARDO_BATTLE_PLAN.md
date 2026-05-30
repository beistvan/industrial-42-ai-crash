# Leonardo Battle Plan — Industrial / Infineon Transformer

This is the runbook for the hackathon-window HPC sprint. Read top to bottom
once before kickoff; then keep it open as a cheat sheet.

## Stage 0 — Sanity check on the login node (≤ 10 min)

```bash
ssh your_username@login01-ext.leonardo.cineca.it
cd $SCRATCH
git clone <repo-url> industrial && cd industrial

# Bring in latest local data + generated extras (scp from your laptop).
# Required:
#   data/raw/infineon/                  (already in repo)
#   data/processed/splits/              (run scripts/make_dev_split.py once)
#   data/processed/dev_eval/            (same)
#   data/generated/infineon/            (run scripts/generate_extra_sequences.py)
#
# Easiest: just `python3 scripts/make_dev_split.py --force` on Leonardo and
# `python3 scripts/generate_extra_sequences.py --count-per-family 250 --seed 101`.

# Verify torch sees CUDA from the login node (compute-node check happens via Slurm).
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# If False: install CUDA-enabled torch into your pixi env (see Z10 deck p.85).
```

## Stage 1 — Shortlist sweep (~1.5 h/run × 4 GPUs in parallel)

The sweep YAML at `configs/sweeps/leonardo_v1.yaml` declares ~12 configurations
across dropout, lr, label_smoothing, scheduler, and extra-data variations.
Each row trains for **10 epochs** with **AMP + cosine-warmup + cheap Task-1
eval every epoch**. Task 2 eval is skipped here (it's too slow during a sweep).

```bash
# Submit array (auto-sizes to row count; here 12 rows → 0..11):
N=$(python3 scripts/sweep_transformer.py \
    --sweep configs/sweeps/leonardo_v1.yaml --stage shortlist --dry-run \
    | grep -c '^\[')
sbatch --array=0-$((N-1))%4 scripts/leonardo/sweep_array.slurm \
    configs/sweeps/leonardo_v1.yaml shortlist
```

`%4` caps concurrency at 4 — the per-team GPU budget. With 12 rows × 1.5h each
this finishes in ~5h wall-clock.

Per-row outputs land at:
- `models/sweeps/<run_name>.pt` (last) + `<run_name>.pt.best` (best by dev_mrr)
- `artifacts/sweeps/<run_name>.json` (history + per-epoch task-1)

## Stage 2 — Leaderboard + pick top 3

```bash
python3 scripts/summarize_runs.py \
    --metrics-dir artifacts/sweeps \
    --out artifacts/sweeps/LEADERBOARD.md \
    --sort task1_mrr

# Inspect:
head -30 artifacts/sweeps/LEADERBOARD.md
```

The leaderboard CSV (`LEADERBOARD.csv`) is the source of truth. Pick the **top
3 by `task1_mrr`** (or `task1_top1` if Top-1 is the priority).

Edit `configs/sweeps/leonardo_v1.yaml` and fill the `finalists:` block with
the three winning `run_name`s, bumping `epochs: 50` and
`eval_task2_every: 10` so we see Task 2 trajectories.

## Stage 3 — Finalist runs (~5 h/run × 3)

```bash
N=$(python3 scripts/sweep_transformer.py \
    --sweep configs/sweeps/leonardo_v1.yaml --stage finalists --dry-run \
    | grep -c '^\[')
sbatch --array=0-$((N-1))%3 scripts/leonardo/sweep_array.slurm \
    configs/sweeps/leonardo_v1.yaml finalists
```

Each finalist runs 50 epochs with a Task-2 eval every 10 epochs (5 expensive
eval passes per run). Pick the **single best by `task2_token_acc`** or
`task1_mrr`, whichever your judge story emphasizes.

## Stage 4 — Submission

```bash
# scp the best checkpoint back to your laptop, OR run prediction on Leonardo.
# Submission CSVs use the official organizer schema.
python3 scripts/predict_submission.py \
    --model models/sweeps/<winner>.pt.best \
    --eval-valid data/raw/infineon/eval/eval_input_valid.csv \
    --eval-anomaly data/raw/infineon/eval/eval_input_anomaly.csv \
    --out-dir extras/results_submission \
    --rule-constrained --candidate-pool 5
```

Submit `extras/results_submission/{nextstep,completion,anomaly}.csv` via Tally.

## Time budget (rough)

| Stage | Wall-clock | GPU-hours |
|---|---|---|
| Shortlist (12 × 10ep, 4 in parallel) | ~5h | ~18 |
| Leaderboard + decide | 5 min | 0 |
| Finalists (3 × 50ep + Task2) | ~6h | ~18 |
| Submission predict | ~30 min | 0.5 |
| **Total** | **~12h** | **~37** |

Comfortable inside a 24h Slurm budget; leaves headroom for one re-run if a
config crashes.

## When things go wrong

- **`OOM`**: lower `batch_size` (256 → 128 → 64). Medium config + AMP at 256
  fits in 80GB, but very long sequences plus big batches can spike.
- **`NaN` loss**: drop `label_smoothing` to 0, switch to `--scheduler linear`,
  and check `grad_clip` is on (default 1.0).
- **Task-1 eval gives 0.0 every epoch**: `dev_eval/` files missing on the
  compute node. Rerun `make dev-split` on Leonardo or `scp` the files over.
- **Compute node has no internet**: HF caches were the only reason we needed
  it; the from-scratch Transformer does not. Don't bother with the proxy.
- **Run dies after best checkpoint saved**: that's fine — `.pt.best` is on
  disk. Resume not implemented; just re-launch the row.

## Watch from your laptop

```bash
# tail latest slurm log:
ssh login01-ext.leonardo.cineca.it 'cd $SCRATCH/industrial && tail -f logs/slurm-sweep-*.out | head -100'

# leaderboard live:
ssh login01-ext.leonardo.cineca.it 'cd $SCRATCH/industrial && \
    python3 scripts/summarize_runs.py --sort task1_mrr 2>&1 | tail -20'
```
