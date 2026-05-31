# Model checkpoints

## Three folders — don’t confuse them

| Path | What it is |
|---|---|
| **`src/ml/`** | Python **source** — Transformer classes, n-gram, beam search. Always in git. |
| **`models/`** (repo root) | **Trained weights** — `.pt.best` / `.pkl` files written by training. |
| ~~`src/models/`~~ | **Removed** — was an empty git placeholder; never used. Code is in `src/ml/`, weights in `models/`. |

Winning train recipes (YAML rows): [`configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md)

## What to keep on disk

| Path | Role | Keep? |
|---|---|---|
| `sweeps/h_mod_nosched_mrr.pt.best` | Task 1 + Task 3 submission | **yes** (in git via LFS if configured) |
| `sweeps/g_drop15_nosched_t2.pt.best` | Task 2 submission | **yes** |
| `ngram_baseline.pkl` | Level-1 baseline demo | yes (small) |
| `sweeps/*.pt.best` (other runs) | Sweep experiments | **no** — delete after picking leaders |
| `transformer_small.pt` | Legacy smoke checkpoint | **no** — retrain with `make train-transformer-smoke` if needed |

## `src/ml/` vs `models/`

- **`src/ml/`** — Python **source code** (Transformer, n-gram, beam search). **Always keep**; this is the architecture, not weights.
- **`models/`** — **trained weights** on disk. Only the two submission `.pt.best` files need to ship for a full demo without retraining.

Reproduce checkpoints: [`configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md) · [`docs/SUBMISSION.md`](../docs/SUBMISSION.md) · [`HANDOFF.md`](../HANDOFF.md)
