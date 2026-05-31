# Leonardo GPU Results — Infineon Industrial Track

Structured by [Track 1 Levels](https://docs.zero-one.lumos-consulting.at/tracks/track-1/).

## Level 1 — Baseline (dev holdout)

| Model | Task 1 Top-1 | Task 1 MRR | Task 2 token-acc | Task 3 F1 |
|---|---:|---:|---:|---:|
| N-gram baseline | 0.687 | 0.807 | 0.421 | 1.00 |

## Level 2 — Trained → optimized (dev holdout)

| Stage | Model | Task 1 MRR | Task 2 tok-acc | Δ vs baseline |
|---|---|---:|---:|---|
| Baseline | n-gram | 0.807 | 0.421 | — |
| Trained (Wave 1) | `f_drop15_100_mrr` | **0.873** | 0.437 | +8.2% MRR |
| Optimized (Wave 2 T2) | `g_drop15_nosched_t2` | 0.867 | **0.455** | +3.4 pp tok |

## Submission artifacts (hybrid — 2 checkpoints, 3 CSV files)

| Task | CSV | Checkpoint |
|---|---|---|
| Next-step + anomaly | `result/submission/nextstep.csv` | `h_mod_nosched_mrr.pt.best` |
| Completion | `result/submission/completion.csv` | `g_drop15_nosched_t2.pt.best` |

Full sweep table: [`artifacts/sweeps/LEADERBOARD_FINAL.md`](artifacts/sweeps/LEADERBOARD_FINAL.md).

## Level 3 — Scaling signals (dev holdout)

| Data recipe | Run | MRR | tok-acc | train_seconds |
|---|---|---:|---:|---:|
| Real only | `m_real_only` | 0.866 | 0.441 | 162 |
| +250/family (1×) | `m_real_extras_1x` | **0.872** | 0.439 | 183 |
| +250/family (2×) | `m_real_extras_2x` | 0.869 | 0.436 | 184 |
| +500/family | `f_extras_500_100_mrr` | 0.869 | 0.446 | 2067 |

**Finding:** 1× synthetic extras helps modestly; more volume diverges from real distribution.

**In flight:** Wave 3 (modern architecture), Wave 4 (Task-2 prefix training).
