# Leonardo GPU Results — Infineon Industrial Track

Structured by [Track 1 Levels](https://docs.zero-one.lumos-consulting.at/tracks/track-1/).

## Level 1 — Baseline (dev holdout)

| Model | Task 1 Top-1 | Task 1 MRR | Task 2 token-acc | Task 3 F1 |
|---|---:|---:|---:|---:|
| N-gram baseline | 0.687 | 0.807 | 0.421 | 1.00 |

## Level 2 — Trained → optimized (dev holdout)

| Stage | Model | Task 1 MRR | Task 1 Top-1 | Task 2 tok-acc | Δ vs baseline |
|---|---|---:|---:|---:|---|
| Baseline | n-gram | 0.807 | 0.687 | 0.421 | — |
| Trained (Wave 1) | `f_drop15_100_mrr` | 0.873 | 0.748 | 0.437 | +8.2% MRR |
| Optimized (Wave 2 T2) | `g_drop15_nosched_t2` | 0.867 | 0.738 | **0.455** | +3.4 pp tok |

## Final hybrid submission (2 checkpoints, 3 CSV files)

| Task | CSV | Checkpoint | Dev headline |
|---|---|---|---|
| Next-step + anomaly | `result/submission/nextstep.csv` | `h_mod_nosched_mrr.pt.best` | MRR **0.8735**, Top-1 **0.75** |
| Completion | `result/submission/completion.csv` | `g_drop15_nosched_t2.pt.best` | tok **0.4545** |

Full sweep table (27 runs): [`artifacts/sweeps/LEADERBOARD_FINAL.md`](artifacts/sweeps/LEADERBOARD_FINAL.md).

## Level 3 — Scaling & architecture (dev holdout)

| Experiment | Run / wave | MRR | tok-acc | Outcome |
|---|---|---:|---:|---|
| +250/family (1×) | `m_real_extras_1x` | **0.872** | 0.439 | modest help |
| +250/family (2×) | `m_real_extras_2x` | 0.869 | 0.436 | hurts vs 1× |
| +500/family | `f_extras_500_100_mrr` | 0.869 | 0.446 | diverges from real dist |
| Modern arch (RoPE/RMSNorm/SwiGLU) | `h_mod_nosched_mrr` (Wave 3) | **0.874** | 0.437 | **new T1 leader** |
| Task-2 prefix training | Wave 4 (`t2_mod_*`) | 0.867 | ≤0.44 | did not beat T2 bar |

**Finding:** Wave 3 modern stack broke the Task-1 plateau; Wave 4/5 did not improve submission picks.
