# Leonardo GPU Results — Infineon Industrial Track

## Baseline vs submission (dev holdout)

| Model | Task 1 Top-1 | Task 1 MRR | Task 2 token-acc | Task 3 F1 (invalid) |
|---|---:|---:|---:|---:|
| N-gram baseline | 0.687 | 0.807 | 0.421 | 1.00 |
| Submission T1 (`f_drop15_100_mrr`) | **0.748** | **0.873** | 0.437 | 1.00 |
| Submission T2 (`f_extras_1x_100_t2`) | 0.743 | 0.870 | **0.451** | 1.00 |

Full sweep table: [`artifacts/sweeps/LEADERBOARD_FINAL.md`](artifacts/sweeps/LEADERBOARD_FINAL.md).

## Submission artifacts

Hybrid checkpoint strategy (2 models, 3 CSV files):

| Task | CSV | Checkpoint |
|---|---|---|
| Next-step | `extras/results_submission/nextstep.csv` | `models/sweeps/f_drop15_100_mrr.pt.best` |
| Completion | `extras/results_submission/completion.csv` | `models/sweeps/f_extras_1x_100_t2.pt.best` |
| Anomaly | `extras/results_submission/anomaly.csv` | rule validator (same run as Task 1) |

Checkpoints are gitignored (~140 MB each). Reproduce with the commands in
[`README.md`](README.md) or [`HANDOFF.md`](HANDOFF.md).

## Interpretation

The Wave-1 GPU sweep on Leonardo pushed Task-1 MRR from 0.807 (n-gram) to
0.873 (+8.2% relative) and Task-2 token accuracy from 0.421 to 0.451 (+3.0 pp).
Task 3 remains rule-based through the official Infineon validator.
