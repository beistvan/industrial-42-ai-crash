# Results

Judge-format outputs and local eval scratch space.

| Path | Purpose | In git? |
|---|---|---|
| `submission/` | **Final hybrid submission** — `nextstep.csv`, `completion.csv`, `anomaly.csv` | yes |
| `.staging_t2/` | Temporary Task-2 predict during `make regenerate-submission` | no (gitignored) |
| `dev/` | Local dev-split smoke predictions | no |
| `reproduce/` | Optional manual reproduce outputs (see `REPORT.md`) | no |

**Current hybrid:**

- Task 1 + 3: `h_mod_nosched_mrr` → `nextstep.csv`, `anomaly.csv`
- Task 2: `g_drop15_nosched_t2` → `completion.csv`

Regenerate: `make regenerate-submission`
