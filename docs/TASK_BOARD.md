# Task Board — Industrial: Models that learn how processes unfold

## Phase 0 — before case reveal
| Task | Owner | Tool | Acceptance |
|---|---|---|---|
| Verify repo runs | A | Codex/Copilot | `make test` and `make run-demo` work |
| Draft demo scenario | B | Claude/DeepSeek | One scenario judges understand in 30 seconds |
| Prepare fallback dataset | A | Claude/Codex | Loader works on sample data |
| Prepare pitch skeleton | B | Claude | 6-slide pitch outline exists |

## Phase 1 — Friday night
| Task | Owner | Tool | Acceptance |
|---|---|---|---|
| Ingest event data | A | Codex | Data schema validated |
| Build first baseline | A | Copilot/Codex | Baseline metric printed |
| UI scenario screen | B | Copilot | Demo screen loads sample |
| Track-specific story | B | Claude | Problem/solution/metric locked |

## Phase 2 — Saturday
| Task | Owner | Tool | Acceptance |
|---|---|---|---|
| Improved model/layer | A | Codex | Beats baseline or is justified |
| Evaluation report | A | Claude/Codex | Metrics + chart saved |
| Integrated UI | B | Copilot/Codex | Model output appears in UI |
| Mentor feedback loop | Both | Human | 3 feedback items implemented/cut |

## Phase 3 — Sunday
| Task | Owner | Tool | Acceptance |
|---|---|---|---|
| Freeze code | A | Codex | Clean run from README |
| Record fallback demo | B | Human | 90-sec video/screen capture |
| Final pitch | B | Claude/DeepSeek | 3-minute pitch rehearsed |
| Judge Q&A | Both | DeepSeek | 10 hard questions answered |

---

# Task board — Industrial: Models that learn how processes unfold

Use this file as the only task board.

## Frozen / done

- Baseline scaffold exists.
- Mock-data generator exists.
- Baseline metric artifact exists.
- Tests pass on mock data.
- Streamlit demo has a working scenario/recommendation shell.

## Do now, before the reveal

- [ ] Commit the current passing state.
- [ ] Verify `make smoke`, `make test`, and `make run-demo`.
- [ ] Do not tune synthetic metrics.
- [ ] Do not rewrite architecture.
- [ ] Do not follow old orchestration paths.

## Do first tomorrow, after real data arrives

- [ ] Create `docs/REAL_BRIEF.md` with the official task, success metric, constraints, and deliverable format.
- [ ] Copy real data into `data/raw/` without modifying the original file.
- [ ] Inspect schema and examples.
- [ ] Identify target/outcome and evaluation metric.
- [ ] Map real data to baseline features.
- [ ] Run the baseline and update `artifacts/metrics.json`.
- [ ] Update demo labels so they match the official case.

## Improve only after baseline works

Possible improvements:

- better feature engineering;
- calibrated confidence/uncertainty;
- model comparison table;
- clearer explanation trace;
- real-data error analysis;
- judge-friendly demo flow.

## Do not do

- Do not improve synthetic scores for their own sake.
- Do not touch the other two track repos after selection.
- Do not spend time reconciling old instruction files.
- Do not build a complex model before real-data baseline is running.
