# Implementation Plan — Alignment Review

Comparing `docs/implementation-plan-hu.md` (the friend's plan, written in
Hungarian; English mirror in `docs/implementation-plan-en.md`) against the
audit done in this session and against the existing track docs
(`docs/DATA_SPEC.md`, `docs/EVALUATION_PLAN.md`, `docs/MODEL_PLAN.md`,
`CLAUDE.md`).

**Verdict: strongly aligned.** Same diagnosis, same architecture, same
priority order. Merge the plan as-is and adopt it as the working roadmap. A
few small deltas are worth tracking — listed at the bottom.

## Where the plan and the audit agree

| Topic | Friend's plan | Our audit / existing docs |
| --- | --- | --- |
| Drop the pre-hack mock RF baseline | "shelve the mock RandomForest baseline" | Audit said the same: `src/ml/baseline.py` solves the wrong problem |
| Vendor official Infineon data into the repo | step 1 of the plan | Already done in PR #1 (merged) — `data/raw/infineon/` |
| Loader returning per-family sequences | `src/data/infineon_loader.py` | `docs/DATA_SPEC.md` defines the same contract: `dict[seq_id, list[step_token]]` |
| First model = n-gram with suffix backoff | step 4 | Audit recommended exactly this as the naive baseline |
| Improved model = small decoder-only transformer (d_model 128–256, 4–6 layers) | step 5 | `docs/MODEL_PLAN.md` calls for "transformer encoder/decoder" + scaling experiments; the small/medium sizing is right for the ~120-token vocab |
| Task 3 anomaly via the official rule validator + neural likelihood | step 2 + step 5 | Audit flagged anomaly as the highest-leverage task precisely because the 10 rules are documented |
| Build a local dev split because eval inputs arrive later | step 3 | Matches `docs/EVALUATION_PLAN.md` (no train-test leakage, saved split files) |
| Generate extra sequences via the official generator for scaling | step 6 | Matches stretch goal in the track briefing |
| Streamlit demo = side-by-side baseline vs transformer + rule attribution | "Demo" section | Matches `CLAUDE.md` UI target ("timeline, predicted next state, anomaly score, bottleneck root-cause candidates") |
| Leonardo only after local pipeline works | "Using Leonardo: only in phase 2" | Matches `CLAUDE.md` rule "working, testable MVP over breadth" |
| No secrets in repo / Discord | explicit "what NOT to do" | Matches global security rules |

## Small deltas to track (non-blocking)

1. **Path conventions.** The plan's example layout uses
   `extras/results/{nextstep,completion,anomaly}.csv` for submission files.
   Our repo already has `artifacts/` (gitignored) and `runs/`. Decision: use
   `artifacts/submission/` for submission CSVs to match existing convention,
   and update the plan's path in our follow-up PRs rather than re-spelunking
   `extras/`.

2. **Data already in place.** The plan's step 1 (`cp -r .../training_data/*
   data/raw/infineon/`) is already complete via PR #1, with the additional
   `reference/` vs `training_data/` subfolder split documented in
   `data/raw/infineon/SOURCE.md`. The friend's `cp` recipe puts everything
   flat under `data/raw/infineon/`; ours is slightly more organized. No
   action needed — the loader will handle either layout.

3. **`docs/official/` mirror.** The plan suggests copying
   `Track_industrial_en.md` and `generation_rules.md` into `docs/official/`.
   Currently `subject/Track_industrial_en.md` exists (untracked) and
   `generation_rules.md` lives at
   `data/raw/infineon/training_data/generation_rules.md`. We may want to
   either track `subject/` or set up a `docs/official/` mirror in a small
   follow-up PR to make the briefing readable from the docs index.

4. **Vocabulary size.** Plan says "vocab ≈ 150." Per the upstream README the
   union across all 3 families is ~120 tokens. Difference is the BOS / EOS /
   FAMILY special tokens — both numbers are correct depending on whether you
   count specials. Loader spec in `docs/DATA_SPEC.md` already reserves
   `<bos>` and `<eos>`; we'll add `<FAMILY_MOSFET>`, `<FAMILY_IGBT>`,
   `<FAMILY_IC>` per the plan.

5. **Team split.** The plan assigns Person A = ML/backend, Person B =
   product/demo/submission. Worth confirming with your teammate before the
   hackathon kicks off so neither of you trains a model with no submission
   writer in place.

## Next concrete step (unchanged from the prior audit)

Implement `src/data/infineon_loader.py` + `Vocabulary` + one pytest, reading
`data/raw/infineon/training_data/*_variants.csv` into
`dict[sequence_id, list[step_token]]`. This is what unblocks both the
n-gram baseline (plan step 4) and the transformer (plan step 5).
