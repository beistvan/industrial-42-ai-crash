# Tomorrow on Leonardo — Run, Compare, Decide

Use this as your seat-of-the-pants checklist while the HPC clock runs.
Cross-references: `docs/LEONARDO_BATTLE_PLAN.md` for the *what to type*,
`docs/TASK_BOARD.md` for the *what's left*.

## 0. Before you SSH (10 min)

- [ ] You have the latest `main` checked out on your laptop.
- [ ] `data/processed/dev_eval/*.csv` are reproducible: run
      `make dev-split` and check md5 sums match the values noted at the
      bottom of this file. **If they don't match, do not push to HPC.**
- [ ] `make smoke` is green locally (33 tests, ~30s).
- [ ] You have a personal Leonardo account + reservation `s_tra_ncc` is
      live for your team.

## 1. Stage 0 — Sanity on the login node (~5 min)

```bash
ssh your_user@login01-ext.leonardo.cineca.it
cd $SCRATCH
git clone <repo-url> industrial && cd industrial
python3 scripts/make_dev_split.py --force
python3 scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force
python3 -m pytest -q                    # 33 passed locally; expect same here
python3 scripts/check_environment.py --require-torch
```

Expected: torch sees CUDA on the login node OR is invoked through a
Singularity container at runtime — confirm one of them works before
submitting any Slurm job.

## 2. Stage 1 — Submit the shortlist (~5h)

```bash
bash scripts/leonardo/run_battle.sh configs/sweeps/leonardo_v1.yaml shortlist
```

This submits:
- An **array job** of 12 configs (concurrency capped at 4 GPUs).
- A **dep-chained leaderboard job** that runs `summarize_runs.py` the moment
  the array finishes.

Watch:
```bash
squeue --me
tail -f logs/slurm-sweep-*_*.out          # any single array task
```

While it runs, **work on the demo** (Streamlit shock-injection page, see
`docs/DEMO_SPEC.md`) and the **pitch script**. Don't sit and watch the bar.

## 3. Stage 2 — Read the leaderboard

After Stage 1 finishes, the dep-chained job writes:
```
artifacts/sweeps/LEADERBOARD.md
artifacts/sweeps/LEADERBOARD.csv
```

Open `LEADERBOARD.md`. The columns to look at, in order:

| Column | What it tells you |
|---|---|
| `task1_top1` | strict next-step accuracy. >0.75 = competitive. >0.85 = winning. |
| `task1_mrr` | softer signal. 0.80–0.85 expected for a working Transformer. |
| `task1_top5` | sanity check — if this is below 0.95 something is wrong. |
| `best_epoch` | did training peak early (overfit) or late (still learning)? |
| `train_seconds` | sanity — should be roughly equal across runs |

### Decision matrix

**Case A — Clear winner emerges (top run beats next by ≥ 3pts top-1):**
- Promote that one config to `finalists:` block with `epochs: 50` and
  `eval_task2_every: 10`. Skip the other two finalist slots — save the
  GPU hours.
- Submit: `bash scripts/leonardo/run_battle.sh configs/sweeps/leonardo_v1.yaml finalists`

**Case B — Top 3 are within 2pts of each other:**
- Promote all three. The Task-2 evaluation (every 10 epochs) will
  differentiate them better than Task-1 does.

**Case C — `m_real_only` beats `m_real_extras_*`:**
- The synthetic generator is hurting you. Either re-seed with
  `python3 scripts/generate_extra_sequences.py --count-per-family 100 --seed 202`
  (less synthetic) or skip extras entirely. Re-run a single config:
  `sbatch scripts/leonardo/train_one.slurm configs/transformer_medium.yaml m_real_only_50ep`
  with `epochs: 50` overridden via env or a separate config file.

**Case D — Everything sits below 0.50 top-1:**
- Something broke. Check Slurm logs for `OOM`, `NaN`, or vocabulary
  mismatches. Most likely culprits: dev-split not generated on HPC, or
  the AMP path silently misbehaving (we fixed both — but check).

**Case E — Top run has weird per-family numbers** (mosfet 0.9, ic 0.4):
- Per-family imbalance in training. Inspect the `extra_counts` field in
  the metrics JSON. The IC family has the shortest sequences (107 steps),
  so it can underfit faster. Add `--limit-extra-sequences 500` to oversample.

## 4. Stage 3 — Finalists (~6h)

The finalist runs do real Task-2 evaluation. After they finish, regenerate
the leaderboard:

```bash
python3 scripts/summarize_runs.py --sort task2_token_acc
```

Now `task2_token_acc` and `task2_ned` are populated.

Pick the single best by **(task1_mrr + task2_token_acc) / 2** — judges weight
both. If the two metrics disagree (best-on-Task1 ≠ best-on-Task2), pick
**best on Task 2** because Task 1 is forgivingly graded and our n-gram is
already strong there.

## 5. Stage 4 — Submission (~30 min)

The organizers drop `eval_input_valid.csv` and `eval_input_anomaly.csv` at
hackathon start. Drop them at the path documented in `docs/DATA_SPEC.md`:

```bash
mkdir -p data/raw/infineon/eval
# scp organizer files into the path above
python3 scripts/predict_submission.py \
    --model models/sweeps/<winner>.pt.best \
    --eval-valid data/raw/infineon/eval/eval_input_valid.csv \
    --eval-anomaly data/raw/infineon/eval/eval_input_anomaly.csv \
    --out-dir extras/results_submission \
    --rule-constrained --candidate-pool 5
```

Sanity check the submission CSVs:
```bash
wc -l extras/results_submission/*.csv
# expect:
#   nextstep.csv  ≈ 600 sequences × 5 ranks = ~3000 rows + header
#   completion.csv ≈ 600 × ~125 step avg   = ~75000 rows + header
#   anomaly.csv   = 987 + header
head -3 extras/results_submission/anomaly.csv
# expect SEQUENCE_ID,IS_VALID,PRIMARY_RULE,ALL_RULES with sensible values
```

**Also generate the n-gram fallback submission** as a safety net:
```bash
python3 scripts/predict_submission.py \
    --model models/ngram_baseline.pkl \
    --eval-valid data/raw/infineon/eval/eval_input_valid.csv \
    --eval-anomaly data/raw/infineon/eval/eval_input_anomaly.csv \
    --out-dir extras/results_submission_ngram \
    --rule-constrained --candidate-pool 5
```

If the Transformer submission looks dubious (anomaly column all 1s, or
completions truncate at <50 steps), **submit the n-gram instead**. Better a
strong n-gram score than a half-broken Transformer score.

## 6. Things to compare side-by-side after each stage

After Stage 1:
- `task1_top1` distribution across 12 runs — what is the **variance**? If
  variance > 5pts, the model is unstable; pick the highest-stability config
  for finalists.
- `train_seconds` — should be flat across runs. If one config is 2× the
  others, something is off (AMP not actually engaged, batch size dropped to
  fit memory, etc.).
- Per-family `task1_top1` — write down the worst family per run. The finalist
  should not have any family below 0.60.

After Stage 3:
- For each finalist, the **history** array in its metrics JSON shows the
  per-epoch trajectory. Plot it (1-liner in Streamlit eval dashboard) — a
  smooth-converging curve is what you want; a noisy/oscillating curve hints
  at LR too high.
- **Compare to the n-gram baseline** for every metric. If the Transformer
  loses on Task 1 top-1 but wins on Task 2 token_acc, that's a real signal —
  the Transformer learned grammar but the n-gram memorized local context
  better. **Ensemble candidate**: take the n-gram's top-3 for Task 1, the
  Transformer's completion for Task 2.

After submission:
- Diff `nextstep.csv` rows between Transformer and n-gram submissions —
  pick a few `SEQUENCE_ID`s where they disagree and sanity-check by hand
  against `data/raw/infineon/training_data/generation_rules.md`.

## 7. Failure modes and recoveries

| Symptom | Likely cause | Recovery |
|---|---|---|
| All Stage-1 runs crash with `KeyError: 'training'` | YAML in `configs/transformer_medium.yaml` lost a section | Restore via `git checkout HEAD~1 -- configs/transformer_medium.yaml` |
| OOM at batch_size=256 | Long sequences spike memory | Add `--batch-size 128` to one row and re-run only that array task |
| Best checkpoint is epoch 1 | Probably LR too high or scheduler broken | Re-run with `--scheduler linear --lr 1e-4` |
| Task-1 eval crashes with KeyError on FAMILY | Dev-eval CSV missing or wrong path | `make dev-split` on HPC, verify `data/processed/dev_eval/` |
| Submission script returns 0 rows | Eval-input CSV has BOM or wrong column names | Strip BOM with `sed -i '1s/^\xef\xbb\xbf//'`, retry |
| Time budget blown | Stage 1 took >8h | Skip Stage 3, submit best `.pt.best` from Stage 1 directly |

## 8. Reproducibility anchors

These should NOT change between local and HPC. If they do, raise an alarm.

```
data/processed/dev_eval/eval_input_valid_dev.csv      md5 = 8e36875d1f3f8285d4fe443441ec4c58
data/processed/dev_eval/eval_input_valid_dev_gold.csv md5 = e9f2a135bb56fcbf588152e39f2fbed2
data/processed/dev_eval/eval_input_anomaly_dev.csv    md5 = b0b399374ad373fe6dcba1c01897bdbf
```

Current n-gram baseline (your reference floor):
```
task1.overall.top1 ≈ 0.687
task1.overall.top5 ≈ 0.928
task1.overall.mrr  ≈ 0.807
task2.overall.token_accuracy ≈ 0.421 (rule-constrained) or 0.428 (greedy)
task2.overall.normalized_edit_distance ≈ 0.224
task3.overall.f1_invalid = 1.000 (ceiling — validator-as-classifier)
task3.overall.rule_attribution_accuracy ≈ 0.690
```

**The Transformer must beat the n-gram on Task 1 MRR to be worth submitting.**
Anything weaker than that means we ship the n-gram.
