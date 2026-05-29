# Demo Spec — Industrial: Models that learn how processes unfold

The demo is the Streamlit app launched by:

```bash
make run-demo          # src/app/main.py  — live n-gram demo
make run-dashboard     # src/app/eval_dashboard.py — run history + deltas
```

## What it proves

1. The pipeline ran on real Infineon `training_data/` (no mock data).
2. Headline Task 1/2/3 metrics from `artifacts/ngram_metrics.json` are
   visible and reproducible.
3. Each prediction is explainable: the matched n-gram suffix, the top-k
   probabilities, and the official rule validator's violations with step
   indexes are all in the page.

## Sections in `src/app/main.py`

1. **Title + run caption** — `run_id`, model, `max_order`, fit/eval times.
2. **Baseline evidence** — Task 1 / Task 2 / Task 3 headline metrics + a
   per-family breakdown table.
3. **Live demo controls** — family selector, sequence selector (dev split),
   prefix-length slider.
4. **Task 1 — next-step top-5** — predicted tokens with gold-token highlight
   and gold's rank in top-5.
5. **Task 2 — greedy completion** — predicted vs gold continuation, token
   accuracy, normalized edit distance, exact-match flag, token-level diff.
6. **Task 3 — anomaly detection** — official rule validator's verdict on the
   gold sequence AND on the model's completion.
7. **Explanation** — matched suffix order, matched suffix tokens, top-5
   tokens with raw counts + renormalized probabilities (or a clear
   "backoff fired" notice).
8. **Audit trace** — single JSON block with input metadata, model info,
   Task 1/2/3 records including per-violation rule + step_index +
   description.

## Backup plan

If the live app breaks during the pitch:
- screenshots of the 8 sections above (capture once before pitch),
- the metric table from `artifacts/ngram_metrics.json`,
- the run-history chart from `make run-dashboard`,
- a 60-second screen recording.

## Do not show

- raw notebooks as the main demo,
- huge logs,
- claims without metrics,
- hidden manual intervention.

## Pitch flow, 45 seconds

1. "We trained on real Infineon `training_data/` — three families, ~3,000
   sequences."
2. "Here is the Task 1/2/3 dev evidence on the held-out 300 sequences."
3. "Pick a sequence, set the prefix length — top-5 prediction, completion,
   and rule-validator verdict update live."
4. "The Explanation section shows which n-gram suffix actually matched
   and the renormalized top-k probabilities — no black box."
5. "The Audit trace is the JSON we'd hand to a reviewer."
