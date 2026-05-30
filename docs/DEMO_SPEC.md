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
4. "The Explanation section shows which suffix actually matched (n-gram) or
   how the decoder ranked candidates (Transformer) — no black box."
5. "The Audit trace is the JSON we'd hand to a reviewer."

## Planned — "Process Resilience Lab" page

A second tab in `src/app/main.py` that demonstrates the model's understanding,
not just its metrics. **Story for the judge:** when one fab tool goes offline
or a regulation blocks a process step, today fab planners replan by hand. Our
model proposes valid recovery routes in <100ms.

### Mechanic

1. Pick a family + a base prefix (reuse existing controls).
2. **Inject a shock**: multi-select tokens to disable (e.g., specific litho
   or implant steps), with a short menu of presets:
   - "EUV scanner offline" → block all EUV-litho tokens
   - "Cleanroom contamination" → block all CLEAN tokens
   - "Supply-chain shock on Cu" → block all Cu-METAL tokens
   - "Regulatory change" → block any user-picked subset
3. Run two completions side by side:
   - **Baseline** — model's normal greedy completion.
   - **Shocked** — same model but blocked tokens are masked out before argmax
     at every step (post-processing on `predict_topk`, no retraining).
4. For each completion, display:
   - Predicted route length and the route as a token chip strip.
   - Official rule-validator verdict (valid / first violation + step index).
   - **Illustrative cost delta**: `len(route) * avg_cost_per_step` with a
     hand-tuned `{step_token: cost}` dict for ~15 expensive steps (litho,
     implant, CMP). Frame as "plausible numbers, real reasoning."
   - **Yield proxy**: `1 - cumulative_violation_prob` style number.
5. Bottom of page — a running counter: "routes rerouted: N / sequences
   recovered: M" that increments as the demo user clicks.

### Honest disclaimers (in the demo)

- "Costs and yields are illustrative — we don't have Infineon's real fab
  economics."
- "The model is not trained to be resilient; we're masking outputs at
  inference. It only succeeds when the grammar admits an alternative path."
- "When no valid route exists post-shock, we show that too — the validator
  catches it."

### Implementation surface

- `src/app/shock_demo.py` — new Streamlit page.
- `src/ml/inference_helpers.py` — small helper: `predict_topk_masked(model,
  family, prefix, blocked_set, k)` that wraps the existing API.
- `src/data/cost_table.py` — hand-tuned cost dict.
- Reuse `cached_load_model`, `classify_sequence`, `FAMILIES`.

### Effort budget

~3 hours: 1h shock-injection wrapper + cost table, 1.5h Streamlit page,
0.5h preset menu + counters + screenshots for the backup plan.

### What NOT to add

- No tycoon-game time progression (out of scope).
- No agent-vs-agent simulation (out of scope).
- No animated graphs — static side-by-side wins.
- No chatbot interface (forbidden by `README.md::What NOT to do`).
