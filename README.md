# Industrial: Models that learn how processes unfold

> Infineon track (Zero/One Hack) — sequence model that learns semiconductor
> fabrication process trajectories and predicts next-step, sequence
> completion, and rule-violation anomalies.

**START HERE.** This README is the single operating path for this track.

## What the task actually is

Given long sequences of semiconductor fabrication steps (MOSFET / IGBT / IC,
107–151 steps per sequence, ~120-token vocabulary), train a model that can:

1. **Task 1 — Next-step prediction:** given a partial sequence, return the
   top-k most likely next step tokens (Top-1 / Top-3 / Top-5 / MRR).
2. **Task 2 — Sequence completion:** given a 60% or 80% prefix, complete the
   sequence to `SHIP LOT` (exact match, token accuracy, edit distance).
3. **Task 3 — Anomaly detection:** classify a full sequence as valid or
   rule-violating, and attribute the violated rule (binary F1 + rule
   attribution).

A hidden 4th product family is used by the organizers post-submission for
out-of-distribution generalization scoring (Task 4).

## Authoritative files

For execution during the hackathon, read these in order:

1. `README.md` — this file (one-command path + current numbers).
2. `docs/PIPELINE.md` — split → train → validate → improve → review loop.
3. `docs/DATA_SPEC.md` — real Infineon data contract.
4. `docs/implementation-plan-en.md` — full strategic plan (Hungarian
   original at `docs/implementation-plan-hu.md`).
5. `docs/FINETUNE_OPTION_REVIEW.md` — why we are not leading with a
   HuggingFace pretrained fine-tune.
6. `docs/LEONARDO_ONBOARDING.md` — AI:AT HPC access checklist (phase 2).
7. `artifacts/ngram_metrics.json` — latest baseline evidence.

Older planning docs (`AGENTS.md`, `MEMORY.md`, `RULES.md`, `SKILLS.md`,
`docs/GPU_COMPUTE_PLAN.md`, `docs/LLM_ORCHESTRATION_FOR_THIS_REPO.md`) are
reference only.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

make smoke              # dev split + n-gram train + tests + smoke check
make run-demo           # Streamlit demo (Task 1/2/3 + explanation + audit trace)
make run-dashboard      # Streamlit run-history dashboard
```

`make smoke` produces:
1. a deterministic train/dev split under `data/processed/`,
2. a fitted n-gram baseline at `models/ngram_baseline.pkl` + metrics at
   `artifacts/ngram_metrics.json` + a per-run snapshot at `artifacts/runs/`,
3. 20 passing pytests and a smoke check that the real-data pipeline is wired.

`make run-demo` reads those artifacts and shows Task 1/2/3 headline numbers,
per-family breakdown, a live next-step demo, an explanation of the
suffix-backoff match, and a full audit trace (matched suffix, top-k probs,
rule violations with step indexes).

## Current baseline state (n-gram, max_order=8, 50 dev/family, n=300)

| Task | Metric | Overall |
| --- | --- | ---: |
| 1 | Top-1 | **0.683** |
| 1 | Top-5 | **0.990** |
| 1 | MRR | **0.834** |
| 2 | Token accuracy | 0.419 |
| 2 | Normalized edit distance | 0.227 |
| 3 | F1 (invalid) | **1.000** |
| 3 | Rule attribution | 0.633 |

Task 3 perfect F1 is the upper bound from using the official rule validator
directly. Re-run `python scripts/train_ngram.py --max-order N` with
different orders to see the dashboard's improvement trend.

## Next phases

| Phase | Status | Owner |
| --- | --- | --- |
| Vendor official Infineon data | done (PR #1) | — |
| N-gram baseline + dev split + evaluator + tests | done (PR #2) | — |
| Eval dashboard + run history | done (PR #3) | — |
| Independent evaluator | placeholder at `src/eval/local_eval.py` | friend |
| Small from-scratch transformer | placeholder at `src/ml/transformer_model.py` + `scripts/train_transformer.py` | next |
| Leonardo HPC access | doc'd at `docs/LEONARDO_ONBOARDING.md` | per-person |
| One contrast HF fine-tune (`distilgpt2`) | optional, see `docs/FINETUNE_OPTION_REVIEW.md` | stretch |

## What NOT to do

- Do not build a RAG / LLM chatbot.
- Do not commit any credential, SSH key, certificate, Leonardo link, or
  Discord invite.
- Do not start cluster training before the local pipeline (n-gram + dev
  eval + dashboard) is green.

## Stop rule

Once this track is selected at kickoff, stop touching the other two track
repos. If this track is not selected, leave it frozen.
