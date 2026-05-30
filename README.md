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
2. `docs/TASK_BOARD.md` — what's done, what's next, what's blocked.
3. `docs/PIPELINE.md` — split → train → validate → improve → review loop.
4. `docs/DATA_SPEC.md` — real Infineon data contract.
5. `docs/METRICS_INTERPRETATION.md` — how to explain the dashboard numbers.
6. `docs/TRANSFORMER_MODEL.md` — compact decoder-only Transformer usage.
7. `docs/LOCAL_AUGMENTATION.md` — local generated-data augmentation.
8. `docs/LEONARDO_ONBOARDING.md` — AI:AT HPC access checklist (per person).
9. `docs/LEONARDO_BATTLE_PLAN.md` — runbook for the HPC sweep stages.
10. `docs/DEMO_SPEC.md` — Streamlit demo spec (incl. planned shock page).
11. `docs/implementation-plan-en.md` — full strategic plan (Hungarian original at `docs/implementation-plan-hu.md`).
12. `artifacts/ngram_metrics.json` — latest baseline evidence.

Older planning docs (`AGENTS.md`, `MEMORY.md`, `RULES.md`, `SKILLS.md`) are
reference only. Historical merge/fix notes and superseded specs live in
`docs/archive/` — see `docs/archive/README.md` for the index.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# Base pipeline + CPU PyTorch for the step-5 Transformer smoke path.
make setup-cpu

make smoke              # dev split + n-gram train + no-skip tests + smoke check
make run-demo           # Streamlit demo (n-gram or Transformer, Task 1/2/3 + audit trace)
make run-dashboard      # Streamlit run-history dashboard

# Local Transformer path, GPU-free but requires CPU PyTorch:
make train-transformer-smoke
make train-transformer-small-local

# Step 6 local augmentation, still GPU-free:
make generate-extra-local
make train-ngram-extra
make train-transformer-extra-smoke
make train-transformer-small-extra-local
```

If `train_transformer.py` says PyTorch is missing, run:

```bash
make setup-cpu
```

On Leonardo/GPU, do not use the CPU wheel command; install the CUDA-enabled
PyTorch build according to the HPC environment docs.

`make smoke` produces:

1. a deterministic train/dev split under `data/processed/`,
2. a fitted n-gram baseline at `models/ngram_baseline.pkl` + metrics at
   `artifacts/ngram_metrics.json` + a per-run snapshot at `artifacts/runs/`,
3. 26 passing pytests with **0 skipped** when PyTorch is installed,
4. a smoke report proving the real-data pipeline is wired.

`make run-demo` reads those artifacts and shows Task 1/2/3 headline numbers,
per-family breakdown, a live next-step demo, an explanation of the active model,
and a full audit trace. The sidebar can switch between `models/ngram_baseline.pkl`
and any available `models/transformer_*.pt` checkpoint.

## Current local baseline state

Typical current run after `make dev-split && make train-ngram`. Dev eval is
100 sequences per family truncated at 60% / 80% for Tasks 1+2 (→ 200 items/
family, 600 overall) and 200 full sequences per family for Task 3 (50% valid,
50% rule-injected). Numbers refresh on every `make train-ngram` in
`artifacts/ngram_metrics.json`.

| Task | Metric | Overall |
| --- | --- | ---: |
| 1 | Top-1 | **~0.687** |
| 1 | Top-3 | **~0.928** |
| 1 | Top-5 | **~0.928** |
| 1 | MRR | **~0.807** |
| 2 | Token accuracy | ~0.421 |
| 2 | Normalized edit distance | ~0.224 |
| 3 | F1 (invalid) | **1.000** |
| 3 | Rule attribution | ~0.690 |

Task 2 supports two decoding modes — toggleable via `--rule-constrained` /
`--no-rule-constrained` on both `scripts/predict_submission.py` and
`src/eval/local_eval.py`. **Default is rule-constrained** (top-5 candidates
filtered against the official `validate_sequence` each step). Empirical note:
on the n-gram baseline, plain greedy beats rule-constrained by a hair
(tok_acc 0.428 vs 0.421) because the n-gram learns grammar implicitly from
real data. We expect rule-constrained to win on the Transformer once trained,
since it won't be grammar-perfect.

Task 3 `primary_rule` is the rule fired at the smallest `step_index` in the
sequence (ties broken by rule name) — this is what raised rule attribution
from ~0.667 to ~0.690.

Task 3 perfect F1 is expected for this local dev set: invalid examples are
injected rule violations and the detector calls the official Infineon rule
validator. This is not claiming that the n-gram learned anomaly detection; it is
claiming that the system separates learned route distribution from hard
manufacturing invariants. See `docs/METRICS_INTERPRETATION.md`.

## Subject fit / judge story

The repo now matches the selected Industrial/Infineon subject:

- **Task 1/2:** n-gram suffix baseline plus a compact family-conditioned
  decoder-only Transformer interface.
- **Task 3:** official process-rule validator for deterministic anomaly
  detection and rule attribution.
- **Step 5:** small Transformer implemented locally with CPU smoke and small-run
  targets; Leonardo/GPU only needed for larger training.
- **Step 6:** generated valid-route augmentation is available locally and uses
  the same training/eval split for comparable metrics.

Pitch sentence:

> We compare a memorization-prone n-gram baseline against a trained
> family-conditioned Transformer, then constrain anomaly judgment with explicit
> process rules. This separates learned sequence distribution from hard
> manufacturing invariants.

## Next phases

| Phase | Status | Owner |
| --- | --- | --- |
| Vendor official Infineon data | done | — |
| N-gram baseline + dev split + evaluator + tests | done | — |
| Eval dashboard + run history | done | — |
| Independent evaluator | done, shared model loader | `src/eval/local_eval.py` |
| Small from-scratch Transformer | local CPU/smoke implementation done | `src/ml/transformer_model.py`, `scripts/train_transformer.py` |
| Local generated-data augmentation | done for CPU/local prep | `scripts/generate_extra_sequences.py` |
| Leonardo HPC access | doc'd at `docs/LEONARDO_ONBOARDING.md` | per-person |
| One contrast HF fine-tune (`distilgpt2`) | optional, see `docs/FINETUNE_OPTION_REVIEW.md` | stretch |

## What NOT to do

- Do not build a RAG / LLM chatbot.
- Do not commit any credential, SSH key, certificate, Leonardo link, or
  Discord invite.
- Do not start cluster training before the local pipeline (n-gram + dev
  eval + dashboard + Transformer smoke) is green.

## Stop rule

Once this track is selected at kickoff, stop touching the other two track
repos. If this track is not selected, leave it frozen.
