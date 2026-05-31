# REPORT — Industrial AI (Infineon)

**Team**: Andrija Jovanovic, Istvan Beregszaszi, Thánh Trung Nguyen
**Track**: Industrial AI (Infineon)
**Repo**: https://github.com/beistvan/industrial-42-ai-crash (branch: `wave1-submission`)
**License**: MIT

---

## TL;DR

We learn the *logic* of semiconductor process sequences with a compact
decoder-only Transformer (~4M params) trained from scratch on the Infineon
domain vocabulary. Submission uses a **hybrid of two trained models** — best by
MRR for next-step prediction, best by token-accuracy for sequence completion —
plus the official rule validator with a continuous SCORE for anomaly detection.
On our held-out dev split this lands at **Top-1 0.748 / MRR 0.873** for Task 1,
**token-accuracy 0.455 / NED 0.223** for Task 2, and **F1 1.00** for Task 3 —
all materially above the n-gram baseline (Top-1 0.687 / MRR 0.807 / tok 0.421).

Structured by [Track 1 Levels](https://docs.zero-one.lumos-consulting.at/tracks/track-1/):
**Level 1** data + n-gram baseline → **Level 2** Transformer train/tune (Waves 1–2)
→ **Level 3** scaling study + modern architecture (Waves 3–4, in flight).

---

## Track 1 levels (Zero One brief)

| Level | Requirement | What we did | Headline (dev) |
|---|---|---|---|
| **Level 1** | Data + baseline | 3k official sequences + 750 synthetic extras; n-gram suffix-backoff (order 12) | MRR **0.807**, tok **0.421** |
| **Level 2** | Train → tune → visible benchmark | Vanilla Transformer Waves 1–2; 23-run dev leaderboard; Streamlit demo + sweep dashboard | MRR **0.873**, tok **0.455** |
| **Level 3** | Scaling / architecture | Extras scaling (1× helps, 2×/500 hurts); Wave 3 RoPE/RMSNorm/SwiGLU; Wave 4 Task-2 prefix training | in flight |

**Baseline → trained → optimized:**

| Stage | Model | Task 1 MRR | Task 2 tok-acc | Δ vs baseline |
|---|---|---:|---:|---|
| Level 1 baseline | n-gram | 0.807 | 0.421 | — |
| Level 2 trained | `f_drop15_100_mrr` (Wave 1) | **0.873** | 0.437 | +6.6 pp MRR |
| Level 2 optimized | `g_drop15_nosched_t2` (Wave 2 T2) | 0.867 | **0.455** | +3.4 pp tok |

---

## Problem

Semiconductor manufacturing is long, ordered sequences of fabrication steps
(115–150 steps per wafer lot, 3 product families — MOSFET, IGBT, IC, ~140 unique
step tokens per family). The challenge: can a model learn the **logic** of these
sequences — which steps must follow which, where a deviation matters — not just
memorize them?

This matters because tiny process deviations cause yield loss and scrap. A model
that captures process logic could flag anomalies early, guide route planning,
and generalize to product variants without retraining.

The official benchmark has three tasks:
1. **Next-step prediction** — given a prefix, rank the most likely next step (Top-1/3/5, MRR).
2. **Sequence completion** — given a prefix, generate the full remaining sequence (token-accuracy, normalized edit distance, exact match).
3. **Anomaly detection** — classify full sequences as valid or invalid against a 10-rule grammar; identify the violated rule (binary F1, AUC, rule-attribution accuracy).

---

## Approach

Five key technical decisions:

1. **Decoder-only Transformer from scratch — no HuggingFace pretrained model.**
   Our 140-token domain vocabulary (`RECEIVE WAFER LOT`, `GATE OXIDE GROWTH`, …)
   has no overlap with English text. A pretrained LM's BPE tokenizer would shred
   each step into meaningless sub-tokens; its embedding priors would actively
   hurt. We train ~4M params (6 layers, 8 heads, d_model=256) on the domain
   vocabulary directly. Rationale: `docs/ADRs/0001-no-hf-pretrained.md`.

2. **Rule-constrained beam search for completion.**
   At each decode step, the model's top-k candidates are filtered through the
   official `validate_sequence` rule grammar. Invalid candidates are dropped
   before the beam advances. Eliminates a class of "physically impossible"
   completions while leaving the model free to learn statistics during
   training. ADR: `docs/ADRs/0002-rule-constrained-decoding.md`.

3. **Hybrid Task-1 / Task-2 submission.**
   Wave-1 finalists showed the configuration that maximizes `dev_mrr` is *not*
   the same as the one that maximizes `dev_token_acc`. Rather than pick one
   model and accept the trade-off, we train **two specialists** and use each
   for the task it wins on. `predict_submission.py` produces judge-format CSVs;
   the orchestrator copies the Task-2 specialist's `completion.csv` over the
   Task-1 model's.

4. **N-gram suffix-backoff baseline (order 12).**
   Pure frequency model on the same training data. Serves as a transparent
   sanity check — if the Transformer couldn't beat suffix-backoff with 12k+
   training samples, the architecture would be over-engineered for the problem.
   It beats it by ~6.6pp Top-1 and 6.6pp MRR.

5. **Continuous anomaly SCORE.**
   The judge's anomaly task asks for `IS_VALID, SCORE, PREDICTED_RULE`. SCORE
   feeds AUC. A binary 1.0/0.0 score works only if classification is perfect —
   ours uses `1 − n_violations/10` to remain monotone and informative under
   noisier validators.

### Hyperparameter sweep on Leonardo (CINECA EuroHPC A100)

| Wave | Rows | Epochs | Notes |
|---|---|---|---|
| Shortlist | 12 | 50 | One-knob-at-a-time over lr, dropout, label_smoothing, scheduler, warmup, extras |
| Wave 1 finalists | 6 | 100–150 | Extended winners, added Task-2 specialists, re-enabled AMP, eval_task2_every=5 |
| Wave 2 fine grid | 5 | 100–150 | Dropout neighbours, no-scheduler × Task-2 selection — **best T2: `g_drop15_nosched_t2`** |
| Wave 3 modern arch | 4 | 100 | RoPE + RMSNorm + SwiGLU + pre-norm (in flight) |
| Wave 4 Task-2 prefix | 4 | 100 | 60–80% prefix training + beam eval (in flight) |

Submitted hybrid uses Wave 1 Task-1 winner + Wave 2 Task-2 winner.

---

## How to run it

### Quickstart (CPU, ~5 min)
```bash
git clone -b wave1-submission https://github.com/beistvan/industrial-42-ai-crash.git
cd industrial-42-ai-crash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
make dev-split && make train-ngram && pytest -q
```

### Full reproduction (GPU, ~90 min on one A100)
```bash
# Synthetic augmentation (deterministic, seed=101)
python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

# Train both finalists from the sweep YAML
python scripts/sweep_transformer.py --sweep configs/sweeps/leonardo_final.yaml \
    --stage finalists --row 2     # f_drop15_100_mrr   (Task-1 winner)
python scripts/sweep_transformer.py --sweep configs/sweeps/leonardo_fine.yaml \
    --stage finalists --row 4     # g_drop15_nosched_t2 (Task-2 winner, Wave 2)

# Hybrid submission
python scripts/predict_submission.py \
    --model models/sweeps/f_drop15_100_mrr.pt.best \
    --eval-valid EVAL_DATA/eval_input_valid.csv \
    --eval-anomaly EVAL_DATA/eval_input_anomaly.csv \
    --out-dir result/reproduce \
    --rule-constrained --beam-width 5 --candidate-pool 5 --device cuda

python scripts/predict_submission.py \
    --model models/sweeps/g_drop15_nosched_t2.pt.best \
    --eval-valid EVAL_DATA/eval_input_valid.csv \
    --out-dir result/reproduce_t2 \
    --rule-constrained --beam-width 5 --candidate-pool 5 --device cuda

cp result/reproduce_t2/completion.csv result/reproduce/completion.csv

# Score against ground truth (using the judge's official script)
python EVAL_DATA/eval_metrics.py --task next-step \
    --ground-truth <YOUR_GT.csv> --predictions result/reproduce/nextstep.csv
```

The hackathon submission CSVs are checked in under `result/submission/`.

**What you need**: Python 3.10+, ~2 GB disk, a CUDA-12.1 PyTorch build for the
GPU path. **No API keys. No external services.** Leonardo access is **not**
required to reproduce — the sweep YAMLs work with any Slurm cluster or as
plain CLI rows via `--row N`.

---

## Results

### Final hybrid submission (dev split, n=300)

| File | Model | Selection criterion | Headline metric |
|---|---|---|---|
| `nextstep.csv` | `f_drop15_100_mrr.pt.best` | save-best-by `dev_mrr` | **MRR 0.8731, Top-1 0.7483** |
| `completion.csv` | `g_drop15_nosched_t2.pt.best` | save-best-by `dev_token_acc` | **dev_token_acc 0.4545** |
| `anomaly.csv` | rule validator + SCORE heuristic | — | **F1 1.00, AUC 1.00 (rules separate cleanly)** |

### Baseline vs transformer (all three tasks)

| Model | Task 1 MRR | Task 2 tok-acc | Task 3 F1 |
|---|---:|---:|---:|
| N-gram suffix-backoff (order 12) | 0.807 | 0.421 | 1.00 |
| Transformer Wave-1 (T1 specialist) | **0.873** | 0.437 | 1.00 |
| **Hybrid Wave 1+2** | **0.873** | **0.455** | 1.00 |

### Baseline vs transformer (Task 1 detail)

| Model | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|
| N-gram suffix-backoff (order 12) | 0.687 | 0.928 | 0.928 | 0.807 |
| Transformer shortlist best | 0.747 | — | 1.000 | 0.872 |
| **Transformer Wave-1 hybrid** | **0.748** | — | **1.000** | **0.873** |

Per-family n-gram breakdown:

| Family | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|
| MOSFET | 0.690 | 0.915 | 0.803 |
| IGBT | 0.735 | 0.970 | 0.852 |
| IC | 0.635 | 0.900 | 0.768 |

### Sweep evidence

Full leaderboard with **23 ranked runs** (12 shortlist + 6 wave-1 finalists + 5 wave-2)
is at `artifacts/sweeps/LEADERBOARD_FINAL.md`. Per-run training history is in
`artifacts/sweeps/{f,m,g}_*.json` — auditable.

Top 5 by `dev_mrr`:

| Rank | Run | dev_mrr | Top-1 | best_epoch | Note |
|---:|---|---:|---:|---:|---|
| 1 | `f_drop15_100_mrr` | **0.8731** | 0.7483 | 84/100 | Wave 1 — dropout 0.15, AMP on |
| 2 | `g_drop15_t2` | 0.8722 | 0.7483 | 90/108 | Wave 2 — peak MRR, T2 selection |
| 3 | `m_drop15` | 0.8718 | 0.7467 | 40/50 | shortlist version |
| 4 | `m_drop20` | 0.8715 | 0.7467 | 48/50 | dropout 0.20 |
| 5 | `m_real_extras_1x` | 0.8715 | 0.7467 | 50/50 | default recipe + 1× extras |

Top by `dev_token_acc` (Task 2 specialists):

| Run | dev_token_acc | best_epoch | Wave |
|---|---:|---:|---|
| **`g_drop15_nosched_t2`** | **0.4545** | 80/54 | 2 |
| `f_extras_1x_100_t2` | 0.4511 | 65/100 | 1 |
| `f_no_sched_100_t2` | 0.4491 | 75/100 | 1 |

### Level 3 — data scaling (dev holdout)

| Data recipe | Run | MRR | tok-acc | train_seconds |
|---|---|---:|---:|---:|
| Real only | `m_real_only` | 0.866 | 0.441 | 162 |
| +250/family (1×) | `m_real_extras_1x` | **0.872** | 0.439 | 183 |
| +250/family (2×) | `m_real_extras_2x` | 0.869 | 0.436 | 184 |
| +500/family | `f_extras_500_100_mrr` | 0.869 | 0.446 | 2067 |

**Finding:** 1× synthetic extras helps modestly; more volume diverges from the real distribution.

---

## What worked

- **Rule-constrained decoding** gave a visible, qualitative jump on Task 2.
  Unconstrained greedy frequently produced step sequences violating the grammar;
  constrained beam=5 stays physically plausible end-to-end.
- **Wave 2 fine grid (~45 min GPU).** Five rows around the winning recipe
   tightened Task 2: `g_drop15_nosched_t2` reached **0.4545 token-acc** (+0.3 pp
   over Wave 1). Task 1 MRR plateau held at ~0.873 — Wave 1 winner unchanged.
- **Hybrid two-model approach.** Decoupling selection criteria was a small change
  with a real delta on Task 2.
- **N-gram suffix-backoff as a baseline.** Sanity-checked everything and gave
  the jury a transparent comparison point. The Transformer beating it by ~6pp
  is more convincing than reporting an absolute number alone.
- **AMP re-enabled for Wave 1.** A prior config ran with `amp: false` — 2×
  slower with no measurable quality difference. Restored at the finalist stage.
- **Synthetic extras at 1× (250/family).** +1.2pp Top-1 vs no-extras baseline.
  Cheap, deterministic, manifested via `manifest.json`.

## What didn't

- **2× synthetic extras was *worse* than 1×** (`m_real_extras_2x` MRR 0.8693 vs
  `m_real_extras_1x` 0.8715). Diminishing returns — augmentation diverged from
  the real distribution enough to hurt. `f_extras_500_100_mrr` confirmed:
  500/family at 100 epochs scored MRR 0.8686, below the 1× recipe.
- **Earlier submission used the wrong format.** First-pass predictions used the
  column names `SEQUENCE_ID/RANK/STEP` instead of the judge's
  `EXAMPLE_ID/RANK_1..5`. Caught during the verification pass; rewrote
  `scripts/predict_submission.py` to auto-detect input format and emit the
  judge's exact schema.
- **A teammate's commit flipped 99% of repo line endings to CRLF.** Caused a
  noisy 400k-line "diff" on a 12k-line repo. Resolved by cherry-picking only
  the semantic changes and re-normalizing on pull.
- **Anomaly SCORE is a heuristic, not a true likelihood.** The proper
  implementation is per-step LM log-probability summed over the sequence.
  Skipped because the rule validator already classifies perfectly on the dev
  set, so an improved SCORE wouldn't move F1 or AUC on this benchmark.

## What we'd do with another 36 hours

1. **Seed-robustness check (~30 min).** Run `make_dev_split.py` with seeds
   `{7, 13, 99}` and re-score the top 3 checkpoints on each.

2. ~~**Wave 2 fine grid (~45 min).**~~ **Done** — `g_drop15_nosched_t2` best T2.

3. ~~**RoPE + RMSNorm + pre-norm Transformer (~2 h impl, ~1 h train).**~~ **Wave 3
   submitted** — results pending on Leonardo.

4. **Per-step LM likelihood for Task 3 SCORE.** Sum log-probabilities of each step
   given prior context — ~50 lines.

5. **Larger model (d_model=384, 8 layers).** Skip if seed-robustness shows
   medium-config noise dominates current gaps.

---

## Track 1 brief — compliance checklist

Mapped to the [Zero One Track 1](https://docs.zero-one.lumos-consulting.at/tracks/track-1/) deliverables.

| Requirement | Status | Evidence |
|---|---|---|
| **MVP:** reproducible end-to-end workflow | ✅ | `Makefile`, `docs/PIPELINE.md`, pytest |
| **MVP:** synthetic data generation | ✅ | `scripts/generate_extra_sequences.py` (+750 seq) |
| **MVP:** at least one trained model | ✅ | Transformer ~4M params + n-gram baseline |
| **MVP:** baseline vs post-training comparison | ✅ | Dev metrics, unified dashboard (`make run-dashboard`) |
| **MVP:** documented benchmark | ✅ | `LEADERBOARD_FINAL`, `REPORT.md`, per-run JSONs |
| **Stretch:** multiple architectures | 🔄 | Vanilla done; Wave 3 modern stack in flight |
| **Stretch:** scaling effects | ✅ | Level 3 data scaling table (1× helps, 2×/500 hurts) |
| **Stretch:** demonstrator (before/after) | ✅ | `make run-dashboard` — Live demo tab, n-gram vs Transformer, 3 tasks |
| **Stretch:** optional process parameters | 🔄 Wave 5 | Training-only aux tokens; ADR 0004; run if Wave 3/4 plateau |
| **Stretch:** OOD / modified sequences | ⏳ judge | Task 4 evaluated by organizers on hidden set |
| **Stretch:** model size ladder | ⏳ | ~4M params only; larger config scaffolded, not run |
| **Format:** training + eval report | ✅ | This report + GPU summary + leaderboard |
| **Learning objectives** | ✅ | Leonardo training, synthetic data, benchmarking, scaling analysis |

**After Wave 3/4 finish:** `make leonardo-wave5-if-needed`, rebuild leaderboard, `make regenerate-submission`, update this table.

---

## Credits & dependencies

**People**: Andrija Jovanovic, Istvan Beregszaszi, Thánh Trung Nguyen.

**Compute**: CINECA Leonardo (EuroHPC) GPU partition, reservation `s_tra_ncc`,
account `EUHPC_D30_031`.

**Code & data we used as-is**:
- `data/raw/infineon/training_data/generate_sequences.py` — the official
  Infineon synthetic-sequence generator and rule validator. Provided by the
  organizers; remains under their license. We import the validator unchanged
  in `src/eval/rule_validator.py`.
- `EVAL_DATA/eval_metrics.py` — the official scoring script. Used as-is for
  local verification; the jury runs it against the hidden ground truth.

**Libraries**: PyTorch (training + inference), NumPy + pandas (data plumbing),
scikit-learn (a few utilities), PyYAML (sweep configs), Streamlit
(`src/app/dashboard.py` — metrics, leaderboard, training curves, live demo), pytest + ruff
(tests + lint), Matplotlib + Plotly (figures).

**External APIs**: none.

**AI coding tools**: GitHub Copilot and Anthropic Claude were used as
sounding-boards and for boilerplate generation (CLI argparse, dashboard
scaffolding, large rename refactors). All experimental decisions, hyperparameter
choices, and the final architecture are team decisions. Every PR was
human-reviewed.

**Inspiration**: nanoGPT (Karpathy) for the from-scratch decoder approach;
the open-LLM community for RoPE/SwiGLU/RMSNorm patterns we'd port next.

---

*Submitted by team **42 AI Crash** for Zero One Hack_01, 2026-05-30.*
