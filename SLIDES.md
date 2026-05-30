---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- 10-slide narrative for the 3-minute pitch.
     Render to PDF:  marp SLIDES.md -o slides/industrial_pitch_deck.pdf
     Or copy each slide block into slides/industrial_pitch_deck.pptx.
     One H2 per slide, kept short for verbal pitch (45–60 words). -->

# Industrial AI — Process Sequence Models
### Team **42 AI Crash** · Infineon track

**Andrija Jovanovic · Istvan Beregszaszi · Thánh Trung Nguyen**

We teach a small Transformer to learn the *logic* of semiconductor fabrication
sequences — predicting the next step, completing partial routes, and detecting
process-rule violations.

---

## Why this matters

Semiconductor wafers go through **115–150 ordered fabrication steps**
across 3 product families.

A wrong step → yield loss → scrapped wafer → real money.

A model that captures process logic can:
- flag anomalies before they reach the line,
- assist engineers planning new product variants,
- generalize without re-training for every routing change.

**The challenge:** can a model learn the *grammar* of the process —
not just memorize examples?

---

## Approach — 5 decisions

1. **From-scratch decoder Transformer** (~4M params, 6 layers, 8 heads).
   Domain vocab is 140 multi-word step tokens — pretrained LLMs add no value.
2. **Rule-constrained beam search** for completion (beam=5, top-k filtered
   through the official rule grammar at every step).
3. **N-gram suffix-backoff baseline** (order 12) for transparency.
4. **Hyperparameter sweep on Leonardo A100s** — 17 configs ranked by dev MRR
   and dev token-accuracy.
5. **Hybrid two-model submission**: best by MRR for next-step, best by
   token-acc for completion.

---

## Why a hybrid submission?

The config that maximizes **MRR** is **not** the config that maximizes
**token-accuracy**. We trained both specialists and used each on the task it wins.

| Task | Model | Selected by | Dev metric |
|---|---|---|---:|
| 1 — Next-step | `f_drop15_100_mrr` | `dev_mrr` | **MRR 0.8731** |
| 2 — Completion | `f_extras_1x_100_t2` | `dev_token_acc` | **0.4511** |
| 3 — Anomaly | rule validator + heuristic SCORE | — | **F1 1.00** |

One pipeline (`predict_submission.py`) emits judge-format CSVs;
the hybrid is a single `cp completion.csv` swap.

---

## Results — Task 1 (next-step prediction)

| Model | Top-1 | Top-5 | MRR |
|---|---:|---:|---:|
| N-gram suffix-backoff | 0.687 | 0.928 | 0.807 |
| Transformer shortlist best | 0.747 | 1.000 | 0.872 |
| **Transformer Wave-1 hybrid** | **0.748** | **1.000** | **0.873** |

- **Top-5 = 1.000** — the correct next step is *always* in our top 5.
- **+6.6 pp Top-1 over the n-gram baseline** on the same dev split.
- 17 sweep runs, full per-epoch metrics auditable in `artifacts/sweeps/`.

---

## Results — Tasks 2 & 3

**Task 2 — Sequence completion** (overall, dev split):

| Metric | Value |
|---|---:|
| Token accuracy | **0.451** |
| Normalized edit distance | **0.224** |
| Decoding | rule-constrained beam search, width 5 |

**Task 3 — Anomaly detection** (rule validator + SCORE):

| Metric | Value |
|---|---:|
| Accuracy / Precision / Recall / F1 | **1.00** |
| AUC (continuous SCORE) | **1.00** |
| Rule-attribution accuracy | 0.69 |

Rule validator is the official `validate_sequence` — same grammar the jury uses.

---

## Sweep evidence — 17 runs on Leonardo

| Rank | Run | dev MRR | best_epoch | Note |
|---:|---|---:|---:|---|
| 1 | `f_drop15_100_mrr` | **0.8731** | 84/100 | dropout 0.15, AMP on |
| 2 | `m_drop15` | 0.8718 | 40/50 | shortlist version |
| 3 | `m_drop20` | 0.8715 | 48/50 | |
| 4 | `m_real_extras_1x` | 0.8715 | 50/50 | default + 1× synthetic |
| 5 | `m_lr_hi` | 0.8707 | 35/50 | lr 6e-4 |

**Negative result, also reported:** 2× synthetic extras worse than 1× (MRR 0.8693 vs 0.8715).
More augmentation diverged from real distribution.

---

## What worked / what didn't

**Worked**
- Rule-constrained beam search → visibly better completions.
- Hybrid two-model approach → +1.3 pp MRR, +1.2 pp token-acc vs single model.
- AMP re-enabled → 2× faster training, zero quality loss.

**Didn't**
- 2× / 500-per-family augmentation → worse than 1×.
- Earlier submission used wrong column names → caught + fixed in
  `predict_submission.py` (auto-detects judge vs legacy format).
- Friend's editor flipped 99 % of repo to CRLF → cherry-picked semantic changes only.

---

## What we'd do next

| Time | Action | Why |
|---|---|---|
| 30 min | **Seed-robustness sweep** (3 seeds × top 3 ckpts) | Verify the top-1 ranking isn't an artifact of split |
| 45 min | **Wave 2 fine grid** (already scaffolded) | Tighten dropout / scheduler around winner |
| 3 h | **Modern architecture** (RoPE + RMSNorm + SwiGLU) | Most likely way to break the MRR plateau |
| 1 h | **Per-step LM likelihood for Task-3 SCORE** | Replace heuristic with proper sequence probability |

Everything is wired and ready (`make leonardo-wave2` is one command).

---

## Code, license, reproducibility

- **Repo (public, MIT):** github.com/beistvan/industrial-42-ai-crash · branch `wave1-submission`
- **One-command quickstart**: `make dev-split && make train-ngram && pytest -q` (~5 min, CPU)
- **Full retraining**: 2 sbatch jobs on any A100, ~90 min total. No API keys, no external services.
- **REPORT.md** + 17 per-run JSONs + judge-format submission CSVs all checked in.

> **Submission is shipped. The pipeline is reusable. Everything is honest about
> what worked and what didn't.**

Team: **Andrija · Istvan · Trung** — Industrial AI track, Zero One Hack 01.
