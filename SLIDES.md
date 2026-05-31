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
### Team **42 AI Crash** · Infineon track · [Track 1 Levels](https://docs.zero-one.lumos-consulting.at/tracks/track-1/)

**Andrija Jovanovic · Istvan Beregszaszi · Thánh Trung Nguyen**

We teach a small Transformer to learn the *logic* of semiconductor fabrication
sequences — predicting the next step, completing partial routes, and detecting
process-rule violations.

---

## Track 1 levels — our progress

| Level | What | Status | Dev headline |
|---|---|---|---|
| **1** | Data + n-gram baseline | ✅ done | MRR 0.807 · tok 0.421 |
| **2** | Train → tune → benchmark | ✅ done | MRR **0.874** · tok **0.455** |
| **3** | Scaling + modern architecture | ✅ done | Wave 3 broke T1 plateau |

**Baseline → trained → optimized:** n-gram → Wave 1 Transformer → Wave 3 modern T1 + Wave 2 T2 specialist.

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
3. **N-gram suffix-backoff baseline** (order 12) for transparency — Level 1.
4. **Hyperparameter sweep on Leonardo A100s** — 27 configs ranked by dev MRR
   and dev token-accuracy — Level 2.
5. **Hybrid two-model submission**: best by MRR for next-step, best by
   token-acc for completion.

---

## Why a hybrid submission?

The config that maximizes **MRR** is **not** the config that maximizes
**token-accuracy**. We trained both specialists and used each on the task it wins.

| Task | Model | Selected by | Dev metric |
|---|---|---|---:|
| 1 — Next-step | `h_mod_nosched_mrr` | `dev_mrr` | **MRR 0.8735** |
| 2 — Completion | `g_drop15_nosched_t2` | `dev_token_acc` | **0.4545** |
| 3 — Anomaly | rule validator + T1 LM SCORE | — | **F1 1.00** |

One pipeline (`predict_submission.py`) emits judge-format CSVs;
the hybrid is a single `cp completion.csv` swap.

---

## Results — baseline vs trained vs optimized

| Stage | Model | Task 1 MRR | Task 1 Top-1 | Task 2 tok-acc |
|---|---|---:|---:|---:|
| Level 1 baseline | n-gram | 0.807 | 0.687 | 0.421 |
| Level 2 trained | Wave 1 Transformer | 0.873 | 0.748 | 0.437 |
| **Submission** | Wave 3 T1 + Wave 2 T2 | **0.874** | **0.75** | **0.455** |

- **Top-5 = 1.000** on Task 1 — correct next step always in top 5.
- **+6.3 pp Top-1 over n-gram** on the same dev split.
- 27 sweep runs, full per-epoch metrics in `artifacts/sweeps/`.

---

## Level 3 — scaling & architecture

**Data scaling (negative result, also reported):**

| Recipe | MRR | tok-acc |
|---|---:|---:|
| Real only | 0.866 | 0.441 |
| +250/family (1×) | **0.872** | 0.439 |
| +250/family (2×) | 0.869 | 0.436 |
| +500/family | 0.869 | 0.446 |

More augmentation diverged from real distribution.

**Architecture (positive result):** Wave 3 modern stack (RoPE + RMSNorm + SwiGLU)
→ **`h_mod_nosched_mrr`** MRR **0.8735** (+0.05pp vs Wave 1). Wave 4 T2 prefix training did not beat T2 bar.

---

## What worked / what didn't

**Worked**
- Rule-constrained beam search → visibly better completions.
- Wave 2 fine grid → +0.3 pp tok-acc over Wave 1 T2 specialist.
- Wave 3 modern architecture → new Task-1 leader (+0.05pp MRR).
- Hybrid two-model approach → best checkpoint per task.
- LM log-prob SCORE for Task 3 (teacher-forced on T1 checkpoint).

**Didn't**
- 2× / 500-per-family augmentation → worse than 1×.
- Wave 4 Task-2 prefix training → no gain over `g_drop15_nosched_t2`.
- Earlier submission used wrong column names → caught + fixed.

---

## What we'd do next

| Time | Action | Why |
|---|---|---|
| done | ~~Wave 2 fine grid~~ | `g_drop15_nosched_t2` best T2 |
| done | ~~Wave 3 modern architecture~~ | `h_mod_nosched_mrr` best T1 |
| done | ~~LM SCORE for Task 3~~ | shipped in `anomaly.csv` |
| 30 min | **Seed-robustness sweep** (3 seeds × top 3 ckpts) | Verify ranking isn't split artifact |
| 2 h | **Larger model (d_model=384)** | Only if seed sweep shows stable gap |

---

## Code, license, reproducibility

- **Repo (public, MIT):** github.com/beistvan/industrial-42-ai-crash · branch `wave1-submission`
- **One-command quickstart**: `make dev-split && make train-ngram && pytest -q` (~5 min, CPU)
- **Dashboard**: `make run-dashboard` — overview, leaderboard, training curves
- **REPORT.md** + per-run JSONs + judge-format submission CSVs all checked in.

> **Submission is shipped. The pipeline is reusable. Everything is honest about
> what worked and what didn't.**

Team: **Andrija · Istvan · Trung** — Industrial AI track, Zero One Hack 01.
