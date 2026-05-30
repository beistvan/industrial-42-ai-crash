# 42 AI Crash — Industrial AI (Infineon)

## Team

- **Andrija Jovanovic** — ML engineering, model training, evaluation
- **Istvan Beregszaszi** — data pipeline, infrastructure
- **Thánh Trung Nguyen** — experimentation, analysis

Track: **Industrial AI (Infineon)**

---

## TL;DR

We trained a decoder-only Transformer from scratch on synthetic semiconductor process sequences (MOSFET, IGBT, IC families) to learn industrial process logic. The model predicts the next fabrication step, completes partial sequences, and detects process rule violations. Our best configuration (`m_real_extras_2x`) achieves **MRR 0.872** and **Top-1 accuracy 74.8%** on next-step prediction, compared to an n-gram baseline of MRR 0.807 / Top-1 68.7%.

---

## Problem

Industrial semiconductor manufacturing processes are long, ordered sequences of steps (115–150 steps per wafer lot). The challenge: can a model learn the *logic* of these sequences — not just memorize them, but understand which steps must follow which, detect when a rule is violated, and predict or complete unseen variants?

This matters because even small process deviations cause yield loss or wafer scraps. A model that understands process logic could flag anomalies early, assist process engineers in route planning, and generalize to new product families without retraining.

We focused on all three evaluation tasks: next-step prediction, sequence completion, and anomaly detection, benchmarked against an n-gram suffix-backoff baseline.

---

## Approach

- **Decoder-only Transformer** (6 layers, 8 heads, d_model=256, ~4M parameters) trained from scratch on process step tokens. No pre-trained LLM was used — the vocabulary is domain-specific (136–146 unique steps per family) and sequence structure differs fundamentally from natural language.

- **Rule-constrained decoding** for sequence completion: at each step, the top-k model predictions are filtered through a deterministic rule validator before accepting the next token. This enforces process grammar without penalizing the model during training.

- **N-gram suffix-backoff baseline** (order 12) trained on the same data as a transparent, fast comparison point. No neural components — purely frequency-based prediction from the training sequences.

- **Hyperparameter sweep on LEONARDO A100 GPUs**: 12 configurations trained for 50 epochs each, varying learning rate, dropout, label smoothing, scheduler, and data augmentation. Best config selected by dev MRR.

- **Anomaly detection via rule validator**: the `classify_sequence` function checks sequences against 10 process grammar rules. Binary valid/invalid output with rule attribution.

---

## How to run it

```bash
# Clone and install
git clone https://github.com/beistvan/industrial-models-that-learn-how-processes-unfold
cd industrial-models-that-learn-how-processes-unfold
pip install -r requirements.txt

# Train n-gram baseline
python scripts/train_ngram.py

# Generate submission CSVs (transformer model)
python scripts/predict_submission.py \
    --model models/sweeps/m_real_extras_2x.pt.best \
    --eval-valid  data/raw/infineon/eval/eval_input_valid.csv \
    --eval-anomaly data/raw/infineon/eval/eval_input_anomaly.csv \
    --out-dir artifacts/submission

# Run Streamlit demo UI
streamlit run src/app/main.py
```

**Requirements:**
- Python 3.10+
- PyTorch (CPU sufficient for inference, GPU for training)
- The `data/raw/infineon/` directory with the Infineon dataset (not included in repo — provided by organizers)
- For training on LEONARDO: see `scripts/leonardo/sweep_array.slurm`

---

## Results

### Task 1 — Next-Step Prediction

| Model | Top-1 | Top-3 | Top-5 | MRR |
|---|---|---|---|---|
| N-gram baseline | 68.7% | 92.8% | 92.8% | 0.807 |
| **Transformer (m_real_extras_2x)** | **74.8%** | — | — | **0.872** |

N-gram per-family breakdown:

| Family | Top-1 | Top-3 | MRR |
|---|---|---|---|
| MOSFET | 69.0% | 91.5% | 0.803 |
| IGBT | 73.5% | 97.0% | 0.852 |
| IC | 63.5% | 90.0% | 0.768 |

### Task 2 — Sequence Completion (N-gram baseline)

| Family | Exact Match | Token Acc | NED |
|---|---|---|---|
| MOSFET | 1.5% | 49.8% | 0.159 |
| IGBT | 0.0% | 44.8% | 0.225 |
| IC | 0.0% | 31.8% | 0.289 |
| **Overall** | **0.5%** | **42.1%** | **0.224** |

### Task 3 — Anomaly Detection (Rule validator)

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| Rule Attribution Accuracy | 69.0% |

The rule validator achieves perfect binary detection on the training-distribution eval set because it directly encodes the process grammar. Rule attribution (identifying *which* rule was violated) reaches 69% — the harder task of explaining why a sequence is invalid.

Submission files are in `extras/results/`:
- `nextstep.csv` — Task 1 (transformer model)
- `completion.csv` — Task 2 (transformer model, rule-constrained greedy)
- `anomaly.csv` — Task 3 (rule validator)

---

## What worked

- **Rule-constrained decoding** made a visible difference in completion quality — unconstrained greedy frequently produced step sequences that violated process grammar, while constrained decoding stays physically plausible throughout.

- **The sweep on LEONARDO** found a meaningful spread across 12 configurations. The `m_real_extras_2x` config (with synthetic data augmentation, cosine scheduler, label smoothing 0.1) consistently outperformed the no-augmentation baseline, confirming that synthetic data diversity helps even when the grammar is the same.

- **N-gram baseline is strong** — MRR 0.807 on suffix-backoff alone is a high bar, which made the transformer's improvement to 0.872 a meaningful and honest result rather than beating a trivial baseline.

---

## What didn't work

- **AMP (mixed precision) training** failed silently on LEONARDO due to a CUDA driver version mismatch (driver supports CUDA 12.2, torch 2.12 expected 12.4+). We disabled AMP and trained in full fp32, which cost roughly 30% speed but produced correct results.

- **Beam search for Task 2** was implemented but not used in the final submission — beam width > 1 combined with rule-constrained filtering was too slow for 600 sequences on CPU (est. 4+ hours). Greedy rule-constrained decoding was used instead.

- **Transformer anomaly scoring** via per-token perplexity was planned but not completed. The rule validator gives perfect accuracy on known rule types but will not generalize to novel violations outside the grammar — a learned anomaly score would handle this better.

---

## What we'd do with another 36 hours

- Train two additional model sizes (d_model=128 and d_model=512) to produce a scaling curve showing compute vs. performance trade-off.
- Replace binary rule-validator anomaly score with transformer perplexity score for continuous SCORE output, enabling a real ROC-AUC measurement.
- Run beam search (width=4) for Task 2 completion on LEONARDO GPU to measure the exact match improvement over greedy.
- Evaluate the IC family more carefully — it shows the largest performance gap (Top-1 63.5% vs IGBT 73.5%), likely because IC sequences have more branching in the grammar.

---

## Track-specific deliverables

- [x] `extras/results/nextstep.csv` (Task 1 format)
- [x] `extras/results/completion.csv` (Task 2 format)
- [x] `extras/results/anomaly.csv` (Task 3 format)
- [x] Training checkpoint: `models/sweeps/m_real_extras_2x.pt.best`
- [x] Training config: `configs/sweeps/leonardo_v1.yaml`
- [x] Slurm job script: `scripts/leonardo/sweep_array.slurm`
- [ ] Loss curves — to be added
- [ ] eval_metrics.py scores with per-family breakdown — pending ground truth from organizers
- [ ] Demo video

---

## Credits & dependencies

**Open-source libraries:**
- PyTorch 2.3.0
- Streamlit (demo UI)
- NumPy, pandas

**Pre-trained models:** none — trained from scratch on provided data

**External APIs:** none

**AI coding assistants:** Claude Code (Anthropic) — used throughout for implementation, debugging, and HPC job scripting

**Datasets:** Infineon synthetic process sequences provided by hackathon organizers (IC, IGBT, MOSFET families, 3,000 training sequences + 9 reference CSVs)

---

*Submitted by team **42 AI Crash** for Zero One Hack_01, 2026-05-30.*
