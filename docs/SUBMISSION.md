# Submission plan — Industrial AI (Infineon)

**Deadline:** Sunday **10:00 CEST** (submit by **09:45** — Tally can flake under load)  
**Form:** {TALLY_FORM_URL}  
**Team:** Andrija Jovanovic, Istvan Beregszaszi, Thánh Trung Nguyen  
**Repo:** https://github.com/beistvan/industrial-42-ai-crash (branch `wave1-submission`, MIT)

---

## At a glance — what goes where

| Deliverable | Where it lives | Tally field |
|---|---|---|
| **Code + train configs** | Public GitHub repo (see §1) | Repository URL |
| **Results + judge CSVs** | Repo + `REPORT.md` (see §2) | (in repo; referenced in pitch) |
| **Pitch slides (PDF)** | `slides/industrial_pitch_deck.pdf` from `SLIDES.md` (see §3) | Slides upload |
| **Demo video (≤2 min)** | MP4 or unlisted link (see §4) | Demo video |
| **Team name** | — | Text field |

Operational runbook (GPU jobs, regen, status): [`scripts/README.md`](../scripts/README.md) · [`HANDOFF.md`](../HANDOFF.md)

**Live status:** `bash scripts/leonardo/status_now.sh`

---

## 1. Code and training config

The jury clones the repo and expects honest **how to train** instructions. Everything below is in the public repo.

### Repo layout (what reviewers open first)

```
README.md                          Setup, quickstart, reproduce submission
REPORT.md                          Jury write-up (problem, approach, results)
requirements.txt                   Python deps
LICENSE                            MIT

configs/
  transformer_medium.yaml          Vanilla Wave 1–2 stack (~4M params)
  transformer_modern.yaml          Wave 3+ stack (RoPE, RMSNorm, SwiGLU)
  sweeps/WINNING_RECIPES.md          Current submission rows (T1 row 4, T2 row 4)
  sweeps/leonardo_modern.yaml        Wave 3 — T1 winner `h_mod_nosched_mrr`
  sweeps/leonardo_fine.yaml          Wave 2 — T2 winner `g_drop15_nosched_t2`
  sweeps/leonardo_final.yaml         Wave 1 history (superseded T1)

scripts/
  train_transformer.py             Single-run training entry point
  sweep_transformer.py           Expand sweep YAML row → CLI args
  predict_submission.py            Write judge CSVs
  regenerate_submission.sh         Pick best T1/T2 + GPU predict

src/                               Model, eval, rule validator, demo app
tests/                             pytest suite (~33 tests)
EVAL_DATA/                         Organizer eval inputs + eval_metrics.py
result/submission/                 Judge-format CSVs (nextstep, completion, anomaly)
models/sweeps/*.pt.best            Submission checkpoints (2 files — see models/README.md)
src/ml/                            Model **source code** (always keep — not weights)
artifacts/sweeps/                  Per-run metrics JSON + LEADERBOARD_FINAL.csv
```

Checkpoints (`models/sweeps/*.pt.best`) are gitignored except the two submission winners (~43 MB total). Training commands below reproduce them; see [`models/README.md`](../models/README.md) for what to keep vs delete.

### Architecture (both stacks)

| | Vanilla (Wave 1–2 T2) | Modern (Wave 3 T1) |
|---|---|---|
| Config | `configs/transformer_medium.yaml` | `configs/transformer_modern.yaml` |
| Layers / heads / d_model | 6 / 8 / 256 | 6 / 8 / 256 |
| ~params | ~4M | ~4M |
| Extras | `data/generated/infineon/` (+250/family) | same |
| Decoding | rule-constrained beam (T2) | top-k (T1 next-step) |

### Current submission models (hybrid — two checkpoints)

| Task | Checkpoint | Sweep YAML + run | Selection |
|---|---|---|---|
| **T1** next-step | `models/sweeps/h_mod_nosched_mrr.pt.best` | `configs/sweeps/leonardo_modern.yaml` → `h_mod_nosched_mrr` | `save_best_by: dev_mrr` |
| **T2** completion | `models/sweeps/g_drop15_nosched_t2.pt.best` | `configs/sweeps/leonardo_fine.yaml` → `g_drop15_nosched_t2` | `save_best_by: dev_token_acc` |
| **T3** anomaly | *same as T1* | rule validator + LM SCORE | no separate train |

**T1 recipe (`h_mod_nosched_mrr`):** modern arch, dropout 0.15, no scheduler, 100 ep, AMP, label smoothing 0.1, extras 1×, best @ ep 85.

**T2 recipe (`g_drop15_nosched_t2`):** vanilla arch, dropout 0.15, no scheduler, save by token-acc, best @ ep 80.

Full sweep row definitions: [`configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md) (T1 **row 4**, T2 **row 4**).

### Reproduce from a clean checkout

**CPU smoke (~5 min)** — proves repo runs, no GPU needed:

```bash
git clone -b wave1-submission https://github.com/beistvan/industrial-42-ai-crash.git
cd industrial-42-ai-crash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
make dev-split && make train-ngram && pytest -q
```

**Train submission checkpoints (GPU, ~2–3 h on one A100):**

```bash
# Data (once)
python scripts/make_dev_split.py --force
python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

# Task 1 leader (Wave 3 modern)
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_modern.yaml --stage finalists --row 4
# → models/sweeps/h_mod_nosched_mrr.pt.best

# Task 2 leader (Wave 2 fine grid)
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_fine.yaml --stage finalists --row 4
# → models/sweeps/g_drop15_nosched_t2.pt.best
```

On Leonardo, submit sweeps via `scripts/leonardo/submit_sweep.sh` — see [`docs/LEONARDO_GPU_RUNBOOK.md`](LEONARDO_GPU_RUNBOOK.md).

**Generate judge CSVs:**

```bash
make leonardo-leaderboard-final   # refresh picks from metrics JSONs
make regenerate-submission        # Slurm GPU predict on Leonardo login node
# local GPU: FORCE_LOCAL=1 make regenerate-submission
```

Output: `result/submission/{nextstep,completion,anomaly}.csv`

---

## 2. Results

### Headline numbers (local dev holdout — not organizer hidden test)

Eval split: `data/processed/dev_eval/` (600 T1/T2 items). Full per-run table: [`artifacts/sweeps/LEADERBOARD_FINAL.csv`](../artifacts/sweeps/LEADERBOARD_FINAL.csv) and [`.md`](../artifacts/sweeps/LEADERBOARD_FINAL.md).

| Task | Metric | N-gram baseline | Hybrid submission | Lift |
|---|---:|---:|---:|---|
| **1** Next-step | MRR | 0.807 | **0.873** | +6.6 pp |
| **1** Next-step | Top-1 | 0.687 | **0.75** | +6.3 pp |
| **2** Completion | Token-acc | 0.421 | **0.455** | +3.4 pp |
| **2** Completion | NED | — | **0.223** | lower is better |
| **3** Anomaly | F1 (invalid) | — | **1.00** | rule validator |

### Submission files (Infineon track repo deliverables)

| File | Rows | Source model | Notes |
|---|---:|---|---|
| [`result/submission/nextstep.csv`](../result/submission/nextstep.csv) | 601 | `h_mod_nosched_mrr` | Top-5 ranked next steps |
| [`result/submission/completion.csv`](../result/submission/completion.csv) | 601 | `g_drop15_nosched_t2` | Rule-constrained beam completion |
| [`result/submission/anomaly.csv`](../result/submission/anomaly.csv) | 988 | T1 model + validator | `IS_VALID`, `SCORE`, `PREDICTED_RULE` |

### Evidence in repo

| Artifact | Purpose |
|---|---|
| [`REPORT.md`](../REPORT.md) | Jury-facing narrative, ADR links, honest limits |
| [`artifacts/sweeps/LEADERBOARD_FINAL.md`](../artifacts/sweeps/LEADERBOARD_FINAL.md) | GPU sweep summary by Track 1 level |
| `artifacts/sweeps/*.json` | Per-epoch metrics, loss curves (27+ runs) |
| `artifacts/ngram_metrics.json` | Baseline numbers |
| [`EVAL_DATA/eval_metrics.py`](../EVAL_DATA/eval_metrics.py) | Official scoring script |

### Score a checkpoint locally

```bash
python -m src.eval.local_eval \
  --model models/sweeps/h_mod_nosched_mrr.pt.best \
  --device cuda \
  --out artifacts/local_eval_h_mod_nosched_mrr.json
```

Unified dashboard (metrics, leaderboard, training curves, live demo): `make run-dashboard`

---

## 3. Pitch slides

**Styled deck (preferred for Tally):** [`slides/industrial_pitch_deck.pptx`](../slides/industrial_pitch_deck.pptx) — team layout and visuals.

**Content source:** [`SLIDES.md`](../SLIDES.md) — 10-slide Marp narrative (kept in sync with the pitch).

**Export PDF** (upload to Tally):

```bash
make slides-pdf
# → slides/industrial_pitch_deck.pdf
```

Uses LibreOffice to convert the PPTX when available; otherwise renders `SLIDES.md` via Marp (content-only theme). On a laptop without LibreOffice, open the PPTX in PowerPoint/Keynote/Google Slides → Export PDF.

---

## 4. Demo video (≤2 minutes)

Show the **system running**, not slideware:

1. **~15 s** — problem (wrong fab step → scrap)  
2. **~60 s** — Streamlit dashboard: **Live demo** tab — n-gram vs Transformer on same prefix (`make run-dashboard`)  
3. **~30 s** — one number (MRR 0.873 / tok 0.455 vs baseline)  
4. **~15 s** — rule validator flagging an invalid sequence  

Format: MP4, 1080p, audio. Upload to Tally or paste unlisted YouTube/Vimeo/Loom link.

---

## 5. Tally form checklist

Before **09:45**:

- [ ] **Team name** matches registration  
- [ ] **Repo URL** — public, MIT, branch `wave1-submission`  
- [ ] **README.md** + **REPORT.md** + **requirements.txt** + **LICENSE** present  
- [ ] **Train configs** documented (§1 above — sweep YAMLs + commands)  
- [ ] **Judge CSVs** in `result/submission/` — all non-empty (601/601/988 lines)  
- [ ] **Slides PDF** — `slides/industrial_pitch_deck.pdf` from `SLIDES.md`  
- [ ] **Demo video** — under 2 minutes  
- [ ] No secrets in git history  
- [ ] Fresh clone smoke test passes (`make dev-split && make train-ngram && pytest -q`)

Submission is **final** (`h_mod_nosched_mrr` + `g_drop15_nosched_t2`). Re-run only if a new sweep beats the bar: `make leonardo-leaderboard-final && make regenerate-submission`.

---

## 6. Organizer requirements (reference)

The Tally form collects four fields: **team name**, **repo URL**, **slides PDF**, **demo video**.

The repo must additionally contain (Infineon track):

- Eval submission files (`nextstep.csv`, `completion.csv`, `anomaly.csv`)  
- Training reproducibility (configs, scripts, metrics JSONs)  
- `REPORT.md` with baseline comparison and honest evaluation  
- Demo that shows baseline vs trained on identical inputs  

Full organizer wording: Zero One hackathon submission brief. Judging rubric: `/judging/rubrics.md` (organizer repo).

---

## Questions?

`#industrial-ai` on Discord, or Lumos front desk.

Good luck.
