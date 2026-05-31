# Industrial AI — Process Sequence Models (Infineon track)

Hackathon entry for the **42 AI Crash — Industrial AI** track.
We learn the *logic* of semiconductor fabrication process sequences from a small
real corpus (~3,000 sequences, 3 product families, ~140-token vocabulary) and
generate predictions for three tasks: next-step, sequence completion, and
anomaly detection against a 10-rule process grammar.

- **License**: MIT (see `LICENSE`)
- **Team**: Andrija Jovanovic, Istvan Beregszaszi, Thanh Trung Nguyen
- **Track**: Industrial AI (Infineon)
- **Report**: [`REPORT.md`](REPORT.md) · **Handoff**: [`HANDOFF.md`](HANDOFF.md)
- **Submission plan**: [`docs/SUBMISSION.md`](docs/SUBMISSION.md)

## What's in this repo

```
src/, scripts/, tests/, configs/      Code + tests + sweep YAMLs
EVAL_DATA/                            Judge eval inputs + eval_metrics.py
result/submission/                    Judge CSVs (nextstep, completion, anomaly)
artifacts/sweeps/                     Per-run metrics + LEADERBOARD_FINAL
models/sweeps/                        Submission checkpoints (2 × .pt.best)
data/raw/infineon/                    Organizer-provided source data
data/generated/infineon/              Synthetic augmentation (deterministic)
docs/                                 ADRs, data spec, ENGINEERING_PRACTICES, SUBMISSION
slides/                               Pitch deck (PPTX; see slides/README.md)
Makefile                              dev-split, smoke, leaderboard, regen, dashboard
REPORT.md, README.md, HANDOFF.md      Jury + teammate docs
```

## What we built (one-line each)

- A **compact decoder-only Transformer** (~4M params, 6 layers, 8 heads, d_model=256), trained from scratch on the Infineon process step vocabulary.
- An **n-gram suffix-backoff baseline** for transparency and a fair comparison point.
- A **rule-validator + continuous SCORE** anomaly head reusing the official `validate_sequence` grammar.
- A **hybrid submission**: best Task-1 model for nextstep + best Task-2 specialist for completion + rule validator for anomaly.

## Headline numbers (dev split)

| Task | Metric | n-gram | Best transformer (hybrid) |
|---|---|---:|---:|
| 1. Next-step | Top-1 | 0.687 | **0.75** |
| 1. Next-step | MRR | 0.807 | **0.8735** |
| 2. Completion | Token-acc | 0.421 | **0.455** |
| 3. Anomaly | F1 (invalid) | — | **1.00** (rule validator) |

Full per-run leaderboard in [`artifacts/sweeps/LEADERBOARD_FINAL.md`](artifacts/sweeps/LEADERBOARD_FINAL.md).

## Quickstart (≈5 min, CPU-only)

```bash
git clone -b wave1-submission https://github.com/beistvan/industrial-42-ai-crash.git
cd industrial-42-ai-crash

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch

python scripts/check_environment.py --require-torch
make dev-split           # builds data/processed/splits/ and dev_eval/
make train-ngram         # baseline; ~2 minutes
make smoke               # pytest + artifact validation (~1 min more)
# or: pytest -q          # 58 tests
make run-dashboard       # optional — metrics, leaderboard, eval matrix, live demo
```

The repo includes everything **except model checkpoints** (they're ~140 MB and
gitignored). To reproduce the transformer numbers, see **Reproducing the
submission** below.

## Reproducing the submission

Full retraining of Wave 1 + Wave 2 picks takes ~2–3 hours on a single A100 GPU.
Checkpoints already on disk? Regenerate judge CSVs:

```bash
make leaderboard-final      # if metrics JSONs changed
make regenerate-submission  # hybrid T1/T2 predict → result/submission/ (CUDA GPU)
```

Task 3 (`anomaly.csv`) reuses the **Task-1 checkpoint** for the LM `SCORE` column;
rule detection is fixed validator logic — no separate Task-3 model to retrain.

Train from scratch:

```bash
# 1. Synthetic augmentation (deterministic, 1x extras)
python scripts/generate_extra_sequences.py \
    --count-per-family 250 --seed 101 --force

# Task 1 leader (Wave 3 modern, row 4)
python scripts/sweep_transformer.py \
    --sweep configs/sweeps/leonardo_modern.yaml --stage finalists --row 4
# -> models/sweeps/h_mod_nosched_mrr.pt.best

# Task 2 leader (Wave 2 fine grid, row 4)
python scripts/sweep_transformer.py \
    --sweep configs/sweeps/leonardo_fine.yaml --stage finalists --row 4
# -> models/sweeps/g_drop15_nosched_t2.pt.best

# 3. Hybrid predictions on judge inputs
make regenerate-submission
# or: bash scripts/regenerate_submission.sh
```

Winning sweep rows: [`configs/sweeps/WINNING_RECIPES.md`](configs/sweeps/WINNING_RECIPES.md)  
Pipeline checklist: [`docs/ENGINEERING_PRACTICES.md`](docs/ENGINEERING_PRACTICES.md)

## Makefile shortcuts

| Target | Purpose |
|---|---|
| `make dev-split` | Build persisted dev holdout |
| `make train-ngram` | N-gram baseline + metrics JSON |
| `make smoke` | dev-split + n-gram + pytest + artifact check |
| `make validate-artifacts` | Schema checks on CSVs and metrics JSON |
| `make rehearsal-train` | Tiny Transformer train+eval before long GPU runs |
| `make eval-matrix` | 4-arm before/after comparison (dashboard tab) |
| `make leaderboard-final` | Refresh `LEADERBOARD_FINAL.{csv,md}` |
| `make regenerate-submission` | Pick T1/T2 + write judge CSVs (CUDA) |
| `make dashboard` | Unified Streamlit UI |

## What you need to actually run this

- **Python 3.10+**, ~2 GB disk
- **No API keys, no external services** — everything runs locally
- **PyTorch** — CPU for baseline/smoke; **CUDA GPU** for Transformer training and `make regenerate-submission`
- **A100 or equivalent** if you want to retrain both submission models in ≤3 hours
- Sweep YAMLs (`configs/sweeps/leonardo_*.yaml`) expand to CLI via `sweep_transformer.py --row N` — no cluster scripts required

## Honest limits

- **Task 1 plateau broken by Wave 3**: modern stack (`h_mod_nosched_mrr`) reaches MRR **0.8735** (+0.05pp vs Wave 1). Wave 4/5 did not beat submission picks.
- **Task 2** still leaves >50% of completion tokens incorrect — rule-constrained beam=5 helps but `token_accuracy` is structurally limited by the model's medium scale.
- **Task 3** uses rule validator for detection + **T1 LM log-prob SCORE** (`src/eval/anomaly_scoring.py`); F1=1.00 on dev because rules separate cleanly.
- **No pre-trained LLM** — see `docs/ADRs/0001-no-hf-pretrained.md` for why.
- **Pipeline workflow** — rehearsal gate, 4-arm eval matrix, artifact validation; see `docs/ENGINEERING_PRACTICES.md`.

## License

MIT. See `LICENSE` — the copyright holders are the three team members listed above.
Everything in this repo can be freely reused, modified, and redistributed.

## Credits

- **Organizers**: AI Factory Hackathon + Infineon for the track, the dataset, and the rule grammar.
- **Pre-vendored code**: `data/raw/infineon/training_data/generate_sequences.py` (the official validator and synthetic sequence generator) is from the organizers and remains under their license.
- **Libraries**: PyTorch, NumPy, pandas, scikit-learn, Streamlit, PyYAML, pytest, ruff.
- **Compute**: CINECA Leonardo (EuroHPC) was used during the hackathon for GPU sweeps; reproduction is plain Python + Makefile on any CUDA machine.
- **AI coding assistance**: Claude (architecture sounding-board + boilerplate generation). All experimental decisions and the final architecture are ours.
