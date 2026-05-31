# Industrial AI — Process Sequence Models (Infineon track)

Hackathon entry for the **42 AI Crash — Industrial AI** track.
We learn the *logic* of semiconductor fabrication process sequences from a small
real corpus (~3,000 sequences, 3 product families, ~140-token vocabulary) and
generate predictions for three tasks: next-step, sequence completion, and
anomaly detection against a 10-rule process grammar.

- **License**: MIT (see `LICENSE`)
- **Team**: Andrija Jovanovic, Istvan Beregszaszi, Thánh Trung Nguyen
- **Track**: Industrial AI (Infineon)
- **Report**: see [`REPORT.md`](REPORT.md)
- **Submission plan**: see [`docs/SUBMISSION.md`](docs/SUBMISSION.md)

## What's in this repo

```
src/, scripts/, tests/, configs/      Code + tests + sweep YAMLs
EVAL_DATA/                            Judge eval inputs + the official eval_metrics.py
result/submission/            THE submission (3 judge-format CSVs)
artifacts/sweeps/                     Per-run metrics + LEADERBOARD_FINAL
data/raw/infineon/                    Organizer-provided source data
data/generated/infineon/              Synthetic augmentation (deterministic, manifest.json)
docs/                                 ADRs, data spec, Leonardo GPU runbook, submission notes
slides/                               Pitch deck (PPTX; see slides/README.md for PDF)
REPORT.md, README.md, LICENSE         Required jury deliverables
requirements.txt, Makefile            Reproducibility
```

## What we built (one-line each)

- A **compact decoder-only Transformer** (~4M params, 6 layers, 8 heads, d_model=256), trained from scratch on the Infineon process step vocabulary.
- An **n-gram suffix-backoff baseline** for transparency and a fair comparison point.
- A **rule-validator + continuous SCORE** anomaly head reusing the official `validate_sequence` grammar.
- A **hybrid submission**: best Task-1 model for nextstep + best Task-2 specialist for completion + rule validator for anomaly.

## Headline numbers (dev split)

| Task | Metric | n-gram | Best transformer (hybrid) |
|---|---|---:|---:|
| 1. Next-step | Top-1 | 0.687 | **0.748** |
| 1. Next-step | MRR | 0.807 | **0.873** |
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
pytest -q                # ~33 tests, all should pass
```

The repo includes everything **except model checkpoints** (they're ~140 MB and
gitignored). To reproduce the transformer numbers, see **Reproducing the
submission** below.

## Reproducing the submission

Full retraining of Wave 1 + Wave 2 picks takes ~2–3 hours on a single A100 GPU.
Checkpoints already on disk? Regenerate judge CSVs in one step:

```bash
make leonardo-leaderboard-final   # if metrics JSONs changed
make regenerate-submission        # picks best T1/T2; Slurm GPU predict on login node
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

# 4. Hybrid predictions on judge inputs
bash scripts/regenerate_submission.sh
```

On Leonardo (or any Slurm A100 cluster), `make leonardo-wave1` submits the full
6-finalist sweep array directly. See `docs/LEONARDO_GPU_RUNBOOK.md`.

## What you need to actually run this

- **Python 3.10+**, ~2 GB disk
- **No API keys, no external services** — everything runs locally
- **PyTorch** (CPU is fine for inference + baseline; GPU/CUDA 12.1 recommended for training)
- **A100 or equivalent** if you want to retrain in ≤2 hours; CPU works but slow (~12 h)
- **Leonardo access is *not* required** to reproduce — the `configs/sweeps/leonardo_*.yaml` files run on any Slurm cluster or as plain CLI rows via `--row N`

## Honest limits

- **Task 1 plateau ≈ MRR 0.87**: Wave 1–2 hyperparam sweeps land within ~2pp on Task 1. Wave 2 improved Task 2 (+0.3 pp tok-acc). Further Task-1 gains likely need architecture change (Wave 3) or more data.
- **Task 2** still leaves >50% of completion tokens incorrect — rule-constrained beam=5 helps but `token_accuracy` is structurally limited by the model's medium scale.
- **Task 3 SCORE is heuristic** (`1 − n_violations/10`); on the judge eval all invalid sequences contain exactly 1 violation, so AUC = 1.0 by construction.
- **No pre-trained LLM** — see `docs/ADRs/0001-no-hf-pretrained.md` for why.

## License

MIT. See `LICENSE` — the copyright holders are the three team members listed above.
Everything in this repo can be freely reused, modified, and redistributed.

## Credits

- **Organizers**: 42 AI Crash + Infineon for the track, the dataset, and the rule grammar.
- **Pre-vendored code**: `data/raw/infineon/training_data/generate_sequences.py` (the official validator and synthetic sequence generator) is from the organizers and remains under their license.
- **Libraries**: PyTorch, NumPy, pandas, scikit-learn, Streamlit, PyYAML, pytest, ruff.
- **Compute**: CINECA Leonardo (EuroHPC) for GPU sweep, reservation `s_tra_ncc`, account `EUHPC_D30_031`.
- **AI coding assistance**: GitHub Copilot, Claude (architecture sounding-board + boilerplate generation). All experimental decisions and the final architecture are ours.
