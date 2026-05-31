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
Makefile                              dev-split, test, leaderboard, regen, dashboard
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

## Quickstart (CPU verify, ~5 min)

Confirms the repo installs, builds the dev split, and passes tests. No GPU training
in this path — submission checkpoints are gitignored (~43 MB); see **Reproducing
the submission** below.

```bash
git clone https://github.com/beistvan/industrial-42-ai-crash.git
cd industrial-42-ai-crash

python -m venv .venv && source .venv/bin/activate
make setup-cpu
make dev-split              # data/processed/splits/ + dev_eval/
make test                   # 58 pytest tests
make validate-artifacts     # schema checks on dev CSVs + sweep metrics JSON
make dashboard              # optional — metrics, leaderboard, eval matrix, live demo
```

Optional baseline (not required for the Transformer submission):

```bash
make train-ngram            # artifacts/ngram_metrics.json + models/ngram_baseline.pkl
make smoke                  # dev-split + n-gram + pytest + artifact check
```

## Reproducing the submission

**Hybrid picks (frozen):** `h_mod_nosched_mrr` (Task 1 + Task 3 SCORE) +
`g_drop15_nosched_t2` (Task 2). Recipes: [`configs/sweeps/WINNING_RECIPES.md`](configs/sweeps/WINNING_RECIPES.md).

Checkpoints already on disk? Regenerate judge CSVs (CUDA GPU):

```bash
make leaderboard-final      # if metrics JSONs changed
make regenerate-submission  # → result/submission/{nextstep,completion,anomaly}.csv
```

Task 3 (`anomaly.csv`) reuses the **Task-1 checkpoint** for the LM `SCORE` column;
rule detection is fixed validator logic — no separate Task-3 model to retrain.

### Train from scratch (~2–3 h on one A100)

```bash
# 1. Data prep (once)
make dev-split
make generate-extra-local   # +250 valid sequences/family → data/generated/infineon/

# 2. Task 1 — modern stack (base: configs/transformer_modern.yaml)
#    Sweep row 4 → h_mod_nosched_mrr
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_modern.yaml --stage finalists --row 4
# → models/sweeps/h_mod_nosched_mrr.pt.best

# 3. Task 2 — vanilla stack (base: configs/transformer_medium.yaml)
#    Sweep row 4 → g_drop15_nosched_t2
python scripts/sweep_transformer.py \
  --sweep configs/sweeps/leonardo_fine.yaml --stage finalists --row 4
# → models/sweeps/g_drop15_nosched_t2.pt.best

# 4. Hybrid judge CSVs
make leaderboard-final
make regenerate-submission
```

Each `sweep_transformer.py` row expands to `scripts/train_transformer.py` with
the merged YAML knobs (`base_config`, dropout, scheduler, `save_best_by`, etc.).

Pipeline checklist: [`docs/ENGINEERING_PRACTICES.md`](docs/ENGINEERING_PRACTICES.md)

## Makefile shortcuts

| Target | Purpose |
|---|---|
| `make setup-cpu` | venv deps + CPU PyTorch |
| `make dev-split` | Build persisted dev holdout |
| `make test` | Run pytest (58 tests) |
| `make validate-artifacts` | Schema checks on CSVs and metrics JSON |
| `make generate-extra-local` | Synthetic augmentation (250/family) |
| `make rehearsal-train` | Tiny Transformer train+eval before long GPU runs |
| `make eval-matrix` | 4-arm before/after comparison (dashboard tab) |
| `make leaderboard-final` | Refresh `LEADERBOARD_FINAL.{csv,md}` |
| `make regenerate-submission` | Pick T1/T2 + write judge CSVs (CUDA) |
| `make dashboard` | Unified Streamlit UI |
| `make train-ngram` | Optional n-gram baseline |
| `make smoke` | Optional full CPU pipeline incl. n-gram |

## What you need to actually run this

- **Python 3.10+**, ~2 GB disk
- **No API keys, no external services** — everything runs locally
- **PyTorch** — CPU for quickstart/tests; **CUDA GPU** for Transformer training and `make regenerate-submission`
- **A100 or equivalent** if you want to retrain both submission models in ≤3 hours
- Sweep YAMLs (`configs/sweeps/leonardo_*.yaml`) expand to CLI via `sweep_transformer.py --row N`

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
