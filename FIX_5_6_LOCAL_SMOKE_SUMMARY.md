# Fix summary — Step 5/6 local smoke readiness

This update keeps the current repo as the source of truth and fixes the issues
seen in the latest local run logs.

## Fixed

- Streamlit deprecation warnings in `src/app/main.py`:
  - replaced `use_container_width=True` with `width="stretch"`.
- PyTorch dependency handling:
  - added `scripts/check_environment.py`;
  - added `make setup-cpu`, `make check-torch`, and `make check-app`;
  - Transformer targets now fail early with a clear CPU-PyTorch install hint.
- Tests no longer hide missing PyTorch behind skips:
  - `tests/test_transformer_model.py` now fails clearly if PyTorch is missing;
  - added `make test-strict` / `scripts/run_tests_strict.py` for final no-skip verification.
- Makefile smoke stability:
  - `make test` disables third-party pytest plugin autoloading to avoid unrelated plugin hangs;
  - `make smoke` now runs dev split, n-gram training, pytest, and the smoke report.
- Step 6 local augmentation smoke:
  - `train_transformer.py` now supports `--limit-extra-sequences`;
  - added `make train-transformer-extra-smoke` so generated-data wiring can be checked quickly on CPU.
- Metrics interpretation:
  - added `docs/METRICS_INTERPRETATION.md`;
  - `scripts/smoke_test.py` now explains why Task 3 F1=1.000 is expected on injected local anomalies.

## Verified commands in this environment

```bash
make smoke PYTHON=python
make test-strict PYTHON=python
make train-transformer-smoke PYTHON=python
make generate-extra-local PYTHON=python
make train-ngram-extra PYTHON=python
make train-transformer-extra-smoke PYTHON=python
```

Use plain `make ...` locally if your `python3` points to the project venv.

## Important local setup note

The compact Transformer is GPU-free locally, but it is **not** PyTorch-free.
Before running Transformer commands locally:

```bash
make setup-cpu
make train-transformer-smoke
```

On Leonardo, install the CUDA-enabled PyTorch build according to the HPC docs,
then reuse the same `scripts/train_transformer.py` commands with `--device cuda`.
