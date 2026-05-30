#!/bin/bash
# Prepare the Python environment on a Leonardo login node.
# Use this after `git clone` / `git pull`, not inside a GPU Slurm job.
set -euo pipefail

REPO_DIR="${1:-$PWD}"
cd "$REPO_DIR"

module purge || true
module load python/3.11.7 || true

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python scripts/check_environment.py --require-torch

# These are short/light enough for login-node preparation in this repo. If your
# session warns about CPU limits, run them via lrd_all_serial instead.
python scripts/make_dev_split.py --force
python scripts/generate_extra_sequences.py --count-per-family "${EXTRA_COUNT_PER_FAMILY:-250}" --seed 101 --force

cat <<'EOF'
Leonardo env prepared.
Next GPU smoke:
  sbatch scripts/leonardo/01_gpu_smoke.slurm
Then:
  sbatch scripts/leonardo/02_train_small.slurm
  sbatch scripts/leonardo/03_train_small_extra.slurm
EOF
