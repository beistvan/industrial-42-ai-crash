#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=inf-gen-10k
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/generated/infineon

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

python scripts/make_dev_split.py --force
python scripts/generate_extra_sequences.py \
  --count-per-family "${EXTRA_COUNT_PER_FAMILY:-10000}" \
  --seed "${EXTRA_SEED:-101}" \
  --force
