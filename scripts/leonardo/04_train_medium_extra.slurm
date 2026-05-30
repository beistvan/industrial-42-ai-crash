#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --job-name=inf-medium-extra
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models artifacts data/generated/infineon

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

python scripts/leonardo/check_gpu.py
python scripts/make_dev_split.py --force
if [ ! -f data/generated/infineon/manifest.json ]; then
  python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force
fi
python scripts/train_transformer.py \
  --config configs/transformer_medium.yaml \
  --device cuda \
  --amp \
  --extra-data-dir data/generated/infineon \
  --eval-task1-every 1 \
  --eval-task2-every 0 \
  --skip-eval \
  --save-best-by dev_mrr \
  --model-path models/transformer_medium_extra.pt \
  --metrics-path artifacts/transformer_medium_extra_metrics.json \
  --run-name transformer_medium_extra
