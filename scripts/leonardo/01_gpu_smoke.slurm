#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --job-name=inf-gpu-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models artifacts

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

python scripts/leonardo/check_gpu.py
python scripts/make_dev_split.py --force
python scripts/train_transformer.py \
  --config configs/transformer_small.yaml \
  --device cuda \
  --amp \
  --limit-train-sequences 4 \
  --limit-dev-sequences 2 \
  --epochs 1 \
  --skip-eval \
  --d-model 32 \
  --n-layers 1 \
  --n-heads 2 \
  --dim-feedforward 64 \
  --batch-size 4 \
  --model-path models/transformer_gpu_smoke.pt \
  --metrics-path artifacts/transformer_gpu_smoke_metrics.json \
  --run-name transformer_gpu_smoke
