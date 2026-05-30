#!/bin/bash
# Train one Transformer config on Leonardo. Single GPU.
#
# Submit:
#   sbatch scripts/leonardo/train_one.slurm \
#     configs/transformer_medium.yaml \
#     m_real_extras_1x
#
# Args: $1 = base config YAML, $2 = run_name (used for paths).

#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --job-name=zoh-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

CONFIG="${1:-configs/transformer_medium.yaml}"
RUN_NAME="${2:-medium_default}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/sweeps artifacts/sweeps

if [[ -f scripts/leonardo/load_env.sh ]]; then
    # shellcheck source=scripts/leonardo/load_env.sh
    source scripts/leonardo/load_env.sh
fi

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Pixi-style env activation. Adjust if your team uses Singularity or conda.
# Example singularity invocation:
#   export RUN="singularity exec --nv --bind \$SCRATCH:/scratch pytorch.sif"
#   $RUN python3 scripts/train_transformer.py ...
RUN="${RUN:-python3}"

# Wire proxy if compute nodes need internet for first-time HF caches etc.
# (per Z10 deck p.95). Comment out if not needed.
# export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
# export HTTPS_PROXY=$HTTP_PROXY

echo "[$(date -Is)] Starting run=${RUN_NAME} config=${CONFIG}"
$RUN scripts/train_transformer.py \
  --config "$CONFIG" \
  --device cuda \
  --extra-data-dir data/generated/infineon \
  --amp \
  --scheduler cosine \
  --warmup-steps 500 \
  --label-smoothing 0.1 \
  --save-best-by dev_mrr \
  --eval-task1-every 1 \
  --eval-task2-every 0 \
  --run-name "$RUN_NAME" \
  --model-path "models/sweeps/${RUN_NAME}.pt" \
  --metrics-path "artifacts/sweeps/${RUN_NAME}.json"

echo "[$(date -Is)] Done."
