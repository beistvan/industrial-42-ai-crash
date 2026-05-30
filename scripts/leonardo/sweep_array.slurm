#!/bin/bash
# Array-job sweep on Leonardo. Each array task runs one row of the sweep YAML.
#
# Submit:
#   # Find out how many rows the stage has (rows are 0-indexed):
#   python3 scripts/sweep_transformer.py \
#       --sweep configs/sweeps/leonardo_v1.yaml --stage shortlist --dry-run | grep -c "^\["
#   # Suppose it prints 12 — submit 0..11:
#   sbatch --array=0-11 scripts/leonardo/sweep_array.slurm \
#       configs/sweeps/leonardo_v1.yaml shortlist
#
# Each task = 1 GPU = ~1.5h for a 10-epoch medium run on 3k real + 750 synth.
# Default reservation gives ~4 simultaneous tasks (1 node × 4 GPUs).

#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --job-name=zoh-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-sweep-%A_%a.out

set -euo pipefail

SWEEP="${1:-configs/sweeps/leonardo_v1.yaml}"
STAGE="${2:-shortlist}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/sweeps artifacts/sweeps

RUN="${RUN:-python3}"

echo "[$(date -Is)] Sweep ${SWEEP} stage=${STAGE} row=${SLURM_ARRAY_TASK_ID}"
$RUN scripts/sweep_transformer.py \
    --sweep "$SWEEP" \
    --stage "$STAGE" \
    --row "$SLURM_ARRAY_TASK_ID"

echo "[$(date -Is)] Sweep row ${SLURM_ARRAY_TASK_ID} complete."
