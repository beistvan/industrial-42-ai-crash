#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
# Reservation optional: sbatch --reservation=s_tra_ncc ...
#SBATCH --job-name=inf-predict-tr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs result/submission

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

MODEL="${MODEL:-models/sweeps/f_drop15_100_mrr.pt.best}"
OUT_DIR="${OUT_DIR:-result/submission}"
VALID="${VALID:-EVAL_DATA/eval_input_valid.csv}"
ANOM="${ANOM:-EVAL_DATA/eval_input_anomaly.csv}"
BEAM_WIDTH="${BEAM_WIDTH:-5}"
CANDIDATE_POOL="${CANDIDATE_POOL:-5}"

python scripts/leonardo/check_gpu.py

PREDICT_ARGS=(
  --model "$MODEL"
  --device cuda
  --eval-valid "$VALID"
  --out-dir "$OUT_DIR"
  --rule-constrained
  --beam-width "$BEAM_WIDTH"
  --candidate-pool "$CANDIDATE_POOL"
)
if [[ -n "${ANOM:-}" ]]; then
  PREDICT_ARGS+=(--eval-anomaly "$ANOM")
fi

python scripts/predict_submission.py "${PREDICT_ARGS[@]}"
