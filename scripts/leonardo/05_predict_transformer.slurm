#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
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
mkdir -p logs extras/results_transformer

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

MODEL="${MODEL:-models/transformer_small_extra.pt}"
OUT_DIR="${OUT_DIR:-extras/results_transformer}"
VALID="${VALID:-data/processed/dev_eval/eval_input_valid_dev.csv}"
ANOM="${ANOM:-data/processed/dev_eval/eval_input_anomaly_dev.csv}"

python scripts/leonardo/check_gpu.py
python scripts/predict_submission.py \
  --model "$MODEL" \
  --device cuda \
  --eval-valid "$VALID" \
  --eval-anomaly "$ANOM" \
  --out-dir "$OUT_DIR" \
  --rule-constrained \
  --candidate-pool 5
