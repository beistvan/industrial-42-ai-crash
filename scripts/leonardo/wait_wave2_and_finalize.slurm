#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=EUHPC_D30_031
#SBATCH --job-name=wave2-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/wave2-finalize-%j.out
#SBATCH --error=logs/wave2-finalize-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/leonardo/wait_wave2_and_finalize.sh "${WAVE2_JOB_ID:?set WAVE2_JOB_ID}"
