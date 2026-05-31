#!/bin/bash
# Submit one sweep YAML as a Slurm array job.
#
# Usage:
#   bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_modern.yaml finalists
#
# Env:
#   SWEEP_CONCURRENCY  max simultaneous array tasks (default 32)
#   SLURM_ACCOUNT      CINECA account (default EUHPC_D30_031)
#   SLURM_RESERVATION  optional reservation name (unset = no reservation)
set -euo pipefail

YAML="${1:?usage: submit_sweep.sh SWEEP.yaml [STAGE]}"
STAGE="${2:-finalists}"
REPO="${REPO:-$PWD}"
cd "$REPO"

CONCURRENCY="${SWEEP_CONCURRENCY:-32}"
ACCOUNT="${SLURM_ACCOUNT:-EUHPC_D30_031}"

module purge 2>/dev/null || true
module load python/3.11.7 2>/dev/null || true
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

N=$(python3 scripts/sweep_transformer.py --sweep "$YAML" --stage "$STAGE" --dry-run | grep -c '^\[')
if [[ "$N" -lt 1 ]]; then
    echo "No rows in ${YAML} stage=${STAGE}" >&2
    exit 1
fi

mkdir -p logs
SBATCH_ARGS=(
    --account="$ACCOUNT"
    --array="0-$((N - 1))%${CONCURRENCY}"
    --job-name="zoh-sweep"
)
if [[ -n "${SLURM_RESERVATION:-}" ]]; then
    SBATCH_ARGS+=(--reservation="$SLURM_RESERVATION")
fi

echo "Submitting ${YAML} (${STAGE}): ${N} rows, concurrency=${CONCURRENCY}, account=${ACCOUNT}"
JOB_OUT=$(sbatch "${SBATCH_ARGS[@]}" scripts/leonardo/sweep_array.slurm "$YAML" "$STAGE")
echo "$JOB_OUT"
JOB_ID=$(echo "$JOB_OUT" | awk '{print $NF}')
echo "$JOB_ID" >> logs/submitted_jobs.txt
echo "$JOB_ID"
