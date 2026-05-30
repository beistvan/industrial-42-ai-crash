#!/bin/bash
# Submit the full HPC battle as a Slurm dependency chain.
#
# Usage:
#   bash scripts/leonardo/run_battle.sh
#   bash scripts/leonardo/run_battle.sh configs/sweeps/leonardo_v1.yaml
#
# Stage 1 (shortlist) runs as an array with concurrency=4.
# Stage 2 is a tiny job that runs summarize_runs.py the moment stage 1 finishes,
# then prints the next command for you to type after reviewing the leaderboard.
# Stage 3 (finalists) is intentionally NOT auto-submitted — you decide based on
# the leaderboard whether to promote, re-seed, or skip ahead to submission.

set -euo pipefail

SWEEP="${1:-configs/sweeps/leonardo_v1.yaml}"
STAGE="${2:-shortlist}"

if [[ ! -f "$SWEEP" ]]; then
    echo "ERROR: sweep file not found: $SWEEP" >&2
    exit 1
fi

# Count rows from the dry-run output (one '[' marker per row).
N=$(python3 scripts/sweep_transformer.py --sweep "$SWEEP" --stage "$STAGE" --dry-run 2>/dev/null | grep -c '^\[')
if [[ "$N" -lt 1 ]]; then
    echo "ERROR: stage '$STAGE' has 0 rows in $SWEEP" >&2
    exit 1
fi
LAST=$((N - 1))

# Hackathon EuroHPC account (from `saldo -b`). Override: export SLURM_ACCOUNT=...
ACCOUNT="${SLURM_ACCOUNT:-EUHPC_D30_031}"

echo "Submitting stage '$STAGE' as Slurm array 0-$LAST (concurrency 4, account=$ACCOUNT)..."
SWEEP_OUT=$(sbatch --parsable --account="$ACCOUNT" --array=0-$LAST%4 \
    scripts/leonardo/sweep_array.slurm "$SWEEP" "$STAGE")
SWEEP_JOB=${SWEEP_OUT%%;*}
echo "  array job id: $SWEEP_JOB"

echo "Chaining post-sweep leaderboard..."
SUM_JOB=$(sbatch --parsable --account="$ACCOUNT" \
    --partition=lrd_all_serial \
    --time=00:15:00 --mem=4G --cpus-per-task=1 \
    --dependency=afterok:$SWEEP_JOB \
    --output=logs/slurm-leaderboard-%j.out \
    --wrap "set -e; cd \$SLURM_SUBMIT_DIR; \
            python3 scripts/summarize_runs.py \
              --metrics-dir artifacts/sweeps \
              --out artifacts/sweeps/LEADERBOARD.md \
              --csv artifacts/sweeps/LEADERBOARD.csv; \
            echo; \
            echo '==================== NEXT STEPS ===================='; \
            echo 'Review the leaderboard:'; \
            echo '  head -30 artifacts/sweeps/LEADERBOARD.md'; \
            echo; \
            echo 'If a finalist run is warranted, edit the finalists:'; \
            echo 'block of $SWEEP with the top run_names (epochs:50,'; \
            echo 'eval_task2_every:10) and submit:'; \
            echo '  bash scripts/leonardo/run_battle.sh $SWEEP finalists'; \
            echo; \
            echo 'Otherwise jump straight to submission using the best'; \
            echo 'shortlist checkpoint (sorted in LEADERBOARD.csv):'; \
            echo '  python3 scripts/predict_submission.py \\'; \
            echo '    --model models/sweeps/<run_name>.pt.best \\'; \
            echo '    --eval-valid <organizer_valid.csv> \\'; \
            echo '    --eval-anomaly <organizer_anomaly.csv> \\'; \
            echo '    --out-dir extras/results_submission \\'; \
            echo '    --rule-constrained --candidate-pool 5'; \
            echo '===================================================='")
echo "  leaderboard job: $SUM_JOB (waits for $SWEEP_JOB)"

echo ""
echo "Submitted. To watch:"
echo "  squeue --me"
echo "  tail -f logs/slurm-sweep-${SWEEP_JOB}_*.out"
echo "  # After array finishes:"
echo "  cat logs/slurm-leaderboard-${SUM_JOB}.out"
