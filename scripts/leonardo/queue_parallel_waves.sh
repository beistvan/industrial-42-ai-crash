#!/bin/bash
# Fire waves 3 + 4 + 6 in parallel, then start leaderboard watcher.
#
# Usage:
#   SWEEP_CONCURRENCY=32 bash scripts/leonardo/queue_parallel_waves.sh
#   SWEEP_CONCURRENCY=32 bash scripts/leonardo/queue_parallel_waves.sh --with-wave5
set -euo pipefail

REPO="$PWD"
WITH_W5=0
FORCE_W5=0
for arg in "$@"; do
    case "$arg" in
        --with-wave5) WITH_W5=1 ;;
        --force-wave5) WITH_W5=1; FORCE_W5=1 ;;
    esac
done
cd "$REPO"

module purge 2>/dev/null || true
module load python/3.11.7 2>/dev/null || true
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

mkdir -p logs
LOG="logs/queue_parallel.log"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

submit() {
    local yaml="$1"
    local label="$2"
    local jid
    jid=$(bash scripts/leonardo/submit_sweep.sh "$yaml" finalists)
    log "${label}: job ${jid}"
}

log "=== parallel queue (SWEEP_CONCURRENCY=${SWEEP_CONCURRENCY:-32}) ==="
submit configs/sweeps/leonardo_modern.yaml "Wave 3 modern"
submit configs/sweeps/leonardo_task2.yaml "Wave 4 Task-2"
submit configs/sweeps/leonardo_scale.yaml "Wave 6 scale/seeds"

if [[ "$WITH_W5" == "1" ]]; then
    if [[ "${FORCE_W5:-0}" == "1" ]]; then
        submit configs/sweeps/leonardo_params.yaml "Wave 5 params (forced)"
    else
        bash scripts/leonardo/wave5_if_needed.sh 2>&1 | tee -a "$LOG" || true
    fi
fi

if ! pgrep -f "pipeline_watch.sh" >/dev/null 2>&1; then
    nohup bash scripts/leonardo/pipeline_watch.sh >> logs/pipeline-watch.out 2>&1 &
    log "started pipeline_watch (pid $!) — tail -f logs/pipeline_watch.log"
else
    log "pipeline_watch already running"
fi

log "Monitor: squeue --me | grep zoh-sweep"
log "Leaderboard: make leonardo-leaderboard-final"
