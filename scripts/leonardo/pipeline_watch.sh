#!/bin/bash
# Background watcher: refresh leaderboard while sweeps run; finalize when idle.
#
# Usage:
#   nohup bash scripts/leonardo/pipeline_watch.sh >> logs/pipeline-watch.out 2>&1 &
#
# Env:
#   WATCH_INTERVAL       seconds between polls (default 300)
#   AUTO_LEADERBOARD     rebuild when metrics json count changes (default 1)
#   AUTO_SUBMIT          regenerate judge CSVs when idle + new best (default 0)
#   IDLE_POLLS_FOR_DONE  consecutive idle polls before finalize (default 2)
set -u

REPO="${1:-$PWD}"
cd "$REPO"

INTERVAL="${WATCH_INTERVAL:-300}"
AUTO_LB="${AUTO_LEADERBOARD:-1}"
AUTO_SUB="${AUTO_SUBMIT:-0}"
IDLE_NEEDED="${IDLE_POLLS_FOR_DONE:-2}"

module purge 2>/dev/null || true
module load python/3.11.7 2>/dev/null || true
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

LOG="logs/pipeline_watch.log"
mkdir -p logs artifacts/sweeps result/submission

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

last_json_count=-1
idle_polls=0

rebuild_leaderboard() {
    python3 scripts/summarize_runs.py \
        --metrics-dir artifacts/sweeps \
        --out artifacts/sweeps/LEADERBOARD_FINAL.md \
        --csv artifacts/sweeps/LEADERBOARD_FINAL.csv
    python3 - <<'PY' | tee -a "$LOG"
import sys
sys.path.insert(0, ".")
from scripts.sweep_picks import (
    load_leaderboard_rows, pick_task1_row, pick_task2_row,
    score_task1, score_task2,
)
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
t1 = pick_task1_row(rows)
t2 = pick_task2_row(rows)
print(f"  runs={len(rows)}  T1={t1['run']} MRR={score_task1(t1):.4f}  "
      f"T2={t2['run']} tok={score_task2(t2):.4f}")
PY
}

log "pipeline_watch started (interval=${INTERVAL}s, concurrency hint: SWEEP_CONCURRENCY=${SWEEP_CONCURRENCY:-32})"

while true; do
    running=$(squeue --me -h -o "%j" 2>/dev/null | grep -c zoh-sweep || true)
    n_json=$(find artifacts/sweeps -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    n_best=$(find models/sweeps -maxdepth 1 -name '*.pt.best' 2>/dev/null | wc -l)

    log "running=${running} metrics_json=${n_json} checkpoints=${n_best}"

    if [[ "$AUTO_LB" == "1" && "$n_json" != "$last_json_count" && "$n_json" -gt 0 ]]; then
        log "metrics changed (${last_json_count} -> ${n_json}) — rebuilding leaderboard"
        rebuild_leaderboard || log "WARN: leaderboard rebuild failed"
        last_json_count=$n_json
    fi

    if [[ "$running" -eq 0 ]]; then
        idle_polls=$((idle_polls + 1))
        log "no zoh-sweep jobs (${idle_polls}/${IDLE_NEEDED} idle polls)"
        if [[ "$idle_polls" -ge "$IDLE_NEEDED" ]]; then
            log "sweep queue idle — final leaderboard"
            rebuild_leaderboard || true
    if [[ "$AUTO_SUB" == "1" ]]; then
                log "AUTO_SUBMIT=1 — regenerating submission CSVs (Slurm)"
                USE_SLURM=1 bash scripts/regenerate_submission.sh 2>&1 | tee -a "$LOG" || log "WARN: regenerate failed"
            fi
            log "pipeline_watch done (idle finalize)"
            exit 0
        fi
    else
        idle_polls=0
        squeue --me -o "%.10i %.12j %.2t %.10M" 2>/dev/null | head -20 | tee -a "$LOG" || true
    fi

    sleep "$INTERVAL"
done
