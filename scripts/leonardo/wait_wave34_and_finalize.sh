#!/bin/bash
# Wait for Wave 3/4 arrays to finish, then rebuild leaderboard + regenerate submission.
set -euo pipefail
cd /leonardo_scratch/large/usertrain/a08trd0x/industrial-42-ai-crash

LOG="logs/wave34_finalize.log"
mkdir -p logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

module purge 2>/dev/null || true
module load python/3.11.7 2>/dev/null || true
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

log "Waiting for zoh-sweep jobs to finish..."
while true; do
    if ! out=$(squeue --me -h -o "%j" 2>&1); then
        log "WARN: squeue failed (${out}); retry in 30s"
        sleep 30
        continue
    fi
    if ! grep -q zoh-sweep <<<"$out"; then
        break
    fi
    n=$(grep -c zoh-sweep <<<"$out" || true)
    log "  still running: ${n} zoh-sweep tasks"
    sleep 120
done

log "All zoh-sweep jobs done."

log "=== make leonardo-leaderboard-final ==="
make leonardo-leaderboard-final 2>&1 | tee -a "$LOG"

log "=== make regenerate-submission (Slurm predict on login node) ==="
USE_SLURM=1 bash scripts/regenerate_submission.sh 2>&1 | tee -a "$LOG"

log "=== Done ==="
python3 - <<'PY' | tee -a "$LOG"
import sys
sys.path.insert(0, ".")
from scripts.sweep_picks import load_leaderboard_rows, pick_task1_row, pick_task2_row, score_task1, score_task2
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
t1 = pick_task1_row(rows)
t2 = pick_task2_row(rows)
print(f"Submission picks: T1={t1['run']} MRR={score_task1(t1):.4f}  T2={t2['run']} tok={score_task2(t2):.4f}")
PY
