#!/bin/bash
# Full pipeline: wait Wave 2 → submit Wave 3 → wait → submit Wave 4 → finalize.
set -euo pipefail

WAVE2_JOB="${1:?usage: wave_orchestrator.sh WAVE2_JOB_ID}"
REPO="${2:-$PWD}"
cd "$REPO"

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

SLURM_ACCOUNT="${SLURM_ACCOUNT:-EUHPC_D30_031}"
LOG="logs/wave_orchestrator.log"
mkdir -p logs models/sweeps artifacts/sweeps result/submission

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

wait_array() {
    local jid="$1"
    local label="$2"
    log "Waiting for ${label} (job ${jid})..."
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do
        local running completed
        running=$(squeue -j "$jid" -h 2>/dev/null | wc -l)
        completed=$(find artifacts/sweeps -maxdepth 1 -name '*.json' -newer "$LOG" 2>/dev/null | wc -l || true)
        log "${label}: ${running} tasks still running"
        sleep 120
    done
    log "${label} finished."
}

submit_sweep() {
    local yaml="$1"
    local stage="$2"
    local label="$3"
    local job_id
    log "Submitting ${label} from ${yaml}" >&2
    job_id=$(bash scripts/leonardo/submit_sweep.sh "$yaml" "$stage")
    log "  -> job ${job_id}" >&2
    echo "$job_id"
}

rebuild_leaderboard() {
    python3 scripts/summarize_runs.py \
        --metrics-dir artifacts/sweeps \
        --out artifacts/sweeps/LEADERBOARD_FINAL.md \
        --csv artifacts/sweeps/LEADERBOARD_FINAL.csv
    log "Leaderboard updated."
}

generate_submission() {
    local t1 t2
    read -r t1 t2 <<<"$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.sweep_picks import load_leaderboard_rows, pick_task1_run, pick_task2_run
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
print(pick_task1_run(rows), pick_task2_run(rows))
PY
)"
    log "Submission: T1=${t1} T2=${t2}"
    MODEL="models/sweeps/${t1}.pt.best" sbatch --account="$SLURM_ACCOUNT" \
        scripts/leonardo/05_predict_transformer.slurm
    mkdir -p result/.staging_t2
    MODEL="models/sweeps/${t2}.pt.best" OUT_DIR=result/.staging_t2 ANOM="" \
        sbatch --account="$SLURM_ACCOUNT" scripts/leonardo/05_predict_transformer.slurm

    log "Waiting for predict jobs..."
    sleep 30
    while squeue --me -h 2>/dev/null | grep -q inf-predict; do
        log "predict jobs still running..."
        sleep 60
    done
    if [[ -f result/.staging_t2/completion.csv ]]; then
        cp result/.staging_t2/completion.csv result/submission/completion.csv
        log "Merged Task-2 completion into result/submission/"
    else
        log "WARN: result/.staging_t2/completion.csv missing"
    fi
}

log "=== Wave orchestrator start (Wave2=${WAVE2_JOB}) ==="
wait_array "$WAVE2_JOB" "Wave 2"
rebuild_leaderboard

WAVE3_JOB=$(submit_sweep configs/sweeps/leonardo_modern.yaml finalists "Wave 3")
wait_array "$WAVE3_JOB" "Wave 3"
rebuild_leaderboard

WAVE4_JOB=$(submit_sweep configs/sweeps/leonardo_task2.yaml finalists "Wave 4")
wait_array "$WAVE4_JOB" "Wave 4"
rebuild_leaderboard

WAVE6_JOB=$(submit_sweep configs/sweeps/leonardo_scale.yaml finalists "Wave 6")
wait_array "$WAVE6_JOB" "Wave 6"
rebuild_leaderboard

generate_submission
log "=== Pipeline complete ==="

python3 - <<'PY' | tee -a "$LOG"
import sys
sys.path.insert(0, ".")
import csv
from scripts.sweep_picks import (
    load_leaderboard_rows, pick_task1_row, pick_task2_row,
    score_task1, score_task2,
)
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
w3 = [r for r in rows if r["run"].startswith("h_mod_")]
w4 = [r for r in rows if r["run"].startswith("t2_mod_")]
print("\n=== Wave 3 top (MRR) ===")
for r in sorted(w3, key=score_task1, reverse=True)[:4]:
    print(f"  {r['run']:22s} MRR={score_task1(r):.4f} tok={score_task2(r):.4f}")
print("\n=== Wave 4 top (tok acc) ===")
for r in sorted(w4, key=score_task2, reverse=True)[:4]:
    print(f"  {r['run']:22s} MRR={score_task1(r):.4f} tok={score_task2(r):.4f}")
t1 = pick_task1_row(rows)
t2 = pick_task2_row(rows)
print(f"\nOverall T1: {t1['run']} MRR={score_task1(t1):.4f}")
print(f"Overall T2: {t2['run']} tok={score_task2(t2):.4f}")
PY
