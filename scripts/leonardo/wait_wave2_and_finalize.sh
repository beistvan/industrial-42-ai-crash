#!/bin/bash
# Wait for Wave 2 array job, rebuild leaderboard, regenerate submission CSVs.
set -euo pipefail

JOB_ID="${1:?usage: wait_wave2_and_finalize.sh JOB_ID}"
REPO="${2:-$PWD}"
cd "$REPO"

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

LOG="logs/wave2_finalize-${JOB_ID}.log"
mkdir -p logs result/submission

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "Waiting for array job ${JOB_ID} to finish..."
while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q .; do
    running=$(squeue -j "$JOB_ID" -h 2>/dev/null | wc -l)
    completed=$(find artifacts/sweeps -maxdepth 1 -name 'g_*.json' 2>/dev/null | wc -l)
    log "still running: ${running} tasks, metrics jsons: ${completed}/5"
    sleep 120
done

log "Array job ${JOB_ID} finished. Checking exit codes..."
fail=0
for i in 0 1 2 3 4; do
    err="logs/slurm-sweep-${JOB_ID}_${i}.err"
    out="logs/slurm-sweep-${JOB_ID}_${i}.out"
    if [[ -s "$err" ]]; then
        log "WARN task ${i} stderr:"
        tail -5 "$err" | tee -a "$LOG"
        fail=1
    fi
    if ! grep -q "Sweep row ${i} complete" "$out" 2>/dev/null; then
        log "WARN task ${i} may not have completed cleanly (check ${out})"
        fail=1
    fi
done

log "Rebuilding leaderboard..."
python scripts/summarize_runs.py \
    --metrics-dir artifacts/sweeps \
    --out artifacts/sweeps/LEADERBOARD_FINAL.md \
    --csv artifacts/sweeps/LEADERBOARD_FINAL.csv

read -r T1_RUN T2_RUN <<<"$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.sweep_picks import load_leaderboard_rows, pick_task1_run, pick_task2_run
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
print(pick_task1_run(rows), pick_task2_run(rows))
PY
)"
T1_MODEL="models/sweeps/${T1_RUN}.pt.best"
T2_MODEL="models/sweeps/${T2_RUN}.pt.best"

log "Best Task-1 (MRR): ${T1_RUN} -> ${T1_MODEL}"
log "Best Task-2 (token acc): ${T2_RUN} -> ${T2_MODEL}"

for m in "$T1_MODEL" "$T2_MODEL"; do
    if [[ ! -e "$m" ]]; then
        log "ERROR missing checkpoint: $m"
        exit 1
    fi
done

log "Generating Task 1 + Task 3 submission (GPU)..."
MODEL="$T1_MODEL" sbatch --account="${SLURM_ACCOUNT:-EUHPC_D30_031}" \
    scripts/leonardo/05_predict_transformer.slurm

log "Generating Task 2 completion (GPU)..."
mkdir -p result/.staging_t2
MODEL="$T2_MODEL" OUT_DIR=result/.staging_t2 ANOM="" \
    sbatch --account="${SLURM_ACCOUNT:-EUHPC_D30_031}" \
    scripts/leonardo/05_predict_transformer.slurm

log "Waiting for predict jobs..."
sleep 30
while squeue --me -h 2>/dev/null | grep -q inf-predict; do
    log "predict jobs still running..."
    sleep 60
done

if [[ -f result/.staging_t2/completion.csv ]]; then
    cp result/.staging_t2/completion.csv result/submission/completion.csv
    log "Merged Task-2 completion into result/submission/"
fi

log "Done. Summary:"
python - <<'PY' | tee -a "$LOG"
import csv
from pathlib import Path

rows = {r["run"]: r for r in csv.DictReader(open("artifacts/sweeps/LEADERBOARD_FINAL.csv"))}
wave2 = [r for name, r in rows.items() if name.startswith("g_")]
wave2.sort(key=lambda r: float(r["task1_mrr"]), reverse=True)
print("\n=== Wave 2 results (by Task-1 MRR) ===")
for r in wave2:
    print(f"  {r['run']:22s}  MRR={r['task1_mrr']}  top1={r['task1_top1']}  tok_acc={r['task2_token_acc']}")

all_rows = sorted(rows.values(), key=lambda r: float(r["task1_mrr"]), reverse=True)
print("\n=== Overall best Task-1 ===")
r = all_rows[0]
print(f"  {r['run']}  MRR={r['task1_mrr']}  top1={r['task1_top1']}")

by_t2 = sorted(rows.values(), key=lambda r: float(r["task2_token_acc"]), reverse=True)
r = by_t2[0]
print("\n=== Overall best Task-2 ===")
print(f"  {r['run']}  tok_acc={r['task2_token_acc']}")

subs = Path("result/submission")
for name in ("nextstep.csv", "completion.csv", "anomaly.csv"):
    p = subs / name
    print(f"  {name}: {'OK' if p.exists() else 'MISSING'} ({p.stat().st_size if p.exists() else 0} bytes)")
PY
