#!/bin/bash
# Submit Wave 5 (parameter-enriched training) only if Wave 3/4 did not beat bars.
#
# Usage:
#   bash scripts/leonardo/wave5_if_needed.sh
#   bash scripts/leonardo/wave5_if_needed.sh --force   # skip gate, always submit
#
# Bars (dev holdout, Wave 2 submission):
#   T1 MRR       >= 0.8731  (f_drop15_100_mrr)
#   T2 tok-acc   >= 0.4545  (g_drop15_nosched_t2)
set -euo pipefail

FORCE=0
REPO="$PWD"
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
elif [[ -n "${1:-}" ]]; then
    REPO="$1"
fi
cd "$REPO"

module purge || true
module load python/3.11.7 || true
source .venv/bin/activate

LEADERBOARD="${LEADERBOARD:-artifacts/sweeps/LEADERBOARD_FINAL.csv}"
LOG="logs/wave5_gate.log"
mkdir -p logs

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

if [[ ! -f "$LEADERBOARD" ]]; then
    log "Missing $LEADERBOARD — run make leonardo-leaderboard-final first"
    exit 1
fi

read -r W3_BEST_MRR W4_BEST_TOK OVERALL_T1 OVERALL_T2 <<<"$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.sweep_picks import (
    load_leaderboard_rows, pick_task1_row, pick_task2_row,
    score_task1, score_task2,
)
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
w3 = [r for r in rows if r["run"].startswith("h_mod_")]
w4 = [r for r in rows if r["run"].startswith("t2_mod_")]
w3m = max((score_task1(r) for r in w3), default=0.0)
w4t = max((score_task2(r) for r in w4), default=0.0)
t1 = pick_task1_row(rows)
t2 = pick_task2_row(rows)
print(w3m, w4t, t1["run"], t2["run"])
PY
)"

T1_BAR=0.8731
T2_BAR=0.4545

log "Wave 3 best MRR=${W3_BEST_MRR} (bar ${T1_BAR})"
log "Wave 4 best tok=${W4_BEST_TOK} (bar ${T2_BAR})"
log "Overall T1=${OVERALL_T1}  T2=${OVERALL_T2}"

W3_OK=$(python3 - <<PY
print(1 if float("${W3_BEST_MRR}") >= ${T1_BAR} - 1e-6 else 0)
PY
)
W4_OK=$(python3 - <<PY
print(1 if float("${W4_BEST_TOK}") >= ${T2_BAR} - 1e-6 else 0)
PY
)

if [[ "$FORCE" == "1" ]]; then
    log "FORCE=1 — submitting Wave 5 regardless of Wave 3/4 results"
elif [[ "$W3_OK" == "1" && "$W4_OK" == "1" ]]; then
    log "Wave 3/4 beat both bars — Wave 5 not needed."
    exit 0
elif [[ "$W3_OK" == "1" ]]; then
    log "Wave 3 beat T1 bar; Wave 4 did not beat T2 bar — submitting Wave 5 (T2-focused rows may help)"
elif [[ "$W4_OK" == "1" ]]; then
    log "Wave 4 beat T2 bar; Wave 3 did not beat T1 bar — submitting Wave 5 (T1-focused rows may help)"
else
    log "Neither Wave 3 nor Wave 4 beat bars — submitting Wave 5 (parameter enrichment fallback)"
fi

N=$(python3 scripts/sweep_transformer.py --sweep configs/sweeps/leonardo_params.yaml --stage finalists --dry-run | grep -c '^\[')
log "Submitting Wave 5: ${N} rows"
JOB_ID=$(bash scripts/leonardo/submit_sweep.sh configs/sweeps/leonardo_params.yaml finalists)
log "Wave 5 job ${JOB_ID}"
echo "$JOB_ID" > logs/wave5_job_id.txt
log "After Wave 5 finishes: make leonardo-leaderboard-final && make regenerate-submission"
