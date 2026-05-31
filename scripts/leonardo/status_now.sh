#!/bin/bash
# One-shot pipeline snapshot (queue + leaderboard picks + submission files).
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

module purge 2>/dev/null || true
module load python/3.11.7 2>/dev/null || true
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

echo "======== $(date -Is) — Leonardo pipeline status ========"
echo ""

echo "## Slurm queue ($(squeue --me -h 2>/dev/null | wc -l) jobs)"
squeue --me -o "%.12i %.10j %.2t %.10M %R" 2>/dev/null || echo "(squeue unavailable)"
echo ""

echo "## Submission picks"
if [[ -f artifacts/sweeps/LEADERBOARD_FINAL.csv ]]; then
    python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from scripts.sweep_picks import (
    load_leaderboard_rows, pick_task1_row, pick_task2_row,
    score_task1, score_task2,
)
rows = load_leaderboard_rows("artifacts/sweeps/LEADERBOARD_FINAL.csv")
t1, t2 = pick_task1_row(rows), pick_task2_row(rows)
print(f"  T1 + anomaly: {t1['run']}  MRR={score_task1(t1):.4f}")
print(f"  T2 completion: {t2['run']}  tok={score_task2(t2):.4f}")
for run in (t1["run"], t2["run"]):
    ckpt = Path(f"models/sweeps/{run}.pt.best")
    print(f"  checkpoint {run}: {'OK' if ckpt.exists() else 'MISSING'}")
PY
else
    echo "  (no LEADERBOARD_FINAL.csv — run: make leonardo-leaderboard-final)"
fi
echo ""

echo "## Judge CSVs (result/submission/)"
for f in nextstep.csv completion.csv anomaly.csv; do
    p="result/submission/$f"
    if [[ -f "$p" ]]; then
        lines=$(wc -l < "$p")
        bytes=$(stat -c%s "$p" 2>/dev/null || stat -f%z "$p")
        echo "  $f  lines=$lines  size=${bytes}B"
    else
        echo "  $f  MISSING"
    fi
done
echo ""

echo "## Running sweep progress (latest log lines)"
for f in logs/slurm-sweep-*.out; do
    [[ -f "$f" ]] || continue
    jid=$(basename "$f" .out | sed 's/slurm-sweep-//')
    parent="${jid%%_*}"
    if ! squeue -j "$parent" -h 2>/dev/null | grep -q .; then
        continue
    fi
    run=$(grep -m1 "run='" "$f" 2>/dev/null | sed "s/.*run='//;s/'.*//" || echo "?")
    epoch=$(grep -E "epoch [0-9]+/" "$f" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' || true)
    best=$(grep "new best" "$f" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' || true)
    echo "  ${jid} ${run}"
    [[ -n "$epoch" ]] && echo "    ${epoch}"
    [[ -n "$best" ]] && echo "    ${best}"
done
echo ""

n_json=$(find artifacts/sweeps -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
n_best=$(find models/sweeps -maxdepth 1 -name '*.pt.best' 2>/dev/null | wc -l)
echo "## Artifacts: ${n_json} metrics JSONs, ${n_best} .pt.best checkpoints"
echo ""
echo "Next: make leonardo-leaderboard-final && make regenerate-submission"
