#!/bin/bash
# Pick best T1/T2 from LEADERBOARD_FINAL.csv and write judge CSVs.
#
# Task 3 (anomaly.csv) is hybrid — no separate checkpoint:
#   IS_VALID, PREDICTED_RULE  → official rule validator (fixed logic)
#   SCORE                     → T1 model LM log-prob (same --model as nextstep)
# So a new T1 leader updates nextstep.csv + anomaly.csv; T2 only updates completion.csv.
#
# Requires CUDA (or set DEVICE=cpu for a slow smoke run).
set -euo pipefail

REPO="${1:-$PWD}"
cd "$REPO"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LEADERBOARD="${LEADERBOARD:-artifacts/sweeps/LEADERBOARD_FINAL.csv}"
OUT="${OUT_DIR:-result/submission}"
OUT_T2="${OUT_DIR_T2:-result/.staging_t2}"
# Tier C inference: wider beam/pool on T2 completion only (T1 uses top-k, not beam).
T2_BEAM_WIDTH="${T2_BEAM_WIDTH:-5}"
T2_CANDIDATE_POOL="${T2_CANDIDATE_POOL:-8}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$LEADERBOARD" ]]; then
    echo "Missing $LEADERBOARD — run: make leaderboard-final" >&2
    exit 1
fi

read -r T1_RUN T2_RUN T1_SCORE T2_SCORE <<<"$(python3 - <<PY
import sys
sys.path.insert(0, ".")
from scripts.sweep_picks import (
    load_leaderboard_rows,
    pick_task1_run,
    pick_task2_run,
    score_task1,
    score_task2,
    pick_task1_row,
    pick_task2_row,
)
rows = load_leaderboard_rows("${LEADERBOARD}")
t1 = pick_task1_row(rows)
t2 = pick_task2_row(rows)
print(pick_task1_run(rows), pick_task2_run(rows), f"{score_task1(t1):.4f}", f"{score_task2(t2):.4f}")
PY
)"

T1_MODEL="models/sweeps/${T1_RUN}.pt.best"
T2_MODEL="models/sweeps/${T2_RUN}.pt.best"

for m in "$T1_MODEL" "$T2_MODEL"; do
    if [[ ! -e "$m" ]]; then
        echo "Missing checkpoint: $m" >&2
        exit 1
    fi
done

echo "Task 1 + anomaly: ${T1_RUN} (MRR ${T1_SCORE}) -> ${T1_MODEL}"
echo "Task 2 completion: ${T2_RUN} (tok ${T2_SCORE}) -> ${T2_MODEL}"
echo "T2 inference: beam=${T2_BEAM_WIDTH} pool=${T2_CANDIDATE_POOL} device=${DEVICE}"

mkdir -p "$OUT" "$OUT_T2"
python3 scripts/predict_submission.py \
    --model "$T1_MODEL" \
    --eval-valid EVAL_DATA/eval_input_valid.csv \
    --eval-anomaly EVAL_DATA/eval_input_anomaly.csv \
    --out-dir "$OUT" \
    --rule-constrained --beam-width 1 --candidate-pool 5 --device "$DEVICE"
python3 scripts/predict_submission.py \
    --model "$T2_MODEL" \
    --eval-valid EVAL_DATA/eval_input_valid.csv \
    --out-dir "$OUT_T2" \
    --rule-constrained --beam-width "$T2_BEAM_WIDTH" --candidate-pool "$T2_CANDIDATE_POOL" --device "$DEVICE"
cp "$OUT_T2/completion.csv" "$OUT/completion.csv"
echo "Wrote $OUT/{nextstep,completion,anomaly}.csv"
