#!/bin/bash
# Poll GPU jobs and log epoch / metrics progress. Run in background:
#   nohup bash scripts/leonardo/monitor_waves.sh >> logs/monitor-waves.out 2>&1 &
set -u

REPO="${1:-$PWD}"
cd "$REPO"
INTERVAL="${MONITOR_INTERVAL:-180}"

while true; do
    echo ""
    echo "======== $(date -Is) ========"
    squeue --me -o "%.18i %.12P %.20j %.2t %.10M" 2>/dev/null || echo "(no squeue)"
    for pat in "43145135" "h_mod_" "t2_mod_" "g_"; do
        for f in logs/slurm-sweep-*_${pat}*.out logs/slurm-sweep-*_*.out; do
            [[ -f "$f" ]] || continue
            base=$(basename "$f" .out)
            epoch=$(grep -E "epoch [0-9]+/" "$f" 2>/dev/null | tail -1 || true)
            t1=$(grep "task1 eval" "$f" 2>/dev/null | tail -1 || true)
            t2=$(grep "task2 eval" "$f" 2>/dev/null | tail -1 || true)
            if [[ -n "$epoch" || -n "$t1" ]]; then
                echo "  ${base}: ${epoch:-?} | ${t1:-} | ${t2:-}"
            fi
        done 2>/dev/null
    done
    n_json=$(find artifacts/sweeps -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    n_best=$(find models/sweeps -maxdepth 1 -name '*.pt.best' 2>/dev/null | wc -l)
    echo "  artifacts: ${n_json} metrics jsons, ${n_best} .pt.best checkpoints"
    if [[ -f logs/wave_orchestrator.log ]]; then
        tail -2 logs/wave_orchestrator.log
    elif [[ -f logs/wave2_finalize-43145135.log ]]; then
        tail -2 logs/wave2_finalize-43145135.log
    fi
    sleep "$INTERVAL"
done
