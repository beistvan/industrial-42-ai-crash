#!/usr/bin/env bash
# Stage 0 on Leonardo login node: env, data prep, optional submit.
#
# Prerequisite: copy the repo from your laptop first (Leonardo has no GitHub):
#   bash scripts/leonardo/copy_to_leonardo.sh          # run on Mac
#
# Then on Leonardo:
#   cd $SCRATCH/industrial
#   bash scripts/leonardo/stage0_setup.sh
#   bash scripts/leonardo/stage0_setup.sh --submit
#
# Options:
#   --submit  After setup, submit the shortlist sweep (run_battle.sh).

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SCRATCH:?SCRATCH not set}/industrial}"
SUBMIT=0

for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "=== Leonardo Stage 0 ==="
echo "SCRATCH=$SCRATCH"
echo "REPO_DIR=$REPO_DIR"

if [[ ! -f "$REPO_DIR/scripts/leonardo/stage0_setup.sh" ]]; then
    echo "ERROR: repo not found at $REPO_DIR" >&2
    echo "Copy it from your laptop first:" >&2
    echo "  bash scripts/leonardo/copy_to_leonardo.sh" >&2
    exit 1
fi

cd "$REPO_DIR"
mkdir -p logs models/sweeps artifacts/sweeps

# Leonardo login node defaults to Python 3.6 — load 3.11+ first.
# shellcheck source=scripts/leonardo/load_env.sh
source scripts/leonardo/load_env.sh

echo "=== Python venv ==="
if [[ -d .venv ]]; then
    VENV_VER="$(.venv/bin/python -c 'import sys; print(sys.version_info[:2])' 2>/dev/null || echo "(0, 0)")"
    if [[ "$VENV_VER" == "(3, 6)" ]] || [[ "$VENV_VER" == "(0, 0)" ]]; then
        echo "Removing stale .venv (built with old Python)..."
        rm -rf .venv
    fi
fi
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# Install deps WITHOUT torch — requirements.txt pulls cu130 which breaks on Leonardo.
grep -v '^torch' requirements.txt | grep -v '^#' | grep -v '^$' > /tmp/requirements-no-torch.txt
python -m pip install -r /tmp/requirements-no-torch.txt

# Leonardo A100 nodes: CUDA 12.1 driver (not 13.0).
echo "Installing PyTorch cu121 for Leonardo A100..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

python scripts/check_environment.py --require-torch

echo "=== Data prep ==="
python scripts/make_dev_split.py --force
python scripts/generate_extra_sequences.py --count-per-family 250 --seed 101 --force

echo "=== Quick test (HPC-relevant only) ==="
python -m pytest -q --tb=no \
    tests/test_infineon_loader.py \
    tests/test_infineon_dev_split.py \
    tests/test_transformer_model.py \
    tests/test_sweep_transformer.py \
    tests/test_sweep_picks.py \
    tests/test_anomaly_scoring.py

echo "=== Sweep row count (shortlist) ==="
N=$(python scripts/sweep_transformer.py \
    --sweep configs/sweeps/leonardo_v1.yaml --stage shortlist --dry-run \
    | grep -c '^\[')
echo "  $N rows → array 0-$((N - 1))"

if [[ "$SUBMIT" -eq 1 ]]; then
    echo "=== Submitting shortlist sweep ==="
    bash scripts/leonardo/run_battle.sh configs/sweeps/leonardo_v1.yaml shortlist
    echo ""
    echo "Watch: squeue --me"
    echo "Logs:  tail -f logs/slurm-sweep-*_*.out"
else
    echo ""
    echo "Setup complete. To submit the shortlist sweep:"
    echo "  cd $REPO_DIR && source .venv/bin/activate"
    echo "  bash scripts/leonardo/run_battle.sh configs/sweeps/leonardo_v1.yaml shortlist"
fi
