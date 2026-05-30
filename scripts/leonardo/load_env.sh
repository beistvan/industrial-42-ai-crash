#!/usr/bin/env bash
# Load Leonardo modules so python3 is >= 3.10 (required for scikit-learn 1.6.1).
# Source from setup/slurm scripts: source scripts/leonardo/load_env.sh

set -euo pipefail

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    return 0 2>/dev/null || true
fi

echo "Loading Leonardo Python module (system default is 3.6)..."

module purge 2>/dev/null || true

if module is-avail profile/deeplrn 2>/dev/null; then
    module load profile/deeplrn
elif module is-avail profile/base 2>/dev/null; then
    module load profile/base
fi

LOADED=0
for mod in \
    python/3.11.6--gcc--8.5.0 \
    python/3.11.7 \
    python/3.11.6 \
    python/3.10.8--gcc--11.3.0; do
    if module is-avail "$mod" 2>/dev/null; then
        module load "$mod"
        LOADED=1
        break
    fi
done

if [[ "$LOADED" -eq 0 ]]; then
    echo "ERROR: no Python 3.10+ module found." >&2
    echo "Run: modmap -m python" >&2
    echo "Then: module load python/<version>" >&2
    echo "And re-run: bash scripts/leonardo/stage0_setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi

if ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "ERROR: loaded module still gives $(python3 --version 2>&1)" >&2
    return 1 2>/dev/null || exit 1
fi

echo "Using $(python3 --version) at $(command -v python3)"

# CUDA runtime for compute-node jobs (A100 = CUDA 12.1 on Leonardo).
for cuda_mod in cuda/12.1 cuda/12.2 cuda/12.3; do
    if module is-avail "$cuda_mod" 2>/dev/null; then
        module load "$cuda_mod"
        echo "Loaded $cuda_mod"
        break
    fi
done
