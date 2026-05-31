#!/usr/bin/env bash
# Pull Leonardo training outputs back to your Mac (run on laptop).
#
# Usage:
#   bash scripts/leonardo/copy_from_leonardo.sh
#   bash scripts/leonardo/copy_from_leonardo.sh models artifacts/sweeps logs
#
# Default: pulls models/, artifacts/, logs/, result/submission*

set -euo pipefail

LEONARDO_USER="${LEONARDO_USER:-a08trd0x}"
LEONARDO_HOST="${LEONARDO_HOST:-login01-ext.leonardo.cineca.it}"
REMOTE="${LEONARDO_USER}@${LEONARDO_HOST}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_SCRATCH="$(ssh "$REMOTE" 'echo "$SCRATCH"')"
REMOTE_PATH="${REMOTE_SCRATCH}/industrial"

if [[ $# -gt 0 ]]; then
    PATHS=("$@")
else
    PATHS=(models artifacts logs result)
fi

echo "=== Pull from Leonardo ==="
echo "Remote: ${REMOTE}:${REMOTE_PATH}"
echo "Local:  ${REPO_ROOT}"
echo "Paths:  ${PATHS[*]}"
echo ""

for p in "${PATHS[@]}"; do
    rsync -avz --progress "${REMOTE}:${REMOTE_PATH}/${p}/" "${REPO_ROOT}/${p}/" 2>/dev/null || \
        rsync -avz --progress "${REMOTE}:${REMOTE_PATH}/${p}" "${REPO_ROOT}/${p}" 2>/dev/null || \
        echo "  (skip ${p} — not found on Leonardo)"
done

echo ""
echo "Done. Check locally:"
echo "  ls models/ artifacts/sweeps/ logs/"
