#!/usr/bin/env bash
# Copy the local repo to Leonardo $SCRATCH/industrial (run on your laptop).
#
# Leonardo login nodes cannot reach GitHub — use this instead of git clone.
#
# Usage:
#   bash scripts/leonardo/copy_to_leonardo.sh
#   LEONARDO_USER=a08trd0x bash scripts/leonardo/copy_to_leonardo.sh
#   LEONARDO_HOST=login01-ext.leonardo.cineca.it bash scripts/leonardo/copy_to_leonardo.sh
#
# Requires: ssh access to Leonardo (password or step certificate).

set -euo pipefail

LEONARDO_USER="${LEONARDO_USER:-a08trd0x}"
LEONARDO_HOST="${LEONARDO_HOST:-login01-ext.leonardo.cineca.it}"
REMOTE_DIR="${REMOTE_DIR:-\$SCRATCH/industrial}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${LEONARDO_USER}@${LEONARDO_HOST}"

echo "=== Copy repo to Leonardo ==="
echo "Local:  $REPO_ROOT"
echo "Remote: ${REMOTE}:${REMOTE_DIR}"
echo ""

# Expand $SCRATCH on the remote side for rsync target.
REMOTE_SCRATCH="$(ssh "$REMOTE" 'echo "$SCRATCH"')"
REMOTE_PATH="${REMOTE_SCRATCH}/industrial"

echo "Remote SCRATCH=$REMOTE_SCRATCH"
echo "Syncing to $REMOTE_PATH ..."
echo ""

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '.pytest_cache' \
    --exclude 'models/' \
    --exclude 'artifacts/' \
    --exclude 'logs/' \
    --exclude '.ruff_cache' \
    "$REPO_ROOT/" "${REMOTE}:${REMOTE_PATH}/"

echo ""
echo "Done. On Leonardo, run:"
echo "  ssh ${REMOTE}"
echo "  cd \$SCRATCH/industrial"
echo "  bash scripts/leonardo/stage0_setup.sh"
echo "  bash scripts/leonardo/stage0_setup.sh --submit   # setup + submit sweep"
