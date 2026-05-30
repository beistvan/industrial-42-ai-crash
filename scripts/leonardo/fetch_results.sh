#!/bin/bash
# Run locally to copy Leonardo artifacts back after GPU jobs finish.
set -euo pipefail
REMOTE="${1:?usage: scripts/leonardo/fetch_results.sh USER@login01-ext.leonardo.cineca.it:/path/to/repo}"
DEST="${2:-leonardo_results}"
mkdir -p "$DEST"
scp -r "$REMOTE/models" "$DEST/" || true
scp -r "$REMOTE/artifacts" "$DEST/" || true
scp -r "$REMOTE/extras/results_transformer" "$DEST/" || true
scp -r "$REMOTE/extras/results_submission" "$DEST/" || true
scp -r "$REMOTE/logs" "$DEST/" || true
echo "fetched to $DEST"
