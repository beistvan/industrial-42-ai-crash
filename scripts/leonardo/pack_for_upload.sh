#!/bin/bash
# Optional helper when git clone/pull is not available. Prefer git if the repo is pushed.
set -euo pipefail
OUT="${1:-industrial-infineon-leonardo-upload.tgz}"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='logs' \
  -czf "$OUT" .
echo "wrote $OUT"
echo "Upload with: scp $OUT <USER>@login01-ext.leonardo.cineca.it:\$SCRATCH/"
