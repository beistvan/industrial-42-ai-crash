#!/bin/bash
# Export pitch deck to PDF for Tally upload.
#
# Preferred: styled PowerPoint (team layout) → PDF via LibreOffice.
# Fallback: Marp render of SLIDES.md (content-only, default theme).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slides

PPTX="slides/industrial_pitch_deck.pptx"
PDF="slides/industrial_pitch_deck.pdf"

if command -v soffice >/dev/null 2>&1 && [[ -f "$PPTX" ]]; then
    echo "Converting $PPTX → $PDF (LibreOffice)..."
    soffice --headless --convert-to pdf --outdir slides "$PPTX"
    echo "Wrote $PDF"
    exit 0
fi

if command -v libreoffice >/dev/null 2>&1 && [[ -f "$PPTX" ]]; then
    echo "Converting $PPTX → $PDF (libreoffice)..."
    libreoffice --headless --convert-to pdf --outdir slides "$PPTX"
    echo "Wrote $PDF"
    exit 0
fi

if [[ -f "$PPTX" ]]; then
    echo "Note: LibreOffice not found — upload $PPTX to PowerPoint/Google Slides and export PDF for styled deck."
fi

echo "Fallback: Marp render SLIDES.md → $PDF"
if command -v marp >/dev/null 2>&1; then
    marp SLIDES.md -o "$PDF"
elif command -v npx >/dev/null 2>&1; then
    npx --yes @marp-team/marp-cli@latest SLIDES.md -o "$PDF"
else
    echo "ERROR: install marp-cli or npx, or export PDF manually from $PPTX" >&2
    exit 1
fi
echo "Wrote $PDF"
