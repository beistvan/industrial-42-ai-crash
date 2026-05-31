# Pitch deck

| File | Use |
|---|---|
| `industrial_pitch_deck.pptx` | **Styled deck** — upload to Tally after exporting PDF, or present from PPTX |
| `industrial_pitch_deck.pdf` | Tally slides field — generate locally (see below) |

Content draft / speaker notes: [`../SLIDES.md`](../SLIDES.md)

## Export PDF (styled — matches PPTX layout)

**On a laptop** (HPC login nodes typically have no LibreOffice):

1. Open `industrial_pitch_deck.pptx` in PowerPoint, Keynote, or Google Slides  
2. File → Export / Download → **PDF**  
3. Save as `slides/industrial_pitch_deck.pdf`

Or with LibreOffice installed:

```bash
make slides-pdf
```

## Content-only PDF (Marp, no PPTX styling)

```bash
npm i -g @marp-team/marp-cli   # once, on a machine with npm
marp SLIDES.md -o slides/industrial_pitch_deck_marp.pdf
```
