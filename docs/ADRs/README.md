# Architecture Decision Records

Short, dated, blame-free records of *why* we built something a particular way.
Each ADR follows a 5-line shape: **Context · Decision · Consequences ·
Alternatives considered · Status**. Add a new ADR when a decision will be hard
to recover from log or git history later (e.g. "why didn't we just fine-tune
GPT-2?").

| # | Title | Status |
|---|---|---|
| 0001 | Why no HuggingFace pretrained model | accepted |
| 0002 | Rule-constrained decoding default + opt-out | accepted |
| 0003 | Beam search is opt-in, not default | accepted |
| 0004 | Parameter-enrichment training (Wave 5 fallback) | accepted |

When superseded, change the status line in the ADR header and link the
replacement; do not delete.
