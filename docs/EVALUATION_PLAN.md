# Evaluation Plan — Industrial: Models that learn how processes unfold

Mirrors the official Infineon eval protocol (see
[`../data/raw/infineon/training_data/generation_rules.md`](../data/raw/infineon/training_data/generation_rules.md)
§5) and our internal evaluator at `src/eval/run_eval.py`.

## Tasks and metrics

| # | Task | Input | Metrics |
| - | --- | --- | --- |
| 1 | Next-step prediction | partial sequence | Top-1 / Top-3 / Top-5 accuracy, MRR |
| 2 | Sequence completion  | 60% or 80% prefix | Exact match, token accuracy, normalized edit distance |
| 3 | Anomaly detection    | full sequence (valid or rule-violating) | Binary precision / recall / F1, rule attribution accuracy |
| 4 | OOD generalization   | hidden 4th product family | Performance drop ID → OOD (organizer-evaluated post-submission) |

## Required comparisons

For the judge table we report at least:

1. **Naive baseline** — n-gram suffix backoff (`src/ml/ngram_baseline.py`).
2. **Improved model** — small from-scratch decoder-only transformer
   (`src/ml/transformer_model.py`, next phase).
3. **Optional contrast** — `distilgpt2` fine-tune, evaluated through the
   same harness (see [`FINETUNE_OPTION_REVIEW.md`](./FINETUNE_OPTION_REVIEW.md)).
4. **Scaling row** — same model on the 1,000-seq corpus vs the 10,000-seq
   augmented corpus.

All four read the same dev CSVs, so the rows are directly comparable.

## Local evaluation

```bash
python scripts/make_dev_split.py --dev-per-family 100
python scripts/train_ngram.py    --max-order 8
# → artifacts/ngram_metrics.json (latest) + artifacts/runs/{ts}_{run}.json
```

`scripts/train_ngram.py` calls `src/eval/run_eval.evaluate_all(model, eval_dir)`
internally. The same function evaluates the transformer once it lands —
no separate eval script is needed.

## Cross-check evaluators

`src/eval/run_eval.py` is this repo's evaluator. `src/eval/local_eval.py`
is the friend's independent evaluator (placeholder for now). Both consume
the same `data/processed/dev_eval/*.csv` files; their headline numbers
must agree. Disagreement = one of the two has a bug.

## Judge-facing proof

Single chart + table to ship with the pitch:

- Run-history chart from `src/app/eval_dashboard.py` showing Top-1 / Top-5
  / MRR improving across runs (n-gram → small transformer → medium
  transformer → augmented data).
- Per-task / per-family metrics table from the latest
  `artifacts/ngram_metrics.json` (and the transformer's equivalent file
  when ready).
- One-line business interpretation per row.
- Honest limitation: what the model still misses (e.g. greedy completion
  has 0% exact match; beam is the obvious fix).

## Anti-cheating / reliability

- Deterministic split via `scripts/make_dev_split.py --seed N`; split id
  files are committed-safe artifacts under `data/processed/splits/`.
- Fixed seeds for n-gram + dev anomaly injection (`--seed` / `--anomaly-seed`).
- Vendored synthetic data is clearly labeled (`data/raw/infineon/SOURCE.md`);
  never marketed as real partner data.
- Rule validator is the official `validate_sequence` — no re-implementation.
- All runs snapshotted to `artifacts/runs/` so claims in the pitch are
  reproducible from a single command.
