# Pipeline — train, validate, improve, review

> **Note (2026-05-31):** This doc is partially stale. For submission-final
> instructions use [`HANDOFF.md`](../HANDOFF.md), [`docs/SUBMISSION.md`](SUBMISSION.md),
> and [`configs/sweeps/WINNING_RECIPES.md`](../configs/sweeps/WINNING_RECIPES.md).

This is the working pipeline for the Infineon track. It targets the three
scored tasks (next-step prediction, sequence completion, anomaly detection)
and produces ready-to-submit CSVs.

## TL;DR

```bash
python scripts/make_dev_split.py --dev-per-family 100
python scripts/train_ngram.py    --max-order 8
python scripts/predict_submission.py \
  --model models/ngram_baseline.pkl \
  --eval-valid   data/raw/infineon/eval/eval_input_valid.csv \
  --eval-anomaly data/raw/infineon/eval/eval_input_anomaly.csv \
  --out-dir result/dev
```

`make_dev_split.py` writes the split ids and dev eval CSVs under
`data/processed/`. `train_ngram.py` fits on the train ids, evaluates on the
dev CSVs, and writes `artifacts/ngram_metrics.json`. `predict_submission.py`
produces `nextstep.csv`, `completion.csv`, `anomaly.csv` once the organizers
ship the official eval inputs into `data/raw/infineon/eval/`.

The split files are persisted so that all models (n-gram, transformer, any
HF fine-tune) and **all evaluators** (this repo's `src/eval/run_eval.py` and
the friend's `src/eval/local_eval.py`) score against the *same* dev set.
This gives us two independent evaluators whose numbers must agree.

## Module map

```
src/
├── data/
│   ├── infineon_loader.py     load_family / load_all_families, Vocabulary
│   └── splits.py              deterministic_split(train/dev)
├── ml/
│   ├── ngram_baseline.py      NGramBaseline (suffix-backoff next-step)
│   └── transformer_model.py   placeholder for next phase (transformer)
├── eval/
│   ├── metrics.py             top_k_accuracy, MRR, exact_match,
│   │                          token_accuracy, normalized_edit_distance
│   ├── rule_validator.py      thin wrapper around the official
│   │                          validate_sequence() — used for Task 3
│   ├── dev_split.py           build local eval inputs mirroring the
│   │                          official format (60% / 80% truncations,
│   │                          injected anomalies with labels)
│   ├── run_eval.py            evaluate_all(model, eval_dir) -> Task 1/2/3
│   │                          (this repo's evaluator)
│   └── local_eval.py          friend's independent evaluator (placeholder
│                              — left for him to fill in so we cross-check)
└── inference/
    └── predict.py             placeholder for next phase

scripts/
├── make_dev_split.py          load -> split -> persist ids + dev eval CSVs
├── train_ngram.py             load split -> fit -> evaluate -> save metrics
├── train_transformer.py       placeholder for next phase
└── predict_submission.py      produce official submission CSVs

tests/
├── conftest.py                puts repo root on sys.path
├── test_infineon_loader.py    loader + Vocabulary
├── test_ngram_baseline.py     NGramBaseline fit / predict / save / load
├── test_metrics.py            metric formulas on toy inputs
└── test_rule_validator.py     validator wrapper (validates real 1k seqs)
```

## End-to-end loop: split -> train -> validate -> improve -> review

1. **Split.** `python scripts/make_dev_split.py [--dev-per-family K]`
   - Loads all three families, deterministically holds out `K` sequences
     per family, persists train/dev ids and dev eval CSVs under
     `data/processed/`.
2. **Train.** `python scripts/train_ngram.py [--max-order N]`
   - Fits `NGramBaseline(max_order=N)` on the train ids, saves model,
     evaluates on the dev CSVs, writes `artifacts/ngram_metrics.json`.
3. **Validate.** Inspect `artifacts/ngram_metrics.json`. Track:
   - **Task 1:** `overall.top1`, `overall.top5`, `overall.mrr`.
   - **Task 2:** `overall.token_accuracy`, `overall.normalized_edit_distance`.
   - **Task 3:** `f1_invalid`, `rule_attribution_accuracy`.
   - Per-family breakdowns reveal the weakest family.
4. **Improve.** Edit one knob at a time and re-run:
   - `--max-order` (4 → 8 → 12): higher order helps Top-1 but starves rare suffixes.
   - Generate extra training sequences with the official CLI:
     `python data/raw/infineon/training_data/generate_sequences.py --family mosfet --count 10000 --output extra.csv --seed 101`
     then point a future loader at the augmented corpus.
   - Add a beam-search completion in `NGramBaseline.complete()` for Task 2.
5. **Review.** Diff the metrics JSON between runs (or paste both into the
   judge table). The same `evaluate_all(model, eval_dir)` interface will be
   reused by the transformer / any fine-tuned model — see "swap-in surface"
   below. Cross-check against the friend's `src/eval/local_eval.py` once it
   lands: both evaluators read the same dev CSVs and must produce the same
   headline numbers — disagreement means one of the two is wrong.

## Reference dev-set numbers (n-gram, max_order=8, 50 dev/family)

| Task | Metric | Overall | MOSFET | IGBT | IC |
| --- | --- | ---: | ---: | ---: | ---: |
| 1   | Top-1 | 0.683 | 0.650 | 0.710 | 0.690 |
| 1   | Top-5 | 0.990 | 0.990 | 0.990 | 0.990 |
| 1   | MRR   | 0.834 | 0.818 | 0.847 | 0.837 |
| 2   | Token acc | 0.419 | 0.496 | 0.478 | 0.282 |
| 2   | Normed edit dist | 0.227 | 0.164 | 0.215 | 0.302 |
| 3   | F1 (invalid) | 1.000 | — | — | — |
| 3   | Rule attribution | 0.660 | — | — | — |

Anomaly numbers are perfect on the dev set because Task 3 currently uses
the official rule validator directly. On the real eval set the score will
depend on how often injected violations actually trip one of the 10 rules
— the validator is the upper-bound for a rules-only solution. Rule
attribution 0.66 reflects that injected mutations sometimes trigger a
different rule than the one we labelled (e.g. dropping a clean step can
also trip implant / metal-etch rules); the metric is conservative.

## Swap-in surface (for the transformer or a fine-tuned HF model)

Any next-step model that implements this duck-typed interface drops into
the same evaluator with zero changes:

```python
class NextStepModel(Protocol):
    def predict_topk(self, family: Family, prefix: list[str], k: int = 5) -> list[str]: ...
    def complete(self, family: Family, prefix: list[str], *,
                 max_steps: int = 200, stop_token: str = "SHIP LOT") -> list[str]: ...
```

The transformer model class and any HF-pretrained wrapper will implement
exactly these two methods, and `evaluate_all(model, eval_dir)` will return
the same metrics shape — making baseline-vs-transformer-vs-finetuned a
single shared table.

## Submission

The shape we write today:

| File | Columns |
| --- | --- |
| `nextstep.csv`   | SEQUENCE_ID, RANK (1..k), STEP |
| `completion.csv` | SEQUENCE_ID, STEP_INDEX (0..n), STEP |
| `anomaly.csv`    | SEQUENCE_ID, IS_VALID (0/1), PRIMARY_RULE, ALL_RULES |

Files are written to `result/submission/` for judge eval inputs and
`result/dev/` for local dev-split smoke runs. If the organizers'
`eval_metrics.py` expects a different column layout, adjust the writer in
`scripts/predict_submission.py` only; nothing else needs to change.
