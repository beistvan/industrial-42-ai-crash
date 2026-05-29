# Pre-Hack Checklist — Industrial: Models that learn how processes unfold

Readiness checklist for the Infineon track. Do not branch into other
planning docs while you work this.

## Repo state — already done

- [x] Official Infineon training data vendored under
      `data/raw/infineon/` (PR #1).
- [x] N-gram baseline + deterministic split + dev eval generator + rule
      validator + tests (PR #2).
- [x] Streamlit eval dashboard + run-history snapshots (PR #3).
- [x] Implementation plan committed (HU + EN) and reviewed against the
      track briefing.

## Must pass before track kickoff

- [ ] `python -m pytest -q tests/test_metrics.py tests/test_ngram_baseline.py tests/test_infineon_loader.py tests/test_rule_validator.py`
      → 18 passed.
- [ ] `python scripts/make_dev_split.py --dev-per-family 100` runs cleanly.
- [ ] `python scripts/train_ngram.py --max-order 8` writes
      `artifacts/ngram_metrics.json` + `artifacts/runs/{ts}_ngram_max8.json`.
- [ ] `streamlit run src/app/eval_dashboard.py` opens at
      `http://localhost:8501` and shows Task 1/2/3 + run history.
- [ ] README "Current baseline state" matches `artifacts/ngram_metrics.json`.
- [ ] Friend's `src/eval/local_eval.py` lands and its numbers agree with
      `src/eval/run_eval.py` within rounding.
- [ ] Leonardo onboarding (`docs/LEONARDO_ONBOARDING.md`) completed by
      every team member who plans to run cluster jobs.
- [ ] No secrets / SSH keys / certificates / invite links committed
      (check `git log -p` for the last 24h).

## Current baseline evidence

| Task | Metric | Overall |
| --- | --- | ---: |
| 1 | Top-1 | 0.683 |
| 1 | Top-5 | 0.990 |
| 1 | MRR   | 0.834 |
| 2 | Token accuracy | 0.419 |
| 2 | Normalized edit distance | 0.227 |
| 3 | F1 (invalid) | 1.000 |
| 3 | Rule attribution | 0.633 |

Run config: NGramBaseline, `max_order=8`, dev=50 sequences/family (n=300
partial sequences for Task 1, 300 for Task 2, 300 for Task 3). See
`artifacts/ngram_metrics.json` for the live numbers.

## Files allowed during prep

- `README.md`
- `docs/PIPELINE.md`
- `docs/DATA_SPEC.md`
- `docs/implementation-plan-en.md`
- `docs/FINETUNE_OPTION_REVIEW.md`
- `docs/LEONARDO_ONBOARDING.md`
- `docs/JUDGE_CARD.md`
- `docs/DEMO_SPEC.md`
- `docs/PITCH_SCRIPT.md`
- `docs/TASK_BOARD.md`
- `artifacts/ngram_metrics.json`
- `artifacts/runs/*.json`

Other docs are reference / historical only.

## Bring to venue

- laptop, charger, ID;
- account logins;
- GitHub tokens / SSH key;
- AI:AT identity provider credentials + 2FA device for `step ssh login`;
- API keys only if safe and optional;
- sleeping kit if staying overnight.

## Final pre-hack command

```bash
python scripts/make_dev_split.py --dev-per-family 100 && \
python scripts/train_ngram.py    --max-order 8 && \
python -m pytest -q tests/test_metrics.py tests/test_ngram_baseline.py \
                    tests/test_infineon_loader.py tests/test_rule_validator.py
```

Then open the dashboard:

```bash
streamlit run src/app/eval_dashboard.py
```
