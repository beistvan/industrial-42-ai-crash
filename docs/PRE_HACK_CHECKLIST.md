# Pre-Hack Checklist — Industrial: Models that learn how processes unfold

## By T-3 days
- Repo pushed to GitHub with this folder structure.
- Local Python environment works.
- `make test` passes.
- `make run-demo` starts a placeholder app.
- Baseline dataset fallback exists or can be downloaded manually.
- `docs/PITCH_SCRIPT.md` has a 60-second version.

## By T-1 day
- Both teammates can clone and run the repo.
- Claude, Codex, Copilot, and DeepSeek prompts are ready.
- Decide kill criteria: when to stop improving model and polish demo.
- Prepare USB/cloud backup of screenshots and pitch.

## Bring to venue
- laptop, charger, ID;
- account logins;
- GitHub tokens/SSH key;
- API keys only if safe and optional;
- sleeping kit if staying overnight.

---

# Pre-hack checklist — Industrial: Models that learn how processes unfold

This is the readiness checklist. Do not branch into other planning docs.

## Must pass before track reveal

- [ ] `make smoke` passes.
- [ ] `make test` passes.
- [ ] `make run-demo` opens Streamlit.
- [ ] `artifacts/metrics.json` exists.
- [ ] Demo shows metric evidence, scenario input, recommendation/decision, explanation, and data preview.
- [ ] No deprecated Streamlit `use_container_width` warnings.
- [ ] README states the authoritative file path.
- [ ] Working state is committed/tagged.

## Current evidence

- Rows: `2400`
- Accuracy: `0.9983`
- F1 anomaly: `0.9934`
- Baseline: RandomForest anomaly classifier on synthetic process-sequence telemetry

## Files allowed during prep

- `README.md`
- `docs/PRE_HACK_CHECKLIST.md`
- `docs/TASK_BOARD.md`
- `docs/DEMO_SPEC.md`
- `docs/JUDGE_CARD.md`
- `artifacts/metrics.json`

Other docs are reference only.

## Final pre-hack command

```bash
make smoke && make test
```

Then run the demo manually:

```bash
make run-demo
```
