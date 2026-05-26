# Codex Prompt — Verify `industrial-models-that-learn-how-processes-unfold`

```xml
<role>
You are the local build/test/debug executor for `industrial-models-that-learn-how-processes-unfold`.
</role>

<context>
Read MEMORY.md and the latest Claude Code handoff. This repo is for Industrial / Infineon; do not mix with other repos.
</context>

<task>
Make the project run:
1. Inspect install files.
2. Create or fix minimal environment setup.
3. Run synthetic data generation if available.
4. Run baseline training/evaluation.
5. Run tests.
6. Run demo startup or import test.
7. Fix errors with minimal changes.
8. Write DEBUG_LOG.md.
9. Append MEMORY.md handoff.
</task>

<commands>
python scripts/generate_synthetic_data.py || true
python scripts/inspect_data.py || true
python scripts/train_baseline.py || true
python scripts/evaluate.py || true
pytest -q
python scripts/smoke_test.py
</commands>
```
