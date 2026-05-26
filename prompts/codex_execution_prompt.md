# Codex Execution Prompt — Industrial: Models that learn how processes unfold

Work in repository `industrial-models-that-learn-how-processes-unfold`.

Objective:
Implement the next highest-value vertical slice for this hackathon app:
data -> baseline -> model/layer -> evaluation -> Streamlit demo -> smoke test.

Read first:
- `README.md`
- `CLAUDE.md`
- `docs/TASK_BOARD.md`
- `docs/MODEL_PLAN.md`
- `docs/EVALUATION_PLAN.md`
- `docs/DEMO_SPEC.md`

Rules:
- Make small commits or clearly separated changes.
- Run tests after changes.
- Prefer simple, reliable code.
- Do not add heavyweight dependencies unless necessary.
- If external data is unavailable, create a clearly labeled synthetic sample generator.

Acceptance criteria:
- `make test` passes.
- `make smoke` passes.
- `make run-demo` launches.
- README commands are correct.
- At least one metric is generated and saved to `runs/metrics.json` or `artifacts/evaluation_report.md`.

Start by printing a concise plan, then implement it.
