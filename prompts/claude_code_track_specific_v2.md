# Claude Code Prompt — Build `industrial-models-that-learn-how-processes-unfold`

```xml
<role>
You are the senior implementation agent for `industrial-models-that-learn-how-processes-unfold`. Build a reliable vertical slice for Process Sequence Model.
</role>

<context>
Read CLAUDE.md, AGENTS.md, SKILLS.md, MEMORY.md, docs/JUDGE_CARD.md, and docs/LLM_ORCHESTRATION_FOR_THIS_REPO.md.
</context>

<task>
Implement or complete the starting project:
1. `scripts/generate_synthetic_data.py` matching the track.
2. `scripts/inspect_data.py` with schema validation.
3. `scripts/train_baseline.py` for rolling statistics + GRU/1D-CNN baseline.
4. `scripts/evaluate.py` for next-step accuracy, masked reconstruction loss, anomaly AUROC, early-warning lead time, inference latency.
5. `src/models/` with baseline and advanced-model interface for Transformer encoder / TCN / hybrid state-space fallback with masked-step prediction.
6. `src/app/main.py` demo for: process timeline viewer with predicted next steps, drift warning, anomaly heatmap, and what-changed explanation.
7. `tests/` smoke and unit tests.
8. README quickstart commands.
9. Append a handoff entry to MEMORY.md for Codex.
</task>

<constraints>
- Must run locally on synthetic data.
- Must be easy to replace synthetic data with real hackathon data.
- Must not overengineer.
- The advanced model can be a clean extension point if data is not yet available.
- Prioritize evaluation + demo over complex architecture.
</constraints>
```
