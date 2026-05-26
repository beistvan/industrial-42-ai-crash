# ChatGPT Web Prompt — Track Planner for `industrial-models-that-learn-how-processes-unfold`

```xml
<role>
You are the product/ML commander for the Industrial / Infineon track.
</role>

<context>
Repo: industrial-models-that-learn-how-processes-unfold
App idea: Process Sequence Model
Core technical direction: sequence encoder + next-state prediction + anomaly/trajectory explanation
Team: two humans, 36-hour hackathon, likely A100 access, must build a working app.
</context>

<task>
Create a practical implementation plan for this repo. Output:
1. Final problem statement.
2. Data contract.
3. Baseline algorithm.
4. Advanced model path.
5. Evaluation metrics.
6. UI/demo screens.
7. File-level implementation plan.
8. What to ask Gemini, DeepSeek, Claude Code, Codex, and Copilot next.
9. 36-hour schedule.
10. What to cut if we are behind.
</task>

<constraints>
- Keep this repo separate.
- Do not depend on unknown external APIs.
- Must run on small local sample data before using A100s.
- Must produce a judge-visible metric and demo.
</constraints>
```
