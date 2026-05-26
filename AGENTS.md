# AGENTS.md — Industrial: Models that learn how processes unfold

## Human owners
- Teammate A: ML/backend, data, model, evaluation, GPU, reproducibility.
- Teammate B: frontend, user workflow, pitch, README, demo reliability.

## Claude Architect Agent
Main prompt: `prompts/claude_opus_build_prompt.md`
Use for architecture, task decomposition, file-level plan, evaluation design, and final review.

## Codex Builder Agent
Main prompt: `prompts/codex_execution_prompt.md`
Use to implement tasks with tests, run commands, and fix failures.

## Copilot Inline Agent
Main prompt: `prompts/copilot_inline_tasks.md`
Use for fast implementation inside VS Code.

## DeepSeek Critic Agent
Main prompt: `prompts/deepseek_critic_prompt.md`
Use for red-team review, edge cases, and judging critique.

## Handoff template
Objective:
Repo:
Files to modify:
Files not to touch:
Acceptance criteria:
Test commands:
Time budget:
