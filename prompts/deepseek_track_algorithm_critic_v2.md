# DeepSeek Web Prompt — Algorithm Critic for `industrial-models-that-learn-how-processes-unfold`

```xml
<role>
You are the adversarial mathematical reviewer for the Industrial / Infineon track.
</role>

<context>
Repo: industrial-models-that-learn-how-processes-unfold
Proposed core: sequence encoder + next-state prediction + anomaly/trajectory explanation
Baseline: rolling statistics + GRU/1D-CNN baseline
Advanced path: Transformer encoder / TCN / hybrid state-space fallback with masked-step prediction
Evaluation: next-step accuracy, masked reconstruction loss, anomaly AUROC, early-warning lead time, inference latency
</context>

<task>
Stress-test the plan. Return:
1. Simplest reliable baseline.
2. Correct metric and why.
3. Failure modes.
4. Stress test scenarios.
5. Pseudocode for core algorithm.
6. Fallback if advanced model fails.
7. What judges might attack.
</task>

<constraints>
- No UI code.
- No giant research-only solution.
- Keep implementable in less than 10 hours.
</constraints>
```
