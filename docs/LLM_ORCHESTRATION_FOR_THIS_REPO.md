# LLM Orchestration for `industrial-models-that-learn-how-processes-unfold`

## Repo identity

# Industrial / Infineon Repo Operating Brief

Repository: `industrial-models-that-learn-how-processes-unfold`
App: Process Sequence Model
Port: backend `8001`, demo UI `8501`

## Winning problem statement
Build a working application that can learn process evolution from event/sensor sequences using a real model/pipeline, a visible evaluation, and a demo that survives judge questions.

## Best target architecture
- `src/data/`: ingest, validate, split, and synthesize fallback data.
- `src/features/`: feature extraction and schema conversion.
- `src/models/`: baseline model, candidate model, training loop, checkpoint loading.
- `src/eval/`: metrics, stress tests, ablation, judge-readable reports.
- `src/app/`: FastAPI or Streamlit demo layer.
- `docs/`: model card, dataset card, evaluation card, pitch notes.
- `runs/`: generated metrics and plots, never hand-edit.

## Best first implementation
1. Make data contract real.
2. Make baseline train/evaluate end-to-end.
3. Add one stronger model that can use A100 power if provided.
4. Add stress-test/evaluation report.
5. Add demo UI that shows model output, uncertainty/failure mode, and action.

## Model direction
Transformer encoder / TCN / hybrid state-space fallback with masked-step prediction

## Baseline
rolling statistics + GRU/1D-CNN baseline

## Demo shape
process timeline viewer with predicted next steps, drift warning, anomaly heatmap, and what-changed explanation

## Evaluation
next-step accuracy, masked reconstruction loss, anomaly AUROC, early-warning lead time, inference latency

## Judge value
Judge sees a real trained sequence model that predicts/understands process dynamics instead of a static classifier.


## Best LLM usage order before the hackathon

1. ChatGPT web: refine scope and write `docs/JUDGE_CARD.md`.
2. Gemini web: audit repo structure and expected data contract.
3. DeepSeek web: critique algorithm and stress tests.
4. Claude Code: implement missing skeleton, scripts, and docs.
5. Codex: run setup/tests and fix environment issues.
6. Copilot: add unit tests, docstrings, and UI helpers.

## Best LLM usage order during the hackathon

1. Gemini web reads real data/brief and creates `docs/GEMINI_DATA_AUDIT.md`.
2. ChatGPT web converts that into `docs/FINAL_SCOPE.md`.
3. DeepSeek web stress-tests metric/model assumptions.
4. Claude Code implements baseline and demo.
5. Codex keeps the project running.
6. Copilot fills tests and small components.

## Exact first agent prompt

Use `prompts/claude_code_track_specific_v2.md` in Claude Code.

## Exact test-runner prompt

Use `prompts/codex_track_specific_v2.md` in Codex.
