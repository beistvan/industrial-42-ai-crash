"""Smoke test for the real-data n-gram pipeline.

Verifies that:
  - core docs and the Streamlit demo entry exist,
  - `make dev-split` produced split id files + dev eval inputs from the
    real Infineon training_data,
  - `make train-ngram` produced `artifacts/ngram_metrics.json` with
    Task 1 / Task 2 / Task 3 numbers wired up.
"""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FILES = [
    Path("README.md"),
    Path("src/app/main.py"),
    Path("src/app/eval_dashboard.py"),
    Path("docs/MODEL_PLAN.md"),
    Path("docs/EVALUATION_PLAN.md"),
    Path("data/processed/splits/train_ids.json"),
    Path("data/processed/splits/dev_ids.json"),
    Path("data/processed/dev_eval/eval_input_valid_dev.csv"),
    Path("data/processed/dev_eval/eval_input_valid_dev_gold.csv"),
    Path("data/processed/dev_eval/eval_input_anomaly_dev.csv"),
    Path("data/processed/dev_eval/eval_input_anomaly_dev_gold.csv"),
    Path("artifacts/ngram_metrics.json"),
]


def _fail(msg: str) -> None:
    raise SystemExit(f"Smoke test FAILED: {msg}")


def main() -> None:
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        _fail(f"missing required files: {missing}")

    payload = json.loads(
        Path("artifacts/ngram_metrics.json").read_text(encoding="utf-8")
    )
    metrics = payload.get("metrics", {})
    t1 = metrics.get("task1_next_step", {}).get("overall", {})
    t2 = metrics.get("task2_completion", {}).get("overall", {})
    t3 = metrics.get("task3_anomaly", {})

    if "top1" not in t1:
        _fail("task1_next_step.overall.top1 missing")
    if "token_accuracy" not in t2:
        _fail("task2_completion.overall.token_accuracy missing")
    if "f1_invalid" not in t3:
        _fail("task3_anomaly.f1_invalid missing")

    train_ids = json.loads(
        Path("data/processed/splits/train_ids.json").read_text(encoding="utf-8")
    )
    total_train = sum(len(v) for v in train_ids.values())
    if total_train <= 0:
        _fail("train_ids.json has no sequences")

    print(
        "Smoke test passed:\n"
        f"  train sequences (real data): {total_train}\n"
        f"  Task 1 top-1: {t1.get('top1'):.3f}  top-5: {t1.get('top5'):.3f}\n"
        f"  Task 2 token_acc: {t2.get('token_accuracy'):.3f}  "
        f"edit_dist: {t2.get('normalized_edit_distance'):.3f}\n"
        f"  Task 3 F1(invalid): {t3.get('f1_invalid'):.3f}  "
        f"rule_attr: {t3.get('rule_attribution_accuracy'):.3f}\n"
        "  Note: Task 3 F1=1.000 is expected on this local dev set because "
        "invalid examples are injected rule violations and detection uses the "
        "official rule validator."
    )


if __name__ == "__main__":
    main()
