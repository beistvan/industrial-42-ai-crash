"""Smoke test for the real-data n-gram pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_FILES = [
    Path("README.md"),
    Path("REPORT.md"),
    Path("HANDOFF.md"),
    Path("docs/SUBMISSION.md"),
    Path("src/app/dashboard.py"),
    Path("src/app/dashboard_data.py"),
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

    try:
        from src.eval.schema_validation import validate_all_artifacts

        report = validate_all_artifacts(Path("."), require_ngram_metrics=True)
        print(f"  artifact schema check: {len(report['checked'])} paths OK")
    except Exception as exc:
        _fail(f"artifact schema validation: {exc}")

    print(
        "Smoke test passed:\n"
        f"  train sequences (real data): {total_train}\n"
        f"  Task 1 top-1: {t1.get('top1'):.3f}  top-5: {t1.get('top5'):.3f}\n"
        f"  Task 2 token_acc: {t2.get('token_accuracy'):.3f}  "
        f"NED: {t2.get('normalized_edit_distance'):.3f}\n"
        f"  Task 3 F1(invalid): {t3.get('f1_invalid'):.3f}  "
        f"rule_attr: {t3.get('rule_attribution_accuracy'):.3f}\n"
        "  Note: Task 3 F1=1.000 is expected on this local dev set because "
        "invalid examples are injected rule violations and detection uses the "
        "official rule validator."
    )


if __name__ == "__main__":
    main()
