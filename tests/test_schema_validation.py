"""Tests for pipeline artifact schema validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.schema_validation import (
    SchemaValidationError,
    validate_dev_eval_csv,
    validate_local_eval_payload,
    validate_sweep_metrics,
)


@pytest.mark.unit
def test_validate_dev_eval_csv_ok(tmp_path: Path):
    p = tmp_path / "eval.csv"
    p.write_text(
        "SEQUENCE_ID,FAMILY,STEP\n"
        "s1,mosfet,RECEIVE WAFER LOT\n"
        "s1,mosfet,LOT IDENTIFICATION\n",
        encoding="utf-8",
    )
    validate_dev_eval_csv(p)


@pytest.mark.unit
def test_validate_dev_eval_csv_bad_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("ID,FAMILY,STEP\ns1,mosfet,STEP\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="expected columns"):
        validate_dev_eval_csv(p)


@pytest.mark.unit
def test_validate_sweep_metrics_ok():
    payload = {
        "run_name": "test_run",
        "metrics": {
            "task1_next_step": {"overall": {"top1": 0.1, "top5": 0.2, "mrr": 0.15}},
            "task2_completion": {
                "overall": {"token_accuracy": 0.3, "normalized_edit_distance": 0.2},
            },
            "task3_anomaly": {"f1_invalid": 1.0},
        },
        "history": [],
    }
    validate_sweep_metrics(payload)


@pytest.mark.unit
def test_validate_sweep_metrics_missing_task2():
    payload = {
        "run_name": "x",
        "metrics": {
            "task1_next_step": {"overall": {"top1": 0.1, "top5": 0.2, "mrr": 0.15}},
            "task3_anomaly": {"f1_invalid": 1.0},
        },
    }
    with pytest.raises(SchemaValidationError, match="task2"):
        validate_sweep_metrics(payload)


@pytest.mark.unit
def test_validate_local_eval_skipped_payload():
    validate_local_eval_payload({"status": "skipped", "reason": "missing checkpoint"})


@pytest.mark.unit
def test_validate_local_eval_ok():
    validate_local_eval_payload({
        "status": "ok",
        "metrics": {
            "task1_next_step": {"overall": {"top1": 0.1, "top5": 0.2, "mrr": 0.15}},
            "task2_completion": {
                "overall": {"token_accuracy": 0.3, "normalized_edit_distance": 0.2},
            },
            "task3_anomaly": {"f1_invalid": 1.0},
        },
    })


@pytest.mark.unit
def test_validate_anomaly_gold_csv(tmp_path: Path):
    p = tmp_path / "gold.csv"
    p.write_text(
        "SEQUENCE_ID,FAMILY,VALID,INJECTED_LABEL\n"
        "s1,mosfet,1,\n"
        "s2,igbt,0,RULE_X_drop_y\n",
        encoding="utf-8",
    )
    validate_dev_eval_csv(p, schema="anomaly_gold")


@pytest.mark.integration
def test_validate_all_artifacts_on_repo(repo_root: Path):
    """Smoke: dev split CSVs exist in a normal checkout."""
    from src.eval.schema_validation import validate_dev_eval_dir

    eval_dir = repo_root / "data" / "processed" / "dev_eval"
    if not eval_dir.exists():
        pytest.skip("dev eval not built — run make dev-split")
    validate_dev_eval_dir(eval_dir)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
