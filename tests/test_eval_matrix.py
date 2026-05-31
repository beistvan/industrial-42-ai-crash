"""Tests for the 4-arm eval matrix runner."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.eval_matrix import (
    ArmSpec,
    default_arm_specs,
    matrix_comparison_table,
    run_arm,
)


@pytest.mark.unit
def test_default_arm_specs_four_arms():
    specs = default_arm_specs()
    assert len(specs) == 4
    assert [s.arm_id for s in specs] == ["A", "B", "C", "D"]


@pytest.mark.unit
def test_run_arm_unavailable_when_checkpoint_missing(tmp_path: Path):
    eval_dir = tmp_path / "dev_eval"
    eval_dir.mkdir()
    for name in (
        "eval_input_valid_dev.csv",
        "eval_input_valid_dev_gold.csv",
        "eval_input_anomaly_dev.csv",
        "eval_input_anomaly_dev_gold.csv",
    ):
        (eval_dir / name).write_text(
            "SEQUENCE_ID,FAMILY,STEP\nx,mosfet,RECEIVE WAFER LOT\n",
            encoding="utf-8",
        )

    spec = ArmSpec(
        arm_id="Z",
        name="missing",
        description="test",
        model_paths={"all": tmp_path / "no_such_model.pt"},
    )
    result = run_arm(spec, eval_dir, device="cpu")
    assert result.status == "unavailable"
    assert result.metrics is None
    assert result.reason


@pytest.mark.unit
def test_matrix_comparison_table_shape():
    payload = {
        "arms": [
            {
                "arm_id": "A",
                "name": "baseline",
                "status": "ok",
                "runtime_s": 1.0,
                "metrics": {
                    "task1_next_step": {"overall": {"mrr": 0.8, "top1": 0.7}},
                    "task2_completion": {
                        "overall": {"token_accuracy": 0.4, "normalized_edit_distance": 0.22},
                    },
                    "task3_anomaly": {"f1_invalid": 1.0, "rule_attribution_accuracy": 0.66},
                },
            },
            {
                "arm_id": "B",
                "name": "t1",
                "status": "unavailable",
                "metrics": None,
            },
        ],
    }
    rows = matrix_comparison_table(payload)
    assert len(rows) == 2
    assert rows[0]["arm"] == "A"
    assert rows[0]["task2_ned"] == 0.22
    assert rows[1]["status"] == "unavailable"
    assert rows[1]["task1_mrr"] is None
