"""Tests for the sweep driver — config resolution only (no real training)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

scripts_dir = REPO_ROOT / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import sweep_transformer as sw  # noqa: E402


@pytest.mark.unit
def test_resolve_rows_merges_defaults():
    sweep = {
        "defaults": {"epochs": 10, "scheduler": "cosine", "amp": True},
        "shortlist": [
            {"run_name": "a"},
            {"run_name": "b", "epochs": 20, "scheduler": "none"},
        ],
    }
    rows = sw._resolve_rows(sweep, "shortlist")
    assert len(rows) == 2
    assert rows[0]["run_name"] == "a"
    assert rows[0]["epochs"] == 10
    assert rows[0]["scheduler"] == "cosine"
    assert rows[1]["epochs"] == 20
    assert rows[1]["scheduler"] == "none"
    # Override applies cleanly without polluting first row.
    assert rows[0]["scheduler"] == "cosine"


@pytest.mark.unit
def test_resolve_rows_requires_run_name():
    sweep = {"defaults": {}, "shortlist": [{"epochs": 1}]}
    with pytest.raises(SystemExit):
        sw._resolve_rows(sweep, "shortlist")


@pytest.mark.unit
def test_build_command_includes_essentials(tmp_path):
    row = {
        "run_name": "x",
        "base_config": "configs/transformer_medium.yaml",
        "extra_data_dir": "data/generated/infineon",
        "epochs": 5,
        "amp": True,
        "scheduler": "cosine",
        "warmup_steps": 100,
        "label_smoothing": 0.1,
    }
    cmd = sw._build_command(
        row,
        python="python3",
        train_script=Path("scripts/train_transformer.py"),
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
    )
    assert "--run-name" in cmd
    assert "x" in cmd
    assert "--amp" in cmd
    assert "--no-amp" not in cmd
    assert "--scheduler" in cmd
    assert "--label-smoothing" in cmd
    assert "--extra-data-dir" in cmd
    assert "--epochs" in cmd


@pytest.mark.unit
def test_build_command_no_amp_disables_flag(tmp_path):
    row = {"run_name": "y", "amp": False}
    cmd = sw._build_command(
        row,
        python="python3",
        train_script=Path("scripts/train_transformer.py"),
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
    )
    assert "--no-amp" in cmd
    assert "--amp" not in cmd


@pytest.mark.unit
def test_committed_sweep_yaml_parses_and_has_unique_names():
    """The on-disk sweep YAML must parse and every row needs a unique run_name."""
    path = REPO_ROOT / "configs" / "sweeps" / "leonardo_v1.yaml"
    if not path.exists():
        pytest.skip("Sweep YAML not present.")
    sweep = sw._load_sweep(path)
    shortlist = sw._resolve_rows(sweep, "shortlist")
    assert len(shortlist) >= 3, "shortlist should have multiple configs"
    names = [r["run_name"] for r in shortlist]
    assert len(names) == len(set(names)), f"Duplicate run_names: {names}"
