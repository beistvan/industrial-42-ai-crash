from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data import FAMILIES, load_all_families


def _seq_count(path: Path) -> int:
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ids.add(row["SEQUENCE_ID"])
    return len(ids)


@pytest.mark.integration
def test_infineon_raw_data_stats():
    sequences = load_all_families()
    assert set(sequences) == set(FAMILIES)
    assert {fam: len(sequences[fam]) for fam in FAMILIES} == {
        "mosfet": 1000,
        "igbt": 1000,
        "ic": 1000,
    }
    assert sum(len(s) for s in sequences["mosfet"].values()) == 125260
    assert sum(len(s) for s in sequences["igbt"].values()) == 147959
    assert sum(len(s) for s in sequences["ic"].values()) == 115075


@pytest.mark.integration
def test_make_dev_split_outputs_current_eval_files(tmp_path):
    split_dir = tmp_path / "splits"
    eval_dir = tmp_path / "dev_eval"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_dev_split.py",
            "--split-dir",
            str(split_dir),
            "--eval-dir",
            str(eval_dir),
            "--dev-per-family",
            "100",
            "--seed",
            "42",
        ],
        check=True,
    )

    train_ids = json.loads((split_dir / "train_ids.json").read_text(encoding="utf-8"))
    dev_ids = json.loads((split_dir / "dev_ids.json").read_text(encoding="utf-8"))
    assert {fam: len(train_ids[fam]) for fam in FAMILIES} == {"mosfet": 900, "igbt": 900, "ic": 900}
    assert {fam: len(dev_ids[fam]) for fam in FAMILIES} == {"mosfet": 100, "igbt": 100, "ic": 100}

    valid_input = eval_dir / "eval_input_valid_dev.csv"
    valid_gold = eval_dir / "eval_input_valid_dev_gold.csv"
    anomaly_input = eval_dir / "eval_input_anomaly_dev.csv"
    anomaly_gold = eval_dir / "eval_input_anomaly_dev_gold.csv"
    assert _seq_count(valid_input) == 600
    assert _seq_count(valid_gold) == 600
    assert _seq_count(anomaly_input) == 600

    with anomaly_gold.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 600
    assert sum(1 for r in rows if r["VALID"] == "1") == 300
    assert sum(1 for r in rows if r["VALID"] == "0") == 300
    assert all(r["DETECTED_RULES"] for r in rows if r["VALID"] == "0")
