"""Lightweight schema checks for pipeline artifacts.

Validates dev-eval CSVs, sweep metrics JSON, and local-eval payloads so format
drift is caught before Slurm sweeps or submission regen — no jsonschema dep.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEV_EVAL_LONG_COLUMNS = frozenset({"SEQUENCE_ID", "FAMILY", "STEP"})
DEV_ANOMALY_GOLD_COLUMNS = frozenset({"SEQUENCE_ID", "FAMILY", "VALID"})
TASK1_OVERALL_KEYS = frozenset({"top1", "top5", "mrr"})
TASK2_OVERALL_KEYS = frozenset({"token_accuracy", "normalized_edit_distance"})
TASK3_KEYS = frozenset({"f1_invalid"})


class SchemaValidationError(ValueError):
    """Raised when an artifact fails a schema check."""


def _require_keys(obj: dict[str, Any], keys: frozenset[str], label: str) -> None:
    missing = keys - set(obj.keys())
    if missing:
        raise SchemaValidationError(f"{label}: missing keys {sorted(missing)}")


def validate_dev_eval_csv(path: Path, *, schema: str = "long") -> None:
    """Validate dev eval CSV. ``long`` = SEQUENCE_ID,FAMILY,STEP; ``anomaly_gold`` = labels."""
    if not path.exists():
        raise SchemaValidationError(f"dev eval CSV not found: {path}")
    required = DEV_ANOMALY_GOLD_COLUMNS if schema == "anomaly_gold" else DEV_EVAL_LONG_COLUMNS
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SchemaValidationError(f"{path}: empty or headerless CSV")
        cols = {c.strip() for c in reader.fieldnames if c}
        if not required.issubset(cols):
            raise SchemaValidationError(
                f"{path}: expected columns {sorted(required)}, got {sorted(cols)}"
            )
        row_count = 0
        for row in reader:
            row_count += 1
            for key in required:
                if key == "VALID":
                    val = row.get(key)
                    if val is None or str(val).strip() == "":
                        raise SchemaValidationError(f"{path}: blank VALID at row {row_count + 1}")
                elif not (row.get(key) or "").strip():
                    raise SchemaValidationError(f"{path}: blank {key} at row {row_count + 1}")
        if row_count == 0:
            raise SchemaValidationError(f"{path}: no data rows")


def validate_sweep_metrics(payload: dict[str, Any], *, path: str = "") -> None:
    """Sweep run metrics JSON written by train_transformer.py."""
    label = path or "sweep metrics"
    if not payload.get("run_name"):
        raise SchemaValidationError(f"{label}: missing run_name")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise SchemaValidationError(f"{label}: missing metrics object")
    t1 = (metrics.get("task1_next_step") or {}).get("overall")
    t2 = (metrics.get("task2_completion") or {}).get("overall")
    t3 = metrics.get("task3_anomaly")
    if not isinstance(t1, dict):
        raise SchemaValidationError(f"{label}: missing metrics.task1_next_step.overall")
    if not isinstance(t2, dict):
        raise SchemaValidationError(f"{label}: missing metrics.task2_completion.overall")
    if not isinstance(t3, dict):
        raise SchemaValidationError(f"{label}: missing metrics.task3_anomaly")
    _require_keys(t1, TASK1_OVERALL_KEYS, f"{label}.task1")
    _require_keys(t2, TASK2_OVERALL_KEYS, f"{label}.task2")
    _require_keys(t3, TASK3_KEYS, f"{label}.task3")
    history = payload.get("history")
    if history is not None and not isinstance(history, list):
        raise SchemaValidationError(f"{label}: history must be a list")


def validate_local_eval_payload(payload: dict[str, Any], *, path: str = "") -> None:
    """Local eval JSON from src.eval.local_eval or eval matrix runner."""
    label = path or "local eval"
    if payload.get("status") == "skipped":
        if "reason" not in payload:
            raise SchemaValidationError(f"{label}: skipped payload missing reason")
        return
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise SchemaValidationError(f"{label}: missing metrics object")
    validate_sweep_metrics({"run_name": "local", "metrics": metrics, "history": []}, path=label)


def validate_split_ids(path: Path) -> None:
    if not path.exists():
        raise SchemaValidationError(f"split ids not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise SchemaValidationError(f"{path}: expected non-empty family -> ids mapping")


def validate_dev_eval_dir(eval_dir: Path) -> list[str]:
    """Validate all four dev-eval CSVs. Returns list of validated paths."""
    required = [
        ("eval_input_valid_dev.csv", "long"),
        ("eval_input_valid_dev_gold.csv", "long"),
        ("eval_input_anomaly_dev.csv", "long"),
        ("eval_input_anomaly_dev_gold.csv", "anomaly_gold"),
    ]
    validated: list[str] = []
    for name, schema in required:
        p = eval_dir / name
        validate_dev_eval_csv(p, schema=schema)
        validated.append(str(p))
    return validated


def validate_all_artifacts(
    repo_root: Path | None = None,
    *,
    require_ngram_metrics: bool = False,
) -> dict[str, Any]:
    """Run all artifact checks. Raises SchemaValidationError on first failure."""
    root = repo_root or REPO_ROOT
    report: dict[str, Any] = {"status": "ok", "checked": []}

    splits = root / "data" / "processed" / "splits"
    for name in ("train_ids.json", "dev_ids.json"):
        validate_split_ids(splits / name)
        report["checked"].append(str(splits / name))

    eval_dir = root / "data" / "processed" / "dev_eval"
    report["checked"].extend(validate_dev_eval_dir(eval_dir))

    ngram_path = root / "artifacts" / "ngram_metrics.json"
    if ngram_path.exists():
        payload = json.loads(ngram_path.read_text(encoding="utf-8"))
        validate_local_eval_payload(payload, path=str(ngram_path))
        report["checked"].append(str(ngram_path))
    elif require_ngram_metrics:
        raise SchemaValidationError(f"missing {ngram_path}")

    sweeps_dir = root / "artifacts" / "sweeps"
    if sweeps_dir.is_dir():
        for path in sorted(sweeps_dir.glob("*.json")):
            if path.name.startswith("LEADERBOARD"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"{path}: invalid JSON: {exc}") from exc
            validate_sweep_metrics(payload, path=str(path))
            report["checked"].append(str(path))

    manifest = root / "data" / "generated" / "infineon" / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"{manifest}: expected object")
        report["checked"].append(str(manifest))

    return report
