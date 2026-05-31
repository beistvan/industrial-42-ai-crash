#!/usr/bin/env python3
"""Drive a Transformer training sweep from a YAML matrix.

Usage:
    # Run a single row from the sweep (use with Slurm array jobs, one row per task):
    python scripts/sweep_transformer.py \\
        --sweep configs/sweeps/leonardo_v1.yaml \\
        --stage shortlist \\
        --row 0

    # Run all rows sequentially (local debugging):
    python scripts/sweep_transformer.py \\
        --sweep configs/sweeps/leonardo_v1.yaml \\
        --stage shortlist \\
        --all

    # List planned runs without executing:
    python scripts/sweep_transformer.py \\
        --sweep configs/sweeps/leonardo_v1.yaml \\
        --stage shortlist \\
        --dry-run

Each row inherits `defaults:` and overrides them with its own keys. Mandatory
key per row is `run_name`; all other knobs accepted by `scripts/train_transformer.py`
can be set.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_transformer.py"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "sweeps"
DEFAULT_METRICS_DIR = REPO_ROOT / "artifacts" / "sweeps"


# Knobs that map to train_transformer.py CLI flags. Keep this list in sync
# whenever a new flag is added to train_transformer.py.
KNOB_TO_FLAG = {
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "label_smoothing": "--label-smoothing",
    "scheduler": "--scheduler",
    "warmup_steps": "--warmup-steps",
    "save_best_by": "--save-best-by",
    "eval_task1_every": "--eval-task1-every",
    "eval_task2_every": "--eval-task2-every",
    "d_model": "--d-model",
    "n_layers": "--n-layers",
    "n_heads": "--n-heads",
    "dropout": "--dropout",
    "max_len": "--max-len",
    "dim_feedforward": "--dim-feedforward",
    "seed": "--seed",
    "device": "--device",
    "arch": "--arch",
    "completion_prefix_min": "--completion-prefix-min",
    "completion_prefix_max": "--completion-prefix-max",
    "eval_beam_width": "--eval-beam-width",
    "param_enrichment": "--param-enrichment",
}


def _load_sweep(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "defaults" not in payload:
        payload["defaults"] = {}
    return payload


def _resolve_rows(sweep: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    rows = sweep.get(stage)
    if not rows:
        raise SystemExit(f"Stage {stage!r} not found or empty in sweep file.")
    if isinstance(rows, list) and len(rows) == 1 and rows[0] == []:
        return []
    defaults = dict(sweep.get("defaults", {}))
    resolved: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        merged = dict(defaults)
        merged.update(raw)
        if "run_name" not in merged:
            raise SystemExit(f"Row missing required `run_name`: {raw}")
        resolved.append(merged)
    return resolved


def _build_command(
    row: dict[str, Any],
    *,
    python: str,
    train_script: Path,
    models_dir: Path,
    metrics_dir: Path,
) -> list[str]:
    run_name = row["run_name"]
    model_path = models_dir / f"{run_name}.pt"
    metrics_path = metrics_dir / f"{run_name}.json"
    cmd = [
        python, str(train_script),
        "--config", str(row.get("base_config", "configs/transformer_medium.yaml")),
        "--model-path", str(model_path),
        "--metrics-path", str(metrics_path),
        "--run-name", run_name,
    ]
    extra = row.get("extra_data_dir")
    if extra:
        cmd += ["--extra-data-dir", str(extra)]
    if bool(row.get("amp", False)):
        cmd += ["--amp"]
    else:
        cmd += ["--no-amp"]
    if bool(row.get("eval_rule_constrained", True)):
        cmd += ["--eval-rule-constrained"]
    else:
        cmd += ["--eval-no-rule-constrained"]
    if "eval_candidate_pool" in row:
        cmd += ["--eval-candidate-pool", str(row["eval_candidate_pool"])]
    for key, flag in KNOB_TO_FLAG.items():
        if key in row and row[key] is not None:
            cmd += [flag, str(row[key])]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--stage", default="shortlist")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--row", type=int, help="0-indexed row of the stage to run.")
    grp.add_argument("--all", action="store_true",
                     help="Run all rows sequentially (debug only).")
    grp.add_argument("--dry-run", action="store_true",
                     help="Print planned commands; do not execute.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to invoke (default: current).")
    ap.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    args = ap.parse_args()

    sweep = _load_sweep(args.sweep)
    rows = _resolve_rows(sweep, args.stage)
    if not rows:
        print(f"No rows in stage {args.stage!r} — nothing to do.")
        return

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.row is not None:
        if not (0 <= args.row < len(rows)):
            raise SystemExit(f"--row {args.row} out of range (0..{len(rows)-1})")
        target_rows = [rows[args.row]]
    else:
        target_rows = rows

    for idx, row in enumerate(target_rows):
        cmd = _build_command(
            row,
            python=args.python,
            train_script=args.train_script,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
        )
        print(f"[{idx+1}/{len(target_rows)}] run={row['run_name']!r}")
        print("    " + " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            raise SystemExit(
                f"Run {row['run_name']!r} failed (exit {result.returncode}). "
                "Check sweep log; subsequent rows skipped."
            )


if __name__ == "__main__":
    main()
