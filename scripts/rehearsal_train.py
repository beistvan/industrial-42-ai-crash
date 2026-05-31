#!/usr/bin/env python3
"""Pre-flight training rehearsal — catch pipeline bugs before Slurm sweeps.

Runs a tiny Transformer for 1–2 epochs on ~10% of the train split, evaluates
on a capped dev set, validates the metrics JSON schema, and optionally checks
dev-eval CSVs. Intended to finish in ~5 minutes on one GPU (or ~15 min CPU).

Usage:
    python scripts/rehearsal_train.py
    python scripts/rehearsal_train.py --device cuda --epochs 2
    make rehearsal-train
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.schema_validation import (  # noqa: E402
    SchemaValidationError,
    validate_all_artifacts,
    validate_sweep_metrics,
)

DEFAULT_MODEL = REPO_ROOT / "models" / "rehearsal.pt"
DEFAULT_METRICS = REPO_ROOT / "artifacts" / "rehearsal_metrics.json"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_transformer.py"


def _ensure_dev_split(python: str) -> None:
    split = REPO_ROOT / "data" / "processed" / "splits" / "train_ids.json"
    if split.exists():
        return
    print("[rehearsal] dev split missing — running make_dev_split.py")
    subprocess.run(
        [python, str(REPO_ROOT / "scripts" / "make_dev_split.py"), "--force"],
        cwd=REPO_ROOT,
        check=True,
    )


def _run_training(args: argparse.Namespace) -> None:
    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        "--run-name",
        "rehearsal",
        "--limit-train-sequences",
        str(args.limit_train),
        "--limit-dev-sequences",
        str(args.limit_dev),
        "--epochs",
        str(args.epochs),
        "--d-model",
        "32",
        "--n-layers",
        "2",
        "--n-heads",
        "4",
        "--dim-feedforward",
        "64",
        "--batch-size",
        "8",
        "--eval-task1-every",
        "1",
        "--eval-task2-every",
        "1",
        "--model-path",
        str(DEFAULT_MODEL),
        "--metrics-path",
        str(DEFAULT_METRICS),
        "--device",
        args.device,
    ]
    if args.device == "cpu":
        cmd.extend(["--num-workers", "0"])
    print("[rehearsal] training:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=50,
        help="Max train sequences per family (~10%% of 1150).",
    )
    parser.add_argument(
        "--limit-dev",
        type=int,
        default=20,
        help="Max dev sequences per family for in-loop eval.",
    )
    parser.add_argument(
        "--skip-artifact-check",
        action="store_true",
        help="Skip full dev-eval CSV validation after training.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    _ensure_dev_split(args.python)

    if not args.skip_artifact_check:
        try:
            validate_all_artifacts(REPO_ROOT, require_ngram_metrics=False)
            print("[rehearsal] dev-eval CSVs and split ids OK")
        except SchemaValidationError as exc:
            raise SystemExit(f"[rehearsal] pre-flight artifact check failed: {exc}") from exc

    _run_training(args)

    if not DEFAULT_METRICS.exists():
        raise SystemExit(f"[rehearsal] missing metrics output: {DEFAULT_METRICS}")

    payload = json.loads(DEFAULT_METRICS.read_text(encoding="utf-8"))
    validate_sweep_metrics(payload, path=str(DEFAULT_METRICS))

    t1 = payload.get("metrics", {}).get("task1_next_step", {}).get("overall", {})
    t2 = payload.get("metrics", {}).get("task2_completion", {}).get("overall", {})
    elapsed = time.perf_counter() - t0
    print(
        f"[rehearsal] PASSED in {elapsed:.0f}s\n"
        f"  model:  {DEFAULT_MODEL}\n"
        f"  metrics: {DEFAULT_METRICS}\n"
        f"  task1 mrr={t1.get('mrr', 0):.4f} top1={t1.get('top1', 0):.4f}\n"
        f"  task2 tok={t2.get('token_accuracy', 0):.4f} "
        f"ned={t2.get('normalized_edit_distance', 0):.4f}\n"
        "  Safe to submit Slurm sweep array."
    )


if __name__ == "__main__":
    main()
