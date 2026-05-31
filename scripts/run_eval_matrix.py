#!/usr/bin/env python3
"""Run the 4-arm eval matrix (baseline / T1 / T2 / hybrid submission).

Usage:
    python scripts/run_eval_matrix.py
    python scripts/run_eval_matrix.py --device cuda --out artifacts/eval_matrix.json
    make eval-matrix
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.eval_matrix import write_eval_matrix  # noqa: E402
from src.eval.schema_validation import SchemaValidationError, validate_dev_eval_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "dev_eval",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: artifacts/eval_matrix-{timestamp}.json).",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda for Transformer arms.")
    args = parser.parse_args()

    try:
        validate_dev_eval_dir(args.eval_dir)
    except SchemaValidationError as exc:
        raise SystemExit(
            f"Dev eval inputs invalid: {exc}\nRun: python scripts/make_dev_split.py --force"
        ) from exc

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (REPO_ROOT / "artifacts" / f"eval_matrix-{ts}.json")

    payload = write_eval_matrix(
        args.eval_dir,
        out,
        device=args.device,
        repo_root=REPO_ROOT,
    )
    payload["generated_at"] = ts

    latest = REPO_ROOT / "artifacts" / "eval_matrix_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {out}")

    for arm in payload["arms"]:
        status = arm["status"]
        m = arm.get("metrics") or {}
        t1 = (m.get("task1_next_step") or {}).get("overall", {})
        t2 = (m.get("task2_completion") or {}).get("overall", {})
        if status == "unavailable":
            print(f"  arm {arm['arm_id']} ({arm['name']}): unavailable — {arm.get('reason')}")
        else:
            print(
                f"  arm {arm['arm_id']} ({arm['name']}): {status}  "
                f"mrr={t1.get('mrr', 0):.4f}  "
                f"ned={t2.get('normalized_edit_distance', 0):.4f}  "
                f"({arm.get('runtime_s')}s)"
            )


if __name__ == "__main__":
    main()
