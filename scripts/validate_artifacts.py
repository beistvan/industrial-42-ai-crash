#!/usr/bin/env python3
"""Validate pipeline artifact schemas (dev CSVs, metrics JSON, split ids).

Usage:
    python scripts/validate_artifacts.py
    python scripts/validate_artifacts.py --require-ngram
    make validate-artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.schema_validation import SchemaValidationError, validate_all_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-ngram",
        action="store_true",
        help="Fail if artifacts/ngram_metrics.json is missing.",
    )
    parser.add_argument("--json", action="store_true", help="Print report as JSON.")
    args = parser.parse_args()

    try:
        report = validate_all_artifacts(REPO_ROOT, require_ngram_metrics=args.require_ngram)
    except SchemaValidationError as exc:
        raise SystemExit(f"Artifact validation FAILED: {exc}") from exc

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Artifact validation OK — {len(report['checked'])} paths checked")
        for path in report["checked"]:
            print(f"  ✓ {path}")


if __name__ == "__main__":
    main()
