#!/usr/bin/env python3
"""Build the local dev split and dev eval inputs.

Loads all three families from `data/raw/infineon/training_data/`,
deterministically holds out `--dev-per-family` sequences per family, and
writes:
  - data/processed/splits/train_ids.json
  - data/processed/splits/dev_ids.json
  - data/processed/splits/split_summary.json
  - data/processed/dev_eval/eval_input_valid_dev.csv
  - data/processed/dev_eval/eval_input_valid_dev_gold.csv
  - data/processed/dev_eval/eval_input_anomaly_dev.csv
  - data/processed/dev_eval/eval_input_anomaly_dev_gold.csv

This script is intentionally independent from the model implementation. It is
safe to run while baseline/transformer files are being edited, because it only
writes derived files under data/processed/.

Usage:
    python scripts/make_dev_split.py
    python scripts/make_dev_split.py --force
    python scripts/make_dev_split.py --dev-per-family 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import deterministic_split, load_all_families  # noqa: E402
from src.eval.dev_split import build_dev_eval  # noqa: E402


def _fail_if_outputs_exist(paths: list[Path], *, force: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        joined = "\n  - ".join(str(p) for p in existing)
        raise FileExistsError(
            "Refusing to overwrite existing dev-split outputs. "
            "Pass --force to regenerate them intentionally:\n  - " + joined
        )


def _row_count(csv_path: Path) -> int:
    # Excludes header. Cheap sanity count without requiring pandas.
    with csv_path.open(encoding="utf-8-sig") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-per-family", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-dir", type=Path,
                    default=REPO_ROOT / "data" / "processed" / "splits")
    ap.add_argument("--eval-dir", type=Path,
                    default=REPO_ROOT / "data" / "processed" / "dev_eval")
    ap.add_argument("--anomaly-seed", type=int, default=17)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing derived split/eval files.")
    args = ap.parse_args()

    expected_outputs = [
        args.split_dir / "train_ids.json",
        args.split_dir / "dev_ids.json",
        args.split_dir / "split_summary.json",
        args.eval_dir / "eval_input_valid_dev.csv",
        args.eval_dir / "eval_input_valid_dev_gold.csv",
        args.eval_dir / "eval_input_anomaly_dev.csv",
        args.eval_dir / "eval_input_anomaly_dev_gold.csv",
    ]
    _fail_if_outputs_exist(expected_outputs, force=args.force)

    print("[1/3] Loading sequences from data/raw/infineon/training_data/")
    all_seqs = load_all_families()
    print("      " + ", ".join(f"{f}={len(s)}" for f, s in all_seqs.items()))

    print(f"[2/3] Splitting (dev_per_family={args.dev_per_family}, seed={args.seed})")
    split = deterministic_split(all_seqs, dev_per_family=args.dev_per_family,
                                seed=args.seed)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    train_ids = {fam: sorted(s.keys()) for fam, s in split.train.items()}
    dev_ids = {fam: sorted(s.keys()) for fam, s in split.dev.items()}
    (args.split_dir / "train_ids.json").write_text(
        json.dumps(train_ids, indent=2), encoding="utf-8")
    (args.split_dir / "dev_ids.json").write_text(
        json.dumps(dev_ids, indent=2), encoding="utf-8")
    print(f"      wrote {args.split_dir}/train_ids.json, dev_ids.json")
    print("      train: " + ", ".join(f"{f}={n}" for f, n in split.train_lengths().items()))
    print("      dev  : " + ", ".join(f"{f}={n}" for f, n in split.dev_lengths().items()))

    print(f"[3/3] Building dev eval inputs at {args.eval_dir}")
    paths, counts = build_dev_eval(split.dev, args.eval_dir,
                                   anomaly_seed=args.anomaly_seed)
    summary = {
        "seed": args.seed,
        "anomaly_seed": args.anomaly_seed,
        "dev_per_family": args.dev_per_family,
        "train_counts": split.train_lengths(),
        "dev_counts": split.dev_lengths(),
        "valid_input_rows": _row_count(paths.valid_input),
        "valid_gold_rows": _row_count(paths.valid_gold),
        "anomaly_counts": counts,
        "paths": {
            "train_ids": str(args.split_dir / "train_ids.json"),
            "dev_ids": str(args.split_dir / "dev_ids.json"),
            "valid_input": str(paths.valid_input),
            "valid_gold": str(paths.valid_gold),
            "anomaly_input": str(paths.anomaly_input),
            "anomaly_gold": str(paths.anomaly_gold),
        },
    }
    (args.split_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"      anomaly: {counts}")
    print(f"      summary: {args.split_dir/'split_summary.json'}")
    print(f"      tasks 1/2 input: {paths.valid_input}")
    print(f"      tasks 1/2 gold : {paths.valid_gold}")
    print(f"      task 3   input: {paths.anomaly_input}")
    print(f"      task 3   gold : {paths.anomaly_gold}")


if __name__ == "__main__":
    main()
