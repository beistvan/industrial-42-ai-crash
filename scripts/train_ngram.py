#!/usr/bin/env python3
"""Fit the n-gram baseline on the train split and evaluate on the dev split.

Requires `scripts/make_dev_split.py` to have been run first, so the split
id files and dev eval CSVs exist under `data/processed/`.

Usage:
    python scripts/make_dev_split.py
    python scripts/train_ngram.py
    python scripts/train_ngram.py --max-order 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import FAMILIES, load_all_families  # noqa: E402
from src.eval.run_eval import evaluate_all  # noqa: E402
from src.ml import NGramBaseline  # noqa: E402


def _load_split(split_dir: Path):
    train_ids = json.loads((split_dir / "train_ids.json").read_text(encoding="utf-8"))
    all_seqs = load_all_families()
    train = {fam: {sid: all_seqs[fam][sid] for sid in train_ids.get(fam, [])}
             for fam in FAMILIES}
    return train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-order", type=int, default=12)
    ap.add_argument("--split-dir", type=Path,
                    default=REPO_ROOT / "data" / "processed" / "splits")
    ap.add_argument("--eval-dir", type=Path,
                    default=REPO_ROOT / "data" / "processed" / "dev_eval")
    ap.add_argument("--model-path", type=Path,
                    default=REPO_ROOT / "models" / "ngram_baseline.pkl")
    ap.add_argument("--metrics-path", type=Path,
                    default=REPO_ROOT / "artifacts" / "ngram_metrics.json")
    args = ap.parse_args()

    if not (args.split_dir / "train_ids.json").exists():
        raise SystemExit(
            f"Missing {args.split_dir/'train_ids.json'}. "
            f"Run: python scripts/make_dev_split.py"
        )

    print(f"[1/3] Loading split from {args.split_dir}")
    train = _load_split(args.split_dir)
    print("      train: " + ", ".join(f"{f}={len(s)}" for f, s in train.items()))

    print(f"[2/3] Fitting NGramBaseline(max_order={args.max_order})")
    t0 = time.time()
    model = NGramBaseline(max_order=args.max_order).fit(train)
    fit_seconds = time.time() - t0
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"      fit in {fit_seconds:.2f}s -> {args.model_path}")

    print(f"[3/3] Evaluating on dev eval inputs from {args.eval_dir}")
    t0 = time.time()
    metrics = evaluate_all(model, args.eval_dir)
    eval_seconds = time.time() - t0
    payload = {
        "model": model.stats(),
        "fit_seconds": round(fit_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "metrics": metrics.to_dict(),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nMetrics ({eval_seconds:.1f}s) -> {args.metrics_path}")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
