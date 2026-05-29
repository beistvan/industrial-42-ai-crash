"""CLI wrapper for local dev evaluation.

The primary evaluator for implemented models is `src.eval.run_eval`, which
scores a model object with `predict_topk()` and `complete()`. This file provides
a lightweight command-line entrypoint for saved n-gram or Transformer models, so hackathon
runs can be reproduced with one command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.run_eval import evaluate_all  # noqa: E402
from src.ml import load_sequence_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=REPO_ROOT / "models" / "ngram_baseline.pkl",
                        help="Saved model: .pkl n-gram or .pt Transformer checkpoint.")
    parser.add_argument("--eval-dir", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "dev_eval",
                        help="Directory created by scripts/make_dev_split.py.")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / "local_eval_metrics.json")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Missing model file: {args.model}\n"
            "Run: python scripts/train_ngram.py"
        )
    required = [
        args.eval_dir / "eval_input_valid_dev.csv",
        args.eval_dir / "eval_input_valid_dev_gold.csv",
        args.eval_dir / "eval_input_anomaly_dev.csv",
        args.eval_dir / "eval_input_anomaly_dev_gold.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing dev-eval files:\n  - " + "\n  - ".join(missing) +
            "\nRun: python scripts/make_dev_split.py --force"
        )

    model = load_sequence_model(args.model)
    metrics = evaluate_all(model, args.eval_dir).to_dict()
    stats = model.stats() if hasattr(model, "stats") else {"model": type(model).__name__}
    payload = {"model": stats, "metrics": metrics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
