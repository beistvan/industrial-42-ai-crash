#!/usr/bin/env python3
"""Scan sweep metrics JSONs and emit a leaderboard.

Usage:
    python scripts/summarize_runs.py --metrics-dir artifacts/sweeps \\
        --out artifacts/sweeps/LEADERBOARD.md

Picks per-run "best" task1 from the in-loop eval history and (if present) the
final Task 2 numbers. Sorts by `dev_mrr` descending.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _row_from_payload(name: str, payload: dict) -> dict:
    history = payload.get("history") or []
    task1_best = {"epoch": None, "top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0}
    task2_best = {"epoch": None, "token_accuracy": 0.0, "normalized_edit_distance": 0.0}
    for ep in history:
        t1 = (ep or {}).get("task1")
        if t1 and t1.get("mrr", 0.0) > task1_best["mrr"]:
            task1_best = {**t1, "epoch": ep["epoch"]}
        t2 = (ep or {}).get("task2")
        if t2 and t2.get("token_accuracy", 0.0) > task2_best["token_accuracy"]:
            task2_best = {
                "token_accuracy": t2.get("token_accuracy", 0.0),
                "normalized_edit_distance": t2.get("normalized_edit_distance", 0.0),
                "epoch": ep["epoch"],
            }
    final = (payload.get("metrics") or {})
    t2_final = ((final.get("task2_completion") or {}).get("overall")) or {}
    t3 = (final.get("task3_anomaly")) or {}
    best = payload.get("best") or {}
    best_metric = best.get("metric")
    best_value = best.get("value")

    task1_mrr = float(task1_best.get("mrr", 0.0) or 0.0)
    if best_metric == "dev_mrr" and best_value is not None:
        task1_mrr = max(task1_mrr, float(best_value))

    task2_tok = float(task2_best.get("token_accuracy", 0.0) or 0.0)
    if not task2_tok and t2_final:
        task2_tok = float(t2_final.get("token_accuracy", 0.0) or 0.0)
    if best_metric == "dev_token_acc" and best_value is not None:
        task2_tok = max(task2_tok, float(best_value))

    task2_ned = float(task2_best.get("normalized_edit_distance", 0.0) or 0.0)
    if not task2_ned and t2_final:
        task2_ned = float(t2_final.get("normalized_edit_distance", 0.0) or 0.0)

    ckpt_path = payload.get("best", {}).get("path")
    checkpoint_status = "unknown"
    if ckpt_path:
        checkpoint_status = "ok" if Path(ckpt_path).exists() else "missing"

    return {
        "run": name,
        "best_metric": best_metric,
        "best_value": best_value,
        "best_epoch": best.get("epoch"),
        "checkpoint_status": checkpoint_status,
        "task1_best_epoch": task1_best["epoch"],
        "task1_top1": round(task1_best.get("top1", 0.0) or 0.0, 4),
        "task1_top5": round(task1_best.get("top5", 0.0) or 0.0, 4),
        "task1_mrr": round(task1_mrr, 4),
        "task2_token_acc": round(task2_tok, 4),
        "task2_ned": round(task2_ned, 4),
        "task3_f1": round(t3.get("f1_invalid", 0.0) or 0.0, 4),
        "task3_rule_attr": round(t3.get("rule_attribution_accuracy", 0.0) or 0.0, 4),
        "train_seconds": payload.get("train_seconds"),
        "extras": payload.get("extra_data_dir"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", type=Path,
                    default=REPO_ROOT / "artifacts" / "sweeps")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts" / "sweeps" / "LEADERBOARD.md")
    ap.add_argument("--csv", type=Path,
                    default=REPO_ROOT / "artifacts" / "sweeps" / "LEADERBOARD.csv")
    ap.add_argument("--sort", default="task1_mrr",
                    help="Column to sort by (descending).")
    args = ap.parse_args()

    if not args.metrics_dir.exists():
        raise SystemExit(f"No metrics dir: {args.metrics_dir}")

    rows: list[dict] = []
    for path in sorted(args.metrics_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: skipping {path.name}: {exc}")
            continue
        name = payload.get("run_name") or path.stem
        rows.append(_row_from_payload(name, payload))

    if not rows:
        raise SystemExit(f"No metrics JSON files found under {args.metrics_dir}.")

    rows.sort(key=lambda r: r.get(args.sort) or 0.0, reverse=True)

    # CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md = ["# Sweep leaderboard\n",
          f"Sorted by `{args.sort}` (descending). {len(rows)} runs.\n",
          ""]
    headers = ["run", "task1_top1", "task1_top5", "task1_mrr",
               "task2_token_acc", "task2_ned", "task3_rule_attr",
               "checkpoint_status", "best_value", "best_epoch", "train_seconds", "extras"]
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {args.csv}")
    print(f"wrote {args.out}")
    print(f"\nTop 5 by {args.sort}:")
    for r in rows[:5]:
        print(f"  {r['run']:30s}  {args.sort}={r.get(args.sort)}  "
              f"top1={r['task1_top1']}  mrr={r['task1_mrr']}  "
              f"tok_acc={r['task2_token_acc']}  ned={r['task2_ned']}")

    # Submission picks (uses checkpoint best_value where applicable)
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.sweep_picks import pick_task1_row, pick_task2_row, score_task1, score_task2

    t1 = pick_task1_row(rows)
    t2 = pick_task2_row(rows)
    print(f"\nSubmission picks:")
    print(f"  T1: {t1['run']}  MRR={score_task1(t1):.4f}")
    print(f"  T2: {t2['run']}  tok={score_task2(t2):.4f}")


if __name__ == "__main__":
    main()
