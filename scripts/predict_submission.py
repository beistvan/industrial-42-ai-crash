#!/usr/bin/env python3
"""Produce submission CSVs for the three Infineon tasks.

Usage:
    python scripts/predict_submission.py \\
        --model models/ngram_baseline.pkl \\
        --eval-valid data/raw/infineon/eval/eval_input_valid.csv \\
        --eval-anomaly data/raw/infineon/eval/eval_input_anomaly.csv \\
        --out-dir extras/results

Outputs:
    artifacts/submission/nextstep.csv     (Task 1: SEQUENCE_ID, RANK, STEP)
    artifacts/submission/completion.csv   (Task 2: SEQUENCE_ID, STEP_INDEX, STEP)
    artifacts/submission/anomaly.csv      (Task 3: SEQUENCE_ID, IS_VALID, PRIMARY_RULE)

The expected eval CSV format is the same long-format SEQUENCE_ID + STEP layout
the official `read_csv_sequences` uses. Family inference is best-effort: if a
FAMILY column is present it is honored; otherwise the family is inferred from
the SEQUENCE_ID prefix (mosfet/igbt/ic) or defaults to mosfet.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.infineon_loader import Family, FAMILIES  # noqa: E402
from src.eval.rule_validator import classify_sequence  # noqa: E402
from src.ml import load_sequence_model  # noqa: E402


def _infer_family(sid: str, explicit: str | None) -> Family:
    if explicit:
        v = explicit.strip().lower()
        if v in FAMILIES:
            return v  # type: ignore[return-value]
    low = sid.lower()
    for fam in FAMILIES:
        if fam in low:
            return fam
    return "mosfet"


def _read_eval_csv(path: Path) -> dict[str, tuple[Family, list[str]]]:
    out_steps: dict[str, list[str]] = defaultdict(list)
    out_family: dict[str, Family] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize headers.
        fieldnames = {h.lstrip("﻿").strip().strip('"'): h for h in (reader.fieldnames or [])}
        sid_key = fieldnames.get("SEQUENCE_ID")
        step_key = fieldnames.get("STEP")
        fam_key = fieldnames.get("FAMILY")
        if sid_key is None or step_key is None:
            raise ValueError(f"{path} missing SEQUENCE_ID/STEP columns")
        for row in reader:
            sid = row[sid_key].strip()
            step = row[step_key].strip().strip('"')
            if not step:
                continue
            out_steps[sid].append(step)
            if fam_key and sid not in out_family:
                out_family[sid] = _infer_family(sid, row[fam_key])
    return {sid: (out_family.get(sid, _infer_family(sid, None)), steps)
            for sid, steps in out_steps.items()}


def write_nextstep(model, inputs: dict, out_path: Path, k: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "RANK", "STEP"])
        for sid, (family, partial) in inputs.items():
            preds = model.predict_topk(family, partial, k=k)
            for rank, step in enumerate(preds, start=1):
                w.writerow([sid, rank, step])


def write_completion(model, inputs: dict, out_path: Path, *,
                     rule_constrained: bool = True, candidate_pool: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "STEP_INDEX", "STEP"])
        for sid, (family, partial) in inputs.items():
            try:
                completed = model.complete(
                    family, list(partial), max_steps=300,
                    rule_constrained=rule_constrained,
                    candidate_pool=candidate_pool,
                )
            except TypeError:
                completed = model.complete(family, list(partial), max_steps=300)
            for idx, step in enumerate(completed):
                w.writerow([sid, idx, step])


def write_anomaly(inputs: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "IS_VALID", "PRIMARY_RULE", "ALL_RULES"])
        for sid, (_fam, steps) in inputs.items():
            res = classify_sequence(steps)
            w.writerow([sid, int(res["valid"]), res["primary_rule"] or "",
                        ";".join(res["rules"])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--eval-valid", type=Path,
                    help="Tasks 1+2 input CSV (partial sequences).")
    ap.add_argument("--eval-anomaly", type=Path,
                    help="Task 3 input CSV (full sequences).")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "extras" / "results")
    ap.add_argument("--rule-constrained", dest="rule_constrained",
                    action="store_true", default=True,
                    help="(default) Filter Task 2 candidates with validate_sequence.")
    ap.add_argument("--no-rule-constrained", dest="rule_constrained",
                    action="store_false",
                    help="Disable rule-constrained completion (plain greedy / argmax).")
    ap.add_argument("--candidate-pool", type=int, default=5,
                    help="Top-k pool considered per step when rule_constrained is on.")
    args = ap.parse_args()
    model = load_sequence_model(args.model)
    if args.eval_valid:
        valid_inputs = _read_eval_csv(args.eval_valid)
        write_nextstep(model, valid_inputs, args.out_dir / "nextstep.csv")
        write_completion(model, valid_inputs, args.out_dir / "completion.csv",
                         rule_constrained=args.rule_constrained,
                         candidate_pool=args.candidate_pool)
        print(f"wrote {args.out_dir/'nextstep.csv'} and {args.out_dir/'completion.csv'} "
              f"({len(valid_inputs)} sequences)")
    if args.eval_anomaly:
        anomaly_inputs = _read_eval_csv(args.eval_anomaly)
        write_anomaly(anomaly_inputs, args.out_dir / "anomaly.csv")
        print(f"wrote {args.out_dir/'anomaly.csv'} ({len(anomaly_inputs)} sequences)")
    if not args.eval_valid and not args.eval_anomaly:
        ap.error("at least one of --eval-valid / --eval-anomaly is required")


if __name__ == "__main__":
    main()
