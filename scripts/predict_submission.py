#!/usr/bin/env python3
"""Produce submission CSVs for the three Infineon tasks.

Usage:
    python scripts/predict_submission.py \\
        --model models/sweeps/m_real_extras_1x.pt.best \\
        --eval-valid EVAL_DATA/eval_input_valid.csv \\
        --eval-anomaly EVAL_DATA/eval_input_anomaly.csv \\
        --out-dir extras/results_submission \\
        --rule-constrained --beam-width 5 --candidate-pool 5

Auto-detects input format:

  * Judge "wide" format (preferred for submission):
        Valid:   EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE  (steps "|"-joined)
        Anomaly: EXAMPLE_ID,FAMILY,SEQUENCE                              (steps "|"-joined)
    -> outputs the judge's required wide formats.

  * Legacy long format used by the dev split:
        SEQUENCE_ID,STEP[,FAMILY]   (one step per row)
    -> outputs legacy long-format CSVs for backward compatibility.

Judge output formats (per EVAL_DATA/eval_metrics.py):

    nextstep:   EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5
    completion: EXAMPLE_ID,PREDICTED_SEQUENCE          (pipe-separated)
    anomaly:    EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE
                SCORE in [0,1], higher = more likely valid (feeds AUC).
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

JUDGE_FORMAT = "judge_wide"
LEGACY_FORMAT = "legacy_long"

# Anomaly SCORE for judge format. Treated as P(valid). For meaningful AUC,
# replace with per-step LM likelihood once available. Until then, scale
# inversely with the number of rule violations.
_MAX_RULES_FOR_SCORING = 10


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


def _normalize_headers(reader: csv.DictReader) -> dict[str, str]:
    return {h.lstrip("﻿").strip().strip('"'): h for h in (reader.fieldnames or [])}


def _split_pipe(value: str) -> list[str]:
    return [tok.strip().strip('"') for tok in value.split("|") if tok.strip()]


def _read_eval_csv(path: Path) -> tuple[dict[str, tuple[Family, list[str]]], str, list[str]]:
    """Return (mapping, format, ordered_ids).

    `mapping` is id -> (family, steps). `format` is JUDGE_FORMAT or LEGACY_FORMAT.
    `ordered_ids` preserves CSV row order so output rows match input order.
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = _normalize_headers(reader)
        rows = list(reader)

    is_judge = "EXAMPLE_ID" in headers and (
        "PARTIAL_SEQUENCE" in headers or "SEQUENCE" in headers
    )

    if is_judge:
        id_col = headers["EXAMPLE_ID"]
        fam_col = headers.get("FAMILY")
        seq_col = headers.get("PARTIAL_SEQUENCE") or headers.get("SEQUENCE")
        if seq_col is None:
            raise ValueError(f"{path}: judge format missing PARTIAL_SEQUENCE/SEQUENCE column")
        out: dict[str, tuple[Family, list[str]]] = {}
        ordered: list[str] = []
        for row in rows:
            sid = row[id_col].strip()
            if not sid:
                continue
            steps = _split_pipe(row[seq_col])
            family = _infer_family(sid, row[fam_col] if fam_col else None)
            out[sid] = (family, steps)
            ordered.append(sid)
        return out, JUDGE_FORMAT, ordered

    # Legacy long format.
    sid_col = headers.get("SEQUENCE_ID")
    step_col = headers.get("STEP")
    fam_col = headers.get("FAMILY")
    if sid_col is None or step_col is None:
        raise ValueError(
            f"{path}: unrecognized headers {list(headers)}. "
            "Expected EXAMPLE_ID+PARTIAL_SEQUENCE/SEQUENCE (judge) or SEQUENCE_ID+STEP (legacy)."
        )
    out_steps: dict[str, list[str]] = defaultdict(list)
    out_family: dict[str, Family] = {}
    ordered = []
    seen: set[str] = set()
    for row in rows:
        sid = row[sid_col].strip()
        step = row[step_col].strip().strip('"')
        if not sid or not step:
            continue
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
        out_steps[sid].append(step)
        if fam_col and sid not in out_family:
            out_family[sid] = _infer_family(sid, row[fam_col])
    mapping = {sid: (out_family.get(sid, _infer_family(sid, None)), steps)
               for sid, steps in out_steps.items()}
    return mapping, LEGACY_FORMAT, ordered


def _anomaly_score(rules: list[str]) -> float:
    """Heuristic P(valid) in [0, 1]. Replace with LM likelihood for true AUC."""
    n = min(len(rules), _MAX_RULES_FOR_SCORING)
    return 1.0 - n / _MAX_RULES_FOR_SCORING


def write_nextstep(model, inputs: dict, ordered_ids: list[str], out_path: Path,
                   fmt: str, k: int = 5) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fmt == JUDGE_FORMAT:
            w.writerow(["EXAMPLE_ID"] + [f"RANK_{i}" for i in range(1, k + 1)])
            for sid in ordered_ids:
                family, partial = inputs[sid]
                preds = list(model.predict_topk(family, partial, k=k))
                preds += [""] * (k - len(preds))
                w.writerow([sid] + preds[:k])
        else:
            w.writerow(["SEQUENCE_ID", "RANK", "STEP"])
            for sid in ordered_ids:
                family, partial = inputs[sid]
                for rank, step in enumerate(model.predict_topk(family, partial, k=k), start=1):
                    w.writerow([sid, rank, step])


def _complete(model, family: Family, partial: list[str], *,
              rule_constrained: bool, candidate_pool: int,
              beam_width: int, length_normalize: bool) -> list[str]:
    try:
        return model.complete(
            family, list(partial), max_steps=300,
            rule_constrained=rule_constrained,
            candidate_pool=candidate_pool,
            beam_width=beam_width,
            length_normalize=length_normalize,
        )
    except TypeError:
        return model.complete(family, list(partial), max_steps=300)


def write_completion(model, inputs: dict, ordered_ids: list[str], out_path: Path,
                     fmt: str, *, rule_constrained: bool, candidate_pool: int,
                     beam_width: int, length_normalize: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fmt == JUDGE_FORMAT:
            w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
            for sid in ordered_ids:
                family, partial = inputs[sid]
                completed = _complete(
                    model, family, list(partial),
                    rule_constrained=rule_constrained,
                    candidate_pool=candidate_pool,
                    beam_width=beam_width,
                    length_normalize=length_normalize,
                )
                # Submit only the newly predicted tail (judge already has the prefix).
                tail = completed[len(partial):]
                w.writerow([sid, "|".join(tail)])
        else:
            w.writerow(["SEQUENCE_ID", "STEP_INDEX", "STEP"])
            for sid in ordered_ids:
                family, partial = inputs[sid]
                completed = _complete(
                    model, family, list(partial),
                    rule_constrained=rule_constrained,
                    candidate_pool=candidate_pool,
                    beam_width=beam_width,
                    length_normalize=length_normalize,
                )
                for idx, step in enumerate(completed):
                    w.writerow([sid, idx, step])


def write_anomaly(inputs: dict, ordered_ids: list[str], out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fmt == JUDGE_FORMAT:
            w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
            for sid in ordered_ids:
                _fam, steps = inputs[sid]
                res = classify_sequence(steps)
                w.writerow([
                    sid,
                    int(res["valid"]),
                    f"{_anomaly_score(res['rules']):.4f}",
                    res["primary_rule"] or "",
                ])
        else:
            w.writerow(["SEQUENCE_ID", "IS_VALID", "PRIMARY_RULE", "ALL_RULES"])
            for sid in ordered_ids:
                _fam, steps = inputs[sid]
                res = classify_sequence(steps)
                w.writerow([
                    sid,
                    int(res["valid"]),
                    res["primary_rule"] or "",
                    ";".join(res["rules"]),
                ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--eval-valid", type=Path,
                    help="Tasks 1+2 input CSV (partial sequences).")
    ap.add_argument("--eval-anomaly", type=Path,
                    help="Task 3 input CSV (full sequences).")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "extras" / "results")
    ap.add_argument("--device", default=None,
                    help="Device for Transformer checkpoints, e.g. cpu or cuda. Ignored for n-gram .pkl models.")
    ap.add_argument("--rule-constrained", dest="rule_constrained",
                    action="store_true", default=True,
                    help="(default) Filter Task 2 candidates with validate_sequence.")
    ap.add_argument("--no-rule-constrained", dest="rule_constrained",
                    action="store_false",
                    help="Disable rule-constrained completion (plain greedy / argmax).")
    ap.add_argument("--candidate-pool", type=int, default=5,
                    help="Top-k pool considered per step when rule_constrained is on.")
    ap.add_argument("--beam-width", type=int, default=1,
                    help="Beam search width for Task 2 completion (1 = greedy).")
    ap.add_argument("--no-length-normalize", dest="length_normalize",
                    action="store_false", default=True,
                    help="Disable length-normalized beam scoring (default on).")
    args = ap.parse_args()

    model = load_sequence_model(args.model, device=args.device)

    if args.eval_valid:
        valid_inputs, fmt, ordered = _read_eval_csv(args.eval_valid)
        print(f"[predict] {args.eval_valid.name}: format={fmt}, {len(ordered)} examples")
        write_nextstep(model, valid_inputs, ordered, args.out_dir / "nextstep.csv", fmt)
        write_completion(model, valid_inputs, ordered, args.out_dir / "completion.csv", fmt,
                         rule_constrained=args.rule_constrained,
                         candidate_pool=args.candidate_pool,
                         beam_width=args.beam_width,
                         length_normalize=args.length_normalize)
        print(f"[predict] wrote {args.out_dir/'nextstep.csv'} and {args.out_dir/'completion.csv'}")

    if args.eval_anomaly:
        anomaly_inputs, fmt, ordered = _read_eval_csv(args.eval_anomaly)
        print(f"[predict] {args.eval_anomaly.name}: format={fmt}, {len(ordered)} examples")
        write_anomaly(anomaly_inputs, ordered, args.out_dir / "anomaly.csv", fmt)
        print(f"[predict] wrote {args.out_dir/'anomaly.csv'}")

    if not args.eval_valid and not args.eval_anomaly:
        ap.error("at least one of --eval-valid / --eval-anomaly is required")


if __name__ == "__main__":
    main()
