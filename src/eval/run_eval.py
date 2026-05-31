"""End-to-end local evaluation for a next-step model on the dev split.

Currently wired for NGramBaseline; the same interface (predict_topk +
complete) is what the transformer will implement, so swapping in a trained
model later is a one-line change.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.data.infineon_loader import Family
from src.eval.dev_split import TRUNCATION_FRACTIONS
from src.eval.metrics import (
    exact_match,
    mean_reciprocal_rank,
    normalized_edit_distance,
    token_accuracy,
    top_k_accuracy,
)
from src.eval.anomaly_scoring import injection_label_matches_prediction
from src.eval.rule_validator import classify_sequence


class NextStepModel(Protocol):
    def predict_topk(self, family: Family, prefix: list[str], k: int = 5) -> list[str]: ...
    def complete(self, family: Family, prefix: list[str], *, max_steps: int = 200,
                 stop_token: str = "SHIP LOT") -> list[str]: ...


def _read_long_csv(path: Path) -> dict[str, tuple[Family, list[str]]]:
    """Read a long-format eval CSV (SEQUENCE_ID, FAMILY, STEP)."""
    out: dict[str, tuple[Family, list[str]]] = {}
    fam_by_sid: dict[str, Family] = {}
    steps_by_sid: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["SEQUENCE_ID"].strip()
            fam = row["FAMILY"].strip()
            step = row["STEP"].strip()
            fam_by_sid[sid] = fam  # type: ignore[assignment]
            steps_by_sid[sid].append(step)
    for sid, steps in steps_by_sid.items():
        out[sid] = (fam_by_sid[sid], steps)
    return out


@dataclass
class TaskMetrics:
    task1: dict
    task2: dict
    task3: dict

    def to_dict(self) -> dict:
        return {"task1_next_step": self.task1,
                "task2_completion": self.task2,
                "task3_anomaly": self.task3}


# ---------- Task 1: next-step ------------------------------------------------

def evaluate_next_step(
    model: NextStepModel,
    valid_input_csv: Path,
    valid_gold_csv: Path,
) -> dict:
    inputs = _read_long_csv(valid_input_csv)
    golds = _read_long_csv(valid_gold_csv)
    per_family_topk: dict[Family, list[list[str]]] = defaultdict(list)
    per_family_gold: dict[Family, list[str]] = defaultdict(list)
    for sid, (family, partial) in inputs.items():
        full = golds[sid][1]
        if len(partial) >= len(full):
            continue
        gold_next = full[len(partial)]
        preds = model.predict_topk(family, partial, k=5)
        per_family_topk[family].append(preds)
        per_family_gold[family].append(gold_next)
    overall = {"top1": 0.0, "top3": 0.0, "top5": 0.0, "mrr": 0.0, "n": 0}
    per_family_metrics: dict[str, dict] = {}
    for family in per_family_topk:
        preds = per_family_topk[family]
        gold = per_family_gold[family]
        per_family_metrics[family] = {
            "n": len(gold),
            "top1": top_k_accuracy(preds, gold, 1),
            "top3": top_k_accuracy(preds, gold, 3),
            "top5": top_k_accuracy(preds, gold, 5),
            "mrr":  mean_reciprocal_rank(preds, gold),
        }
    # Micro-average across all items.
    all_preds = [p for fam in per_family_topk for p in per_family_topk[fam]]
    all_gold = [g for fam in per_family_gold for g in per_family_gold[fam]]
    overall = {
        "n": len(all_gold),
        "top1": top_k_accuracy(all_preds, all_gold, 1),
        "top3": top_k_accuracy(all_preds, all_gold, 3),
        "top5": top_k_accuracy(all_preds, all_gold, 5),
        "mrr":  mean_reciprocal_rank(all_preds, all_gold),
    }
    return {"overall": overall, "per_family": per_family_metrics}


# ---------- Task 2: completion ----------------------------------------------

def evaluate_completion(
    model: NextStepModel,
    valid_input_csv: Path,
    valid_gold_csv: Path,
    *,
    rule_constrained: bool = True,
    candidate_pool: int = 5,
    beam_width: int = 1,
    length_normalize: bool = True,
) -> dict:
    inputs = _read_long_csv(valid_input_csv)
    golds = _read_long_csv(valid_gold_csv)
    per_family: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "exact": 0, "token_acc_sum": 0.0, "ned_sum": 0.0}
    )
    for sid, (family, partial) in inputs.items():
        full = golds[sid][1]
        completion_only_gold = full[len(partial):]
        if not completion_only_gold:
            continue
        try:
            completed = model.complete(
                family,
                list(partial),
                max_steps=300,
                rule_constrained=rule_constrained,
                candidate_pool=candidate_pool,
                beam_width=beam_width,
                length_normalize=length_normalize,
            )
        except TypeError:
            # Older model objects that don't yet accept the new kwargs.
            completed = model.complete(family, list(partial), max_steps=300)
        completion_only_pred = completed[len(partial):]
        bucket = per_family[family]
        bucket["n"] += 1
        bucket["exact"] += int(exact_match(completion_only_pred, completion_only_gold))
        bucket["token_acc_sum"] += token_accuracy(completion_only_pred, completion_only_gold)
        bucket["ned_sum"] += normalized_edit_distance(completion_only_pred, completion_only_gold)
    per_family_out = {}
    total_n = 0
    total_exact = 0
    total_token = 0.0
    total_ned = 0.0
    for fam, b in per_family.items():
        n = max(b["n"], 1)
        per_family_out[fam] = {
            "n": b["n"],
            "exact_match_rate": b["exact"] / n,
            "token_accuracy": b["token_acc_sum"] / n,
            "normalized_edit_distance": b["ned_sum"] / n,
        }
        total_n += b["n"]
        total_exact += b["exact"]
        total_token += b["token_acc_sum"]
        total_ned += b["ned_sum"]
    overall = {
        "n": total_n,
        "exact_match_rate": (total_exact / total_n) if total_n else 0.0,
        "token_accuracy": (total_token / total_n) if total_n else 0.0,
        "normalized_edit_distance": (total_ned / total_n) if total_n else 0.0,
    }
    return {"overall": overall, "per_family": per_family_out}


# ---------- Task 3: anomaly detection (rule-validator only) -----------------

def evaluate_anomaly(
    anomaly_input_csv: Path,
    anomaly_gold_csv: Path,
) -> dict:
    """Score the rule validator as a sequence classifier.

    Predictor: validator (sequence valid iff zero violations).
    """
    inputs = _read_long_csv(anomaly_input_csv)
    gold_labels: dict[str, dict] = {}
    with anomaly_gold_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold_labels[row["SEQUENCE_ID"].strip()] = {
                "valid": bool(int(row["VALID"])),
                "injected_label": row.get("INJECTED_LABEL", ""),
            }
    tp = fp = tn = fn = 0
    rule_correct = 0
    rule_evaluated = 0
    for sid, (_fam, steps) in inputs.items():
        gold = gold_labels.get(sid)
        if gold is None:
            continue
        pred = classify_sequence(steps)
        # Positive class = "invalid" (anomaly).
        gold_invalid = not gold["valid"]
        pred_invalid = not pred["valid"]
        if gold_invalid and pred_invalid:
            tp += 1
        elif gold_invalid and not pred_invalid:
            fn += 1
        elif (not gold_invalid) and pred_invalid:
            fp += 1
        else:
            tn += 1
        # Rule attribution: when both are invalid and the injected label
        # starts with a rule name like "RULE_X_drop_y", compare prefix.
        if gold_invalid and pred_invalid and gold["injected_label"]:
            if injection_label_matches_prediction(
                gold["injected_label"], pred["rules"], pred["primary_rule"]
            ):
                rule_correct += 1
            rule_evaluated += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return {
        "n": n,
        "accuracy": accuracy,
        "precision_invalid": precision,
        "recall_invalid": recall,
        "f1_invalid": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "rule_attribution_accuracy": (rule_correct / rule_evaluated) if rule_evaluated else 0.0,
        "rule_attribution_n": rule_evaluated,
    }


def evaluate_all(
    model: NextStepModel,
    eval_dir: Path,
    *,
    rule_constrained: bool = True,
    candidate_pool: int = 5,
    beam_width: int = 1,
    length_normalize: bool = True,
) -> TaskMetrics:
    return TaskMetrics(
        task1=evaluate_next_step(
            model,
            eval_dir / "eval_input_valid_dev.csv",
            eval_dir / "eval_input_valid_dev_gold.csv",
        ),
        task2=evaluate_completion(
            model,
            eval_dir / "eval_input_valid_dev.csv",
            eval_dir / "eval_input_valid_dev_gold.csv",
            rule_constrained=rule_constrained,
            candidate_pool=candidate_pool,
            beam_width=beam_width,
            length_normalize=length_normalize,
        ),
        task3=evaluate_anomaly(
            eval_dir / "eval_input_anomaly_dev.csv",
            eval_dir / "eval_input_anomaly_dev_gold.csv",
        ),
    )
