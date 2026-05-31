"""Shared logic for ranking sweep runs and picking submission checkpoints."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_leaderboard_rows(path: Path | str) -> list[dict[str, Any]]:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_task1(row: dict[str, Any]) -> float:
    """Effective Task-1 MRR for ranking (history peak + checkpoint best)."""
    score = _f(row.get("task1_mrr"))
    if row.get("best_metric") == "dev_mrr":
        score = max(score, _f(row.get("best_value")))
    return score


def score_task2(row: dict[str, Any]) -> float:
    """Effective Task-2 token accuracy for ranking (history peak + checkpoint best)."""
    score = _f(row.get("task2_token_acc"))
    if row.get("best_metric") == "dev_token_acc":
        score = max(score, _f(row.get("best_value")))
    return score


def pick_task1_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty leaderboard")
    return max(rows, key=score_task1)


def pick_task2_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty leaderboard")
    return max(rows, key=score_task2)


def pick_task1_run(rows: list[dict[str, Any]]) -> str:
    return str(pick_task1_row(rows)["run"])


def pick_task2_run(rows: list[dict[str, Any]]) -> str:
    return str(pick_task2_row(rows)["run"])
