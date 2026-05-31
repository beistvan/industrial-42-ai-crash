"""Tests for submission checkpoint picking."""
from __future__ import annotations

from scripts.sweep_picks import (
    pick_task1_run,
    pick_task2_run,
    score_task1,
    score_task2,
)


def test_task2_uses_best_value_for_token_acc_runs():
    rows = [
        {
            "run": "m_no_sched",
            "best_metric": "dev_mrr",
            "best_value": "0.8686",
            "task1_mrr": "0.8686",
            "task2_token_acc": "0.446",
        },
        {
            "run": "g_drop15_nosched_t2",
            "best_metric": "dev_token_acc",
            "best_value": "0.45451337751378657",
            "task1_mrr": "0.8671",
            "task2_token_acc": "0.4396",
        },
    ]
    assert pick_task2_run(rows) == "g_drop15_nosched_t2"
    assert score_task2(rows[1]) > score_task2(rows[0])


def test_task1_uses_best_value_for_mrr_runs():
    rows = [
        {
            "run": "f_drop15_100_mrr",
            "best_metric": "dev_mrr",
            "best_value": "0.8730555555555555",
            "task1_mrr": "0.8731",
            "task2_token_acc": "0.437",
        },
        {
            "run": "h_mod_nosched_mrr",
            "best_metric": "dev_mrr",
            "best_value": "0.8734722222222221",
            "task1_mrr": "0.8671",
            "task2_token_acc": "0.4396",
        },
    ]
    assert pick_task1_run(rows) == "h_mod_nosched_mrr"
    assert score_task1(rows[1]) > score_task1(rows[0])
