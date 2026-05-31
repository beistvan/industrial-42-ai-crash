"""Data loading and helpers for the unified Track 1 dashboard."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.track_context import BASELINE, PIPELINE_WAVES

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_DIR = REPO_ROOT / "artifacts" / "sweeps"
LEADERBOARD_CSV = SWEEPS_DIR / "LEADERBOARD_FINAL.csv"
LEADERBOARD_MD = SWEEPS_DIR / "LEADERBOARD_FINAL.md"
NGRAM_METRICS = REPO_ROOT / "artifacts" / "ngram_metrics.json"
MODELS_DIR = REPO_ROOT / "models" / "sweeps"
SUBMISSION_DIR = REPO_ROOT / "result" / "submission"
DEV_EVAL_DIR = REPO_ROOT / "data" / "processed" / "dev_eval"
JUDGE_EVAL_DIR = REPO_ROOT / "EVAL_DATA"

WAVE_PREFIXES = {
    "m_": "Wave 1 shortlist",
    "f_": "Wave 1 finalists",
    "g_": "Wave 2 fine grid",
    "h_mod_": "Wave 3 modern arch",
    "t2_mod_": "Wave 4 Task-2",
    "p5_": "Wave 5 params",
    "s_": "Wave 6 scale/seeds",
}

EVAL_PROTOCOL = {
    "dev_holdout": {
        "path": "data/processed/dev_eval/",
        "split": "600 Task-1/2 items + injected anomalies for Task-3 (100/family holdout)",
        "used_for": "Training checkpoint selection, LEADERBOARD_FINAL, dashboard metrics",
    },
    "judge_submission": {
        "path": "EVAL_DATA/eval_input_*.csv",
        "outputs": "result/submission/{nextstep,completion,anomaly}.csv",
        "used_for": "Official hackathon submission (Slurm GPU predict on Leonardo)",
    },
}


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_scripts_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_leaderboard() -> pd.DataFrame | None:
    if not LEADERBOARD_CSV.exists():
        return None
    return pd.read_csv(LEADERBOARD_CSV)


def pick_winners(df: pd.DataFrame) -> tuple[str, str, float, float]:
    ensure_scripts_path()
    from scripts.sweep_picks import (
        pick_task1_row,
        pick_task2_row,
        score_task1,
        score_task2,
    )

    rows = df.to_dict("records")
    t1 = pick_task1_row(rows)
    t2 = pick_task2_row(rows)
    return (
        str(t1["run"]),
        str(t2["run"]),
        score_task1(t1),
        score_task2(t2),
    )


def run_json_path(run_name: str) -> Path:
    return SWEEPS_DIR / f"{run_name}.json"


def list_sweep_runs() -> list[str]:
    if not SWEEPS_DIR.exists():
        return []
    return sorted(p.stem for p in SWEEPS_DIR.glob("*.json"))


def wave_label(run: str) -> str:
    for prefix, label in WAVE_PREFIXES.items():
        if run.startswith(prefix):
            return label
    return "other"


def slurm_jobs() -> list[str]:
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "-h", "-o", "%j"],
            text=True,
            timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def submission_csv_rows() -> list[dict[str, Any]]:
    rows = []
    for name in ("nextstep.csv", "completion.csv", "anomaly.csv"):
        p = SUBMISSION_DIR / name
        if p.exists() and p.stat().st_size > 0:
            rows.append({
                "file": name,
                "lines": sum(1 for _ in p.open(encoding="utf-8")),
                "size_kb": round(p.stat().st_size / 1024, 1),
                "status": "ok",
            })
        elif p.exists():
            rows.append({"file": name, "lines": 0, "size_kb": 0, "status": "empty"})
        else:
            rows.append({"file": name, "lines": 0, "size_kb": 0, "status": "missing"})
    return rows


def progress_table(df: pd.DataFrame) -> pd.DataFrame:
    ensure_scripts_path()
    from scripts.sweep_picks import score_task1, score_task2, pick_task1_run, pick_task2_run

    def row_for(run: str) -> dict | None:
        hit = df[df["run"] == run]
        return None if hit.empty else hit.iloc[0].to_dict()

    rows: list[dict] = [{
        "stage": "Level 1 baseline",
        "model": "n-gram (order 12)",
        "task1_mrr": BASELINE["task1_mrr"],
        "task2_tok_acc": BASELINE["task2_token_acc"],
        "task3_rule_attr": BASELINE.get("task3_rule_attr"),
    }]
    for stage, run, tag in (
        ("Wave 1 T1", "f_drop15_100_mrr", "vanilla"),
        ("Wave 2 T2", "g_drop15_nosched_t2", "vanilla"),
    ):
        rec = row_for(run)
        if rec:
            rows.append({
                "stage": stage,
                "model": f"{run} ({tag})",
                "task1_mrr": round(score_task1(rec), 4),
                "task2_tok_acc": round(score_task2(rec), 4),
                "task3_rule_attr": rec.get("task3_rule_attr"),
            })
    t1_run = pick_task1_run(df.to_dict("records"))
    t2_run = pick_task2_run(df.to_dict("records"))
    t1_rec, t2_rec = row_for(t1_run) or {}, row_for(t2_run) or {}
    rows.append({
        "stage": "Submission (hybrid)",
        "model": f"{t1_run} + {t2_run}",
        "task1_mrr": round(score_task1(t1_rec), 4),
        "task2_tok_acc": round(score_task2(t2_rec), 4),
        "task3_rule_attr": t1_rec.get("task3_rule_attr"),
    })
    return pd.DataFrame(rows)


def pipeline_rows(jobs: list[str]) -> list[dict]:
    log_text = ""
    for p in sorted((REPO_ROOT / "logs").glob("slurm-sweep-*.out"))[-24:]:
        try:
            log_text += p.read_text(encoding="utf-8", errors="replace")[:800]
        except OSError:
            pass
    patterns = {
        "Wave 1": ("f_", "m_"),
        "Wave 2": ("g_",),
        "Wave 3": ("h_mod_",),
        "Wave 4": ("t2_mod_",),
        "Wave 5": ("p5_",),
        "Wave 6": ("s_",),
    }
    out = []
    for wave, desc, default_status, note in PIPELINE_WAVES:
        status = default_status
        if any("zoh-sweep" in j for j in jobs):
            prefs = patterns.get(wave, ())
            if prefs and any(p in log_text for p in prefs):
                status = "running"
        out.append({"wave": wave, "focus": desc, "status": status, "note": note})
    return out


def metrics_sections(payload: dict[str, Any]) -> tuple[dict, dict, dict, dict]:
    metrics = payload.get("metrics", payload)
    t1 = metrics.get("task1_next_step", {}).get("overall", {})
    t2 = metrics.get("task2_completion", {}).get("overall", {})
    t3 = metrics.get("task3_anomaly", {})
    return metrics, t1, t2, t3


def training_history_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for h in payload.get("history", []):
        t1 = h.get("task1") or {}
        t2 = h.get("task2") or {}
        rows.append({
            "epoch": h.get("epoch"),
            "train_loss": h.get("train_loss"),
            "dev_loss": h.get("dev_loss"),
            "lr": h.get("lr"),
            "task1_mrr": t1.get("mrr"),
            "task1_top1": t1.get("top1"),
            "task2_tok_acc": t2.get("token_accuracy"),
            "task2_ned": t2.get("normalized_edit_distance"),
        })
    return pd.DataFrame(rows)


def model_config_table(payload: dict[str, Any]) -> pd.DataFrame:
    model = payload.get("model", {})
    meta = model.get("metadata", {})
    cfg = model.get("config", {})
    train = meta.get("train_cfg", {})
    best = payload.get("best", {})
    rows = [
        ("architecture", train.get("arch", "vanilla")),
        ("model_type", model.get("model", "—")),
        ("d_model", cfg.get("d_model")),
        ("n_layers", cfg.get("n_layers")),
        ("n_heads", cfg.get("n_heads")),
        ("dropout", cfg.get("dropout")),
        ("vocab_size", model.get("vocab_size")),
        ("config_yaml", meta.get("config_path")),
        ("epochs", train.get("epochs")),
        ("batch_size", train.get("batch_size")),
        ("lr", train.get("lr")),
        ("scheduler", train.get("scheduler")),
        ("label_smoothing", train.get("label_smoothing")),
        ("save_best_by", best.get("metric")),
        ("best_epoch", best.get("epoch")),
        ("best_value", best.get("value")),
        ("checkpoint", repo_path(Path(best.get("path", ""))) if best.get("path") else "—"),
        ("extra_data", payload.get("extra_data_dir", meta.get("extra_data_dir", "—"))),
        ("train_seconds", payload.get("train_seconds", "—")),
        ("eval_seconds", payload.get("eval_seconds", "—")),
    ]
    return pd.DataFrame(rows, columns=["setting", "value"])
