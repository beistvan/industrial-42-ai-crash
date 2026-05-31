"""Streamlit sweep dashboard — GPU sweep findings + Track 1 alignment.

Reads:
    artifacts/sweeps/LEADERBOARD_FINAL.csv
    artifacts/sweeps/{m,f,g,h,t2}_*.json
    logs/wave_orchestrator.log (pipeline status)

Run:
    make run-sweep-dashboard
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.track_context import (
    BASELINE,
    LEVELS,
    PIPELINE_WAVES,
    SCALING_ROWS,
    SUBMISSION,
    TRACK_ALIGNMENT,
    TRACK_DOC_URL,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_DIR = REPO_ROOT / "artifacts" / "sweeps"
LEADERBOARD_CSV = SWEEPS_DIR / "LEADERBOARD_FINAL.csv"
LEADERBOARD_MD = SWEEPS_DIR / "LEADERBOARD_FINAL.md"
GPU_SUMMARY = REPO_ROOT / "RESULTS_GPU_SUMMARY.md"
MODELS_DIR = REPO_ROOT / "models" / "sweeps"
ORCH_LOG = REPO_ROOT / "logs" / "wave_orchestrator.log"
MONITOR_LOG = REPO_ROOT / "logs" / "monitor-waves.out"

WAVE_PREFIXES = {
    "m_": "Wave 1 shortlist",
    "f_": "Wave 1 finalists",
    "g_": "Wave 2 fine grid",
    "h_mod_": "Wave 3 modern arch",
    "t2_mod_": "Wave 4 Task-2",
    "p5_": "Wave 5 params",
    "s_": "Wave 6 scale/seeds",
}

SUBMISSION_DIR = REPO_ROOT / "result" / "submission"
SUBMISSION_FILES = ("nextstep.csv", "completion.csv", "anomaly.csv")


def _row_metrics(df: pd.DataFrame, run: str) -> dict | None:
    hit = df[df["run"] == run]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def _progress_table(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline → Wave 1 → Wave 2 T2 → current hybrid (live picks)."""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sweep_picks import score_task1, score_task2

    rows: list[dict] = [{
        "stage": "Level 1 baseline",
        "model": "n-gram (order 12)",
        "task1_mrr": BASELINE["task1_mrr"],
        "task2_tok_acc": BASELINE["task2_token_acc"],
        "Δ MRR": "—",
        "Δ tok": "—",
    }]
    for stage, run, label in (
        ("Level 2 trained (T1)", "f_drop15_100_mrr", "Wave 1"),
        ("Level 2 optimized (T2)", "g_drop15_nosched_t2", "Wave 2"),
        ("Level 3 + submission", None, "live picks"),
    ):
        if run:
            rec = _row_metrics(df, run)
            if not rec:
                continue
            t1 = score_task1(rec)
            t2 = score_task2(rec)
        else:
            t1_run, t2_run = _best_runs(df)
            t1 = score_task1(_row_metrics(df, t1_run) or {})
            t2 = score_task2(_row_metrics(df, t2_run) or {})
            label = f"{t1_run} + {t2_run}"
        model_label = f"{run} ({label})" if run else label
        rows.append({
            "stage": stage,
            "model": model_label,
            "task1_mrr": round(t1, 4),
            "task2_tok_acc": round(t2, 4),
            "Δ MRR": f"+{(t1 - BASELINE['task1_mrr']) * 100:.1f} pp",
            "Δ tok": f"+{(t2 - BASELINE['task2_token_acc']) * 100:.1f} pp",
        })
    return pd.DataFrame(rows)


def _pipeline_status(jobs: list[str], log_names: list[str]) -> list[dict]:
    """Mark waves running if matching runs appear in Slurm or recent sweep logs."""
    running_logs = " ".join(log_names)
    wave_patterns = {
        "Wave 1": ("f_", "m_"),
        "Wave 2": ("g_",),
        "Wave 3": ("h_mod_",),
        "Wave 4": ("t2_mod_",),
        "Wave 5": ("p5_",),
        "Wave 6": ("s_",),
    }
    rows = []
    for wave, desc, default_status, note in PIPELINE_WAVES:
        status = default_status
        if any("zoh-sweep" in j for j in jobs):
            prefixes = wave_patterns.get(wave, ())
            if prefixes and any(p in running_logs for p in prefixes):
                status = "running"
        rows.append({"wave": wave, "focus": desc, "status": status, "note": note})
    return rows


def _submission_csv_status() -> list[dict]:
    out = []
    for name in SUBMISSION_FILES:
        p = SUBMISSION_DIR / name
        if p.exists():
            out.append({
                "file": name,
                "lines": sum(1 for _ in p.open(encoding="utf-8")),
                "size_kb": round(p.stat().st_size / 1024, 1),
                "status": "ok" if p.stat().st_size > 0 else "empty",
            })
        else:
            out.append({"file": name, "lines": 0, "size_kb": 0, "status": "missing"})
    return out


def _slurm_running() -> list[str]:
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "-h", "-o", "%j"],
            text=True,
            timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def _best_runs(df: pd.DataFrame) -> tuple[str, str]:
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sweep_picks import pick_task1_run, pick_task2_run

    rows = df.to_dict("records")
    return pick_task1_run(rows), pick_task2_run(rows)


st.set_page_config(page_title="Track 1 — Sweep & Pipeline", layout="wide")
st.title("Track 1 — Process Logic Benchmark")
st.caption(
    f"Aligned with [Zero One Track 1]({TRACK_DOC_URL}) · "
    "Leonardo GPU sweep · hybrid submission strategy"
)

# ------------------------------------------------------------------ Level 1 / 2 / 3 progress
st.subheader("Track 1 levels — baseline → trained → optimized")
level_cols = st.columns(3)
for col, lvl in zip(level_cols, LEVELS):
    with col:
        badge = {"done": "✅", "in_progress": "🔄"}.get(lvl["status"], "⏳")
        st.markdown(f"**{lvl['level']} — {lvl['title']}** {badge}")
        for item in lvl["items"]:
            st.caption(f"· {item}")
        st.metric("Headline", lvl["metric"])

st.markdown("**Baseline → trained → optimized → submission (dev holdout)**")
if LEADERBOARD_CSV.exists():
    _df_preview = pd.read_csv(LEADERBOARD_CSV)
    st.dataframe(_progress_table(_df_preview), width="stretch", hide_index=True)
else:
    st.warning("Run `make leonardo-leaderboard-final` to populate progress table.")

# ------------------------------------------------------------------ Track 1 alignment
with st.expander("Are we following Track 1?", expanded=False):
    st.markdown(TRACK_ALIGNMENT)
    st.link_button("Official Track 1 brief", TRACK_DOC_URL)

# ------------------------------------------------------------------ pipeline status
st.subheader("GPU pipeline")
jobs = _slurm_running()
log_dir = REPO_ROOT / "logs"
running_log_text = ""
for p in sorted(log_dir.glob("slurm-sweep-*.out"))[-24:]:
    try:
        running_log_text += p.read_text(encoding="utf-8", errors="replace")[:800]
    except OSError:
        pass
pipe_rows = _pipeline_status(jobs, [running_log_text])
st.dataframe(pd.DataFrame(pipe_rows), width="stretch", hide_index=True)
if jobs:
    st.caption(f"SLURM jobs: {', '.join(set(jobs))} ({len(jobs)} tasks)")
orch_tail = ORCH_LOG.read_text(encoding="utf-8").splitlines()[-3:] if ORCH_LOG.exists() else []
if orch_tail:
    st.code("\n".join(orch_tail), language="text")

st.subheader("Judge submission CSVs")
st.caption(f"`{SUBMISSION_DIR.relative_to(REPO_ROOT)}/` — regenerate with `make regenerate-submission`")
st.dataframe(pd.DataFrame(_submission_csv_status()), width="stretch", hide_index=True)

if not LEADERBOARD_CSV.exists():
    st.error(f"No leaderboard at `{LEADERBOARD_CSV.relative_to(REPO_ROOT)}`. "
             "Sync artifacts from Leonardo or wait for Wave 2 to finish.")
    st.stop()

df = pd.read_csv(LEADERBOARD_CSV)
model_t1, model_t2 = _best_runs(df)
rec_t1 = df[df["run"] == model_t1].iloc[0]
rec_t2 = df[df["run"] == model_t2].iloc[0]

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.sweep_picks import score_task1, score_task2

t1_score = score_task1(rec_t1.to_dict())
t2_score = score_task2(rec_t2.to_dict())

# ------------------------------------------------------------------ submission picks
st.subheader("Current best checkpoints (dev holdout)")
c = st.columns(4)
c[0].metric("Task 1 + anomaly", model_t1)
c[1].metric("Task 1 MRR", f"{t1_score:.4f}")
c[2].metric("Task 2 completion", model_t2)
c[3].metric("Task 2 tok acc", f"{t2_score:.4f}")

hybrid = model_t1 != model_t2
if hybrid:
    st.info(
        f"**Hybrid submission:** `{model_t1}` → nextstep + anomaly · "
        f"`{model_t2}` → completion.csv"
    )
else:
    st.success(f"Single checkpoint covers all tasks: `{model_t1}`")

for run_name in (model_t1, model_t2):
    ckpt = MODELS_DIR / f"{run_name}.pt.best"
    if ckpt.exists():
        st.caption(f"✓ `{ckpt.relative_to(REPO_ROOT)}` ({ckpt.stat().st_size / 1e6:.1f} MB)")
    else:
        st.warning(f"Missing checkpoint: `{ckpt.relative_to(REPO_ROOT)}`")

# ------------------------------------------------------------------ wave breakdown
st.subheader(f"Leaderboard ({len(df)} runs)")

def _wave_label(run: str) -> str:
    for prefix, label in WAVE_PREFIXES.items():
        if run.startswith(prefix):
            return label
    return "other"

display = df.sort_values("task1_mrr", ascending=False).copy().reset_index(drop=True)
display.insert(0, "rank", display.index + 1)
display["wave"] = display["run"].map(_wave_label)
display["role"] = display["run"].map(
    lambda r: "T1/anomaly" if r == model_t1 else ("T2" if r == model_t2 else "")
)
for col in ("task1_top1", "task1_top5", "task1_mrr", "task2_token_acc", "task2_ned"):
    if col in display.columns:
        display[col] = display[col].round(4)
display["extras"] = display["extras"].fillna("none")
st.dataframe(
    display[["rank", "wave", "role", "run", "task1_mrr", "task1_top1",
             "task2_token_acc", "task2_ned", "best_epoch", "train_seconds", "extras"]],
    width="stretch",
    hide_index=True,
)

# ------------------------------------------------------------------ charts
st.subheader("Visual comparison")
chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown("**Task 1 — MRR** (Track 1 metric: Top-1/3/5 + MRR)")
    st.bar_chart(display.set_index("run")[["task1_mrr"]].head(12))
with chart_cols[1]:
    st.markdown("**Task 2 — token accuracy** (60%/80% prefix completion)")
    t2_chart = display.copy()
    t2_chart["task2_rank_score"] = t2_chart.apply(
        lambda r: score_task2(r.to_dict()), axis=1
    )
    st.bar_chart(
        t2_chart.sort_values("task2_rank_score", ascending=False)
        .set_index("run")[["task2_rank_score"]].head(12)
    )

# ------------------------------------------------------------------ Level 3 scaling
st.subheader("Level 3 — data scaling (dev holdout)")
scaling_data = []
for label, run, extras, wall in SCALING_ROWS:
    row = df[df["run"] == run]
    if row.empty:
        continue
    r = row.iloc[0]
    scaling_data.append({
        "recipe": label,
        "run": run,
        "extras": extras,
        "task1_mrr": round(r["task1_mrr"], 4),
        "task2_tok_acc": round(r["task2_token_acc"], 4),
        "train_seconds": round(r["train_seconds"]),
        "wall_time": wall,
    })
if scaling_data:
    st.dataframe(pd.DataFrame(scaling_data), width="stretch", hide_index=True)
    st.caption("1× synthetic extras helps modestly; 2× / 500-per-family diverges from real distribution.")

# ------------------------------------------------------------------ findings
st.subheader("What we learned so far")
st.markdown(
    """
1. **Wave 3 modern arch** broke the T1 ceiling slightly — `h_mod_nosched_mrr` **MRR 0.874** (RoPE/RMSNorm/SwiGLU + nosched LR).
2. **T2 still Wave 2** — `g_drop15_nosched_t2` **tok 0.455**; Wave 4 prefix runs ~0.43–0.44 so far.
3. **Hybrid submission wins** — best MRR checkpoint ≠ best completion checkpoint.
4. **Regenerate judge CSVs:** `make regenerate-submission` (Slurm GPU predict on login node).
5. **Task 3 follows T1** — `anomaly.csv` SCORE uses the T1 checkpoint; validator is fixed logic.
6. **In flight:** Wave 3b tune, Wave 4 T2, Wave 5 param enrichment — refresh this page to update.
    """
)

# ------------------------------------------------------------------ track task mapping
st.subheader("Track 1 submission tasks → our outputs")
st.markdown(
    """
| # | Track 1 task | Judge metrics | Our file | Model |
|---|---|---|---|---|
| 1 | Next-step prediction | Top-1/3/5, MRR | `nextstep.csv` | Best MRR checkpoint |
| 2 | Sequence completion | Token acc, NED, exact match, block acc | `completion.csv` | Best tok-acc checkpoint |
| 3 | Anomaly detection | F1, AUC, rule attribution | `anomaly.csv` | T1 checkpoint (SCORE) + rule validator (IS_VALID, PREDICTED_RULE) |

Task 4 (OOD generalization) is **organizer-only** — we do not submit separately.
    """
)

with st.expander("Raw leaderboard (Markdown)"):
    if LEADERBOARD_MD.exists():
        st.markdown(LEADERBOARD_MD.read_text(encoding="utf-8"))
    else:
        st.caption("Run `make leonardo-leaderboard-final` to generate.")

with st.expander("Per-run metrics JSON"):
    runs = sorted({p for prefix in WAVE_PREFIXES for p in SWEEPS_DIR.glob(f"{prefix}*.json")})
    if not runs:
        st.caption("No per-run JSON files yet.")
    else:
        names = [p.stem for p in runs]
        default_idx = next((i for i, n in enumerate(names) if n == model_t2), 0)
        choice = st.selectbox("Run", names, index=default_idx)
        st.json(json.loads((SWEEPS_DIR / f"{choice}.json").read_text(encoding="utf-8")))

with st.expander("GPU summary (Levels 1–3)"):
    if GPU_SUMMARY.exists():
        st.markdown(GPU_SUMMARY.read_text(encoding="utf-8"))
    else:
        st.caption("No RESULTS_GPU_SUMMARY.md found.")

if MONITOR_LOG.exists():
    with st.expander("Monitor log (tail)"):
        st.code("\n".join(MONITOR_LOG.read_text(encoding="utf-8").splitlines()[-25:]), language="text")
