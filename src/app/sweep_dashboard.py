"""Streamlit sweep dashboard — share Leonardo GPU sweep findings with teammates.

Reads:
    artifacts/sweeps/LEADERBOARD.csv     (sweep summary table)
    artifacts/sweeps/LEADERBOARD.md      (markdown view)
    artifacts/sweeps/m_*.json            (per-run metrics)
    RESULTS_GPU_SUMMARY.md               (friend's local baseline doc, optional)

Run:
    streamlit run src/app/sweep_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_DIR = REPO_ROOT / "artifacts" / "sweeps"
LEADERBOARD_CSV = SWEEPS_DIR / "LEADERBOARD.csv"
LEADERBOARD_MD = SWEEPS_DIR / "LEADERBOARD.md"
GPU_SUMMARY = REPO_ROOT / "RESULTS_GPU_SUMMARY.md"
MODELS_DIR = REPO_ROOT / "models" / "sweeps"

# Recommended winner per analysis: best balance of Task 1 + Task 2,
# still-improving at epoch cap (best_epoch == 50), uses 1x extras.
RECOMMENDED = "m_real_extras_1x"
ALTERNATE = "m_drop15"

st.set_page_config(page_title="Infineon — Sweep Findings", layout="wide")
st.title("Infineon — Leonardo Sweep Findings")
st.caption(
    "Shareable view of the GPU sweep on Leonardo (12 medium-config transformer runs, 50 epochs). "
    "Use this to pick the submission checkpoint and align with teammates."
)

if not LEADERBOARD_CSV.exists():
    st.error(f"No leaderboard at {LEADERBOARD_CSV.relative_to(REPO_ROOT)}. "
             "Pull artifacts from Leonardo first (`scripts/leonardo/copy_from_leonardo.sh`).")
    st.stop()

df = pd.read_csv(LEADERBOARD_CSV)
df = df.sort_values("task1_mrr", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------- headline
top = df.iloc[0]
rec_row = df[df["run"] == RECOMMENDED]
rec = rec_row.iloc[0] if not rec_row.empty else top

st.subheader("Recommended submission")
c = st.columns(5)
c[0].metric("Pick", RECOMMENDED)
c[1].metric("Task 1 Top-1", f"{rec['task1_top1']:.4f}")
c[2].metric("Task 1 MRR", f"{rec['task1_mrr']:.4f}")
c[3].metric("Task 2 token-acc", f"{rec['task2_token_acc']:.4f}")
c[4].metric("Task 2 NED", f"{rec['task2_ned']:.4f}")

ckpt = MODELS_DIR / f"{RECOMMENDED}.pt.best"
if ckpt.exists():
    st.success(f"Checkpoint ready: `{ckpt.relative_to(REPO_ROOT)}` "
               f"({ckpt.stat().st_size / 1e6:.1f} MB)")
else:
    st.warning(f"Checkpoint missing: `{ckpt.relative_to(REPO_ROOT)}`. "
               "Rsync from Leonardo or pick another run from the table.")

st.info(
    f"**Why this pick:** tied #1 on Top-1 with `m_drop15`, "
    "best Task-2 token-acc among the top group, and still climbing at epoch 50 — "
    f"more epochs would help. Alternate: `{ALTERNATE}` (marginally higher MRR by 0.0003)."
)

# --------------------------------------------------------------- leaderboard table
st.subheader(f"Leaderboard ({len(df)} runs kept)")

display = df.copy()
display.insert(0, "rank", display.index + 1)
display["pick"] = display["run"].map(
    lambda r: "🥇 recommended" if r == RECOMMENDED
    else ("🥈 alternate" if r == ALTERNATE else "")
)
display["extras"] = display["extras"].fillna("none").replace({"None": "none"})
for col in ("task1_top1", "task1_top5", "task1_mrr",
            "task2_token_acc", "task2_ned", "task3_rule_attr"):
    if col in display.columns:
        display[col] = display[col].round(4)
display = display[["rank", "pick", "run", "task1_top1", "task1_top5", "task1_mrr",
                   "task2_token_acc", "task2_ned", "task3_rule_attr",
                   "best_epoch", "train_seconds", "extras"]]
st.dataframe(display, width="stretch", hide_index=True)

# --------------------------------------------------------------- charts
st.subheader("Visual comparison")
chart_cols = st.columns(2)
with chart_cols[0]:
    st.markdown("**Task 1 — MRR (higher is better)**")
    st.bar_chart(df.set_index("run")[["task1_mrr"]].sort_values("task1_mrr"))
with chart_cols[1]:
    st.markdown("**Task 2 — token accuracy (higher is better)**")
    st.bar_chart(df.set_index("run")[["task2_token_acc"]].sort_values("task2_token_acc"))

# --------------------------------------------------------------- key findings
st.subheader("Key findings to share")
st.markdown(
    """
1. **Models are plateaued.** Top-1 spread across 12 configs is only ~2pp (0.728–0.747).
   We're at the medium-architecture ceiling, not in *"the hyperparameter matters"* territory.
2. **Top-5 = 1.0 for every run.** The right answer is always in the top 5 — Task 1 is a
   re-ranking problem from here.
3. **Task 3 rule-attr = 0.69 for ALL 12 runs.** The model checkpoint doesn't drive Task 3 —
   the rule validator does. Improving the model won't move this number. To lift Task 3 AUC,
   `predict_submission.py` now emits a continuous SCORE column.
4. **Synthetic data helps modestly. 2x is worse than 1x** (diminishing returns).
   `m_real_extras_1x` beats `m_real_extras_2x`.
5. **Three top runs were still climbing at epoch 50** (`m_real_extras_1x`, `m_warmup_1k`,
   `m_real_only`). A 100-epoch finalist run on the leader could push numbers further.
    """
)

# --------------------------------------------------------------- next steps
st.subheader("Next steps")
st.markdown(
    """
1. **Push code to Leonardo** with the fixed `predict_submission.py` (judge format):
   `bash scripts/leonardo/copy_to_leonardo.sh`
2. **Generate submission** on Leonardo with `m_real_extras_1x.pt.best`,
   `--beam-width 5 --rule-constrained --candidate-pool 5` against `EVAL_DATA/*.csv`.
3. **Rsync `extras/results_submission_v1/` back** to Mac and submit via the Tally form.
4. **In parallel:** kick off 100-epoch finalist run on `m_real_extras_1x` + `m_drop15`,
   re-enable AMP, `eval_task2_every: 10`.
    """
)

# --------------------------------------------------------------- raw artifacts
with st.expander("Raw leaderboard (Markdown view)"):
    if LEADERBOARD_MD.exists():
        st.markdown(LEADERBOARD_MD.read_text(encoding="utf-8"))
    else:
        st.caption("Markdown leaderboard not present.")

with st.expander("Per-run metrics JSON"):
    # Cover shortlist (m_*), finalists (f_*), fine-grid (g_*) runs.
    runs = sorted({p for prefix in ("m_", "f_", "g_")
                   for p in SWEEPS_DIR.glob(f"{prefix}*.json")})
    if not runs:
        st.caption("No per-run JSON files found.")
    else:
        choice = st.selectbox("Run", [p.stem for p in runs],
                              index=next((i for i, p in enumerate(runs)
                                          if p.stem == RECOMMENDED), 0))
        path = SWEEPS_DIR / f"{choice}.json"
        st.json(json.loads(path.read_text(encoding="utf-8")))

with st.expander("Friend's local baseline summary (n-gram vs small transformer)"):
    if GPU_SUMMARY.exists():
        st.markdown(GPU_SUMMARY.read_text(encoding="utf-8"))
    else:
        st.caption("No RESULTS_GPU_SUMMARY.md found.")
