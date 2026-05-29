"""Streamlit eval dashboard.

Shows the current `artifacts/ngram_metrics.json` plus the run history under
`artifacts/runs/` so you can see whether the latest training run improved
the Task 1 / 2 / 3 dev numbers.

Run:
    streamlit run src/app/eval_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
LATEST = ARTIFACTS / "ngram_metrics.json"
RUNS_DIR = ARTIFACTS / "runs"

st.set_page_config(page_title="Infineon — Eval Dashboard", layout="wide")
st.title("Infineon — Eval Dashboard")
st.caption("Reads `artifacts/ngram_metrics.json` and any `artifacts/runs/*.json` "
           "produced by `scripts/train_ngram.py`. Re-run training to refresh.")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_runs(runs_dir: Path) -> list[dict]:
    if not runs_dir.exists():
        return []
    out: list[dict] = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


latest = load_json(LATEST)
runs = load_runs(RUNS_DIR)

if latest is None and not runs:
    st.warning(
        "No metrics found. Run the pipeline first:\n\n"
        "```bash\n"
        "python scripts/make_dev_split.py --dev-per-family 100\n"
        "python scripts/train_ngram.py    --max-order 8\n"
        "```"
    )
    st.stop()

# ------------------------------------------------------------------ headline
st.subheader("Latest run")
if latest is not None:
    m = latest.get("metrics", {})
    t1 = m.get("task1_next_step", {}).get("overall", {})
    t2 = m.get("task2_completion", {}).get("overall", {})
    t3 = m.get("task3_anomaly", {})

    info_cols = st.columns(4)
    info_cols[0].metric("Run id", latest.get("run_id", "—"))
    model_info = latest.get("model", {})
    info_cols[1].metric("Model", model_info.get("model", "—"))
    info_cols[2].metric("Max order", model_info.get("max_order", "—"))
    info_cols[3].metric("Fit time (s)", latest.get("fit_seconds", "—"))

    st.markdown("**Task 1 — next-step prediction**")
    c = st.columns(4)
    c[0].metric("Top-1", f"{t1.get('top1', 0):.3f}")
    c[1].metric("Top-3", f"{t1.get('top3', 0):.3f}")
    c[2].metric("Top-5", f"{t1.get('top5', 0):.3f}")
    c[3].metric("MRR",   f"{t1.get('mrr', 0):.3f}")

    st.markdown("**Task 2 — sequence completion**")
    c = st.columns(3)
    c[0].metric("Exact match", f"{t2.get('exact_match_rate', 0):.3f}")
    c[1].metric("Token acc",   f"{t2.get('token_accuracy', 0):.3f}")
    c[2].metric("Norm edit dist", f"{t2.get('normalized_edit_distance', 0):.3f}")

    st.markdown("**Task 3 — anomaly detection**")
    c = st.columns(4)
    c[0].metric("F1 (invalid)",   f"{t3.get('f1_invalid', 0):.3f}")
    c[1].metric("Precision",      f"{t3.get('precision_invalid', 0):.3f}")
    c[2].metric("Recall",         f"{t3.get('recall_invalid', 0):.3f}")
    c[3].metric("Rule attribution", f"{t3.get('rule_attribution_accuracy', 0):.3f}")

# ----------------------------------------------------------- per-family table
if latest is not None:
    st.subheader("Per-family breakdown (latest)")
    m = latest.get("metrics", {})
    rows = []
    pf1 = m.get("task1_next_step", {}).get("per_family", {})
    pf2 = m.get("task2_completion", {}).get("per_family", {})
    for fam in ("mosfet", "igbt", "ic"):
        a = pf1.get(fam, {})
        b = pf2.get(fam, {})
        rows.append({
            "family": fam,
            "n_task1": a.get("n", 0),
            "top1": round(a.get("top1", 0), 3),
            "top3": round(a.get("top3", 0), 3),
            "top5": round(a.get("top5", 0), 3),
            "mrr":  round(a.get("mrr", 0),  3),
            "n_task2": b.get("n", 0),
            "token_acc": round(b.get("token_accuracy", 0), 3),
            "norm_edit_dist": round(b.get("normalized_edit_distance", 0), 3),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --------------------------------------------------------- improvement over time
st.subheader(f"Run history ({len(runs)} run{'s' if len(runs) != 1 else ''})")
if runs:
    hist_rows = []
    for r in runs:
        rm = r.get("metrics", {})
        t1 = rm.get("task1_next_step", {}).get("overall", {})
        t2 = rm.get("task2_completion", {}).get("overall", {})
        t3 = rm.get("task3_anomaly", {})
        hist_rows.append({
            "run_id": r.get("run_id", "—"),
            "timestamp": r.get("timestamp", ""),
            "max_order": r.get("model", {}).get("max_order", ""),
            "top1": round(t1.get("top1", 0), 4),
            "top5": round(t1.get("top5", 0), 4),
            "mrr":  round(t1.get("mrr",  0), 4),
            "token_acc": round(t2.get("token_accuracy", 0), 4),
            "norm_edit_dist": round(t2.get("normalized_edit_distance", 0), 4),
            "f1_invalid": round(t3.get("f1_invalid", 0), 4),
            "rule_attr":  round(t3.get("rule_attribution_accuracy", 0), 4),
        })
    hist = pd.DataFrame(hist_rows).sort_values("timestamp")
    st.dataframe(hist, width="stretch", hide_index=True)

    st.markdown("**Top-1 / Top-5 / MRR over runs**")
    chart_df = hist.set_index("run_id")[["top1", "top5", "mrr"]]
    st.line_chart(chart_df)

    st.markdown("**Task 2 — token accuracy + edit distance over runs**")
    chart_df2 = hist.set_index("run_id")[["token_acc", "norm_edit_dist"]]
    st.line_chart(chart_df2)

    if len(hist) >= 2:
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        st.markdown("**Latest vs. previous run**")
        delta_cols = st.columns(4)
        for i, key, fmt in [
            (0, "top1", "+.3f"),
            (1, "top5", "+.3f"),
            (2, "mrr", "+.3f"),
            (3, "f1_invalid", "+.3f"),
        ]:
            delta = last[key] - prev[key]
            delta_cols[i].metric(
                key,
                f"{last[key]:.3f}",
                delta=f"{delta:{fmt}}",
                delta_color="normal" if delta >= 0 else "inverse",
            )
else:
    st.info("Only one run so far — train again with different `--max-order` "
            "or more data to populate the history table.")

# ----------------------------------------------------------- raw json
with st.expander("Raw latest metrics JSON"):
    if latest is not None:
        st.json(latest)
