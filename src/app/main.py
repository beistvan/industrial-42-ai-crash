"""Streamlit demo for the Infineon industrial sequence track.

Visualizes the n-gram baseline on the real Infineon training_data:

  - Task 1 — next-step prediction (top-k for a chosen prefix)
  - Task 2 — sequence completion (greedy roll-out vs. gold)
  - Task 3 — anomaly detection (rule validator on full sequence)

Run:
    make smoke         # builds dev split + trains n-gram + writes metrics
    make run-demo      # this app
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import FAMILIES, load_all_families  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    normalized_edit_distance,
    token_accuracy,
)
from src.eval.rule_validator import classify_sequence  # noqa: E402
from src.ml import NGramBaseline  # noqa: E402

METRICS_PATH = REPO_ROOT / "artifacts" / "ngram_metrics.json"
MODEL_PATH = REPO_ROOT / "models" / "ngram_baseline.pkl"
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"

st.set_page_config(
    page_title="Industrial: Process Unfolding Demo (n-gram baseline)",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading n-gram model...")
def load_model(path: Path) -> NGramBaseline | None:
    if not path.exists():
        return None
    return NGramBaseline.load(path)


@st.cache_data(show_spinner="Loading real Infineon training_data...")
def load_sequences() -> dict:
    return load_all_families()


@st.cache_data(show_spinner=False)
def load_dev_ids(path: Path) -> dict[str, list[str]]:
    p = path / "dev_ids.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


metrics_payload = load_metrics(METRICS_PATH)
model = load_model(MODEL_PATH)
sequences = load_sequences() if model is not None else {}
dev_ids = load_dev_ids(SPLITS_DIR)

st.title("Industrial — Models that learn how processes unfold")
st.write(
    "Live demo of the **n-gram suffix-backoff** baseline trained on real "
    "Infineon `training_data/` (MOSFET / IGBT / IC). Replaces the earlier "
    "mock-data RandomForest baseline."
)

if not metrics_payload or model is None:
    st.error(
        "No trained n-gram model or metrics found. Run the real-data pipeline first:\n\n"
        "```bash\nmake smoke\n```\n\n"
        "That builds the dev split, trains the n-gram baseline on the real "
        "Infineon `training_data/`, and writes `artifacts/ngram_metrics.json`."
    )
    st.stop()

run_id = metrics_payload.get("run_id", "—")
model_info = metrics_payload.get("model", {})
metrics = metrics_payload.get("metrics", {})
t1 = metrics.get("task1_next_step", {}).get("overall", {})
t2 = metrics.get("task2_completion", {}).get("overall", {})
t3 = metrics.get("task3_anomaly", {})

st.caption(
    f"Run `{run_id}` · model `{model_info.get('model','?')}` · "
    f"max_order={model_info.get('max_order','?')} · "
    f"fit {metrics_payload.get('fit_seconds','?')}s · "
    f"eval {metrics_payload.get('eval_seconds','?')}s"
)

# ============================================================ headline metrics
st.subheader("Baseline evidence — dev split (real Infineon data)")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Task 1 — next-step prediction**")
    a, b = st.columns(2)
    a.metric("Top-1", f"{t1.get('top1', 0):.3f}")
    b.metric("Top-5", f"{t1.get('top5', 0):.3f}")
    a.metric("Top-3", f"{t1.get('top3', 0):.3f}")
    b.metric("MRR", f"{t1.get('mrr', 0):.3f}")
    st.caption(f"n={t1.get('n', 0)} dev items")

with c2:
    st.markdown("**Task 2 — sequence completion**")
    a, b = st.columns(2)
    a.metric("Exact match", f"{t2.get('exact_match_rate', 0):.3f}")
    b.metric("Token acc", f"{t2.get('token_accuracy', 0):.3f}")
    a.metric("Norm edit dist", f"{t2.get('normalized_edit_distance', 0):.3f}")
    st.caption(f"n={t2.get('n', 0)} dev sequences")

with c3:
    st.markdown("**Task 3 — anomaly detection**")
    a, b = st.columns(2)
    a.metric("F1 (invalid)", f"{t3.get('f1_invalid', 0):.3f}")
    b.metric("Precision", f"{t3.get('precision_invalid', 0):.3f}")
    a.metric("Recall", f"{t3.get('recall_invalid', 0):.3f}")
    b.metric("Rule attribution", f"{t3.get('rule_attribution_accuracy', 0):.3f}")
    st.caption(f"n={t3.get('n', 0)} dev sequences")

pf1 = metrics.get("task1_next_step", {}).get("per_family", {})
pf2 = metrics.get("task2_completion", {}).get("per_family", {})
if pf1 or pf2:
    rows = []
    for fam in FAMILIES:
        a = pf1.get(fam, {})
        b = pf2.get(fam, {})
        rows.append({
            "family": fam,
            "task1_n": a.get("n", 0),
            "top1": round(a.get("top1", 0), 3),
            "top3": round(a.get("top3", 0), 3),
            "top5": round(a.get("top5", 0), 3),
            "mrr": round(a.get("mrr", 0), 3),
            "task2_n": b.get("n", 0),
            "token_acc": round(b.get("token_accuracy", 0), 3),
            "exact_match": round(b.get("exact_match_rate", 0), 3),
            "norm_edit_dist": round(b.get("normalized_edit_distance", 0), 3),
        })
    with st.expander("Per-family breakdown", expanded=False):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.divider()

# ============================================================ live demo
st.subheader("Live demo — n-gram baseline on a real sequence")

ctrl = st.columns([1, 1, 2])
with ctrl[0]:
    family = st.selectbox("Family", list(FAMILIES), index=0)
with ctrl[1]:
    fam_dev = dev_ids.get(family, [])
    fam_all = sorted(sequences.get(family, {}).keys())
    seq_pool = fam_dev or fam_all
    sid = st.selectbox(
        f"Sequence (dev split, {len(seq_pool)} available)",
        seq_pool,
        index=0 if seq_pool else None,
    )
with ctrl[2]:
    full = sequences[family][sid] if sid else []
    prefix_pct = st.slider(
        "Prefix length (% of full sequence)",
        min_value=10, max_value=95, value=50,
    )
    prefix_len = max(1, int(len(full) * prefix_pct / 100))

if not full:
    st.warning("No sequences available for this family.")
    st.stop()

prefix = full[:prefix_len]
gold_continuation = full[prefix_len:]

st.markdown(
    f"Sequence `{sid}` · family **{family}** · total steps **{len(full)}** · "
    f"prefix **{prefix_len}**, gold continuation **{len(gold_continuation)}**"
)

# Task 1: next-step top-5
st.markdown("### Task 1 — next-step top-5")
topk = model.predict_topk(family, prefix, k=5)
gold_next = gold_continuation[0] if gold_continuation else None
rank = None
if gold_next is not None:
    for i, tok in enumerate(topk):
        if tok == gold_next:
            rank = i + 1
            break

t1_df = pd.DataFrame({
    "rank": list(range(1, len(topk) + 1)),
    "predicted_next_step": topk,
    "is_gold": [tok == gold_next for tok in topk],
})
left, right = st.columns([2, 1])
with left:
    st.dataframe(t1_df, width="stretch", hide_index=True)
with right:
    st.metric("Gold next step", gold_next or "—")
    st.metric("Rank of gold in top-5", rank if rank else "miss")

# Task 2: completion
st.markdown("### Task 2 — greedy sequence completion")
completed = model.complete(family, list(prefix), max_steps=400)
pred_continuation = completed[prefix_len:]

if gold_continuation:
    tok_acc = token_accuracy(pred_continuation, gold_continuation)
    ned = normalized_edit_distance(pred_continuation, gold_continuation)
    exact = pred_continuation == gold_continuation
else:
    tok_acc = ned = 0.0
    exact = False

m_cols = st.columns(4)
m_cols[0].metric("Predicted len", len(pred_continuation))
m_cols[1].metric("Gold len", len(gold_continuation))
m_cols[2].metric("Token acc", f"{tok_acc:.3f}")
m_cols[3].metric("Norm edit dist", f"{ned:.3f}")
if exact and gold_continuation:
    st.success("Exact match for the continuation.")
elif gold_continuation:
    st.info("Not an exact match — see token-level diff below.")

max_len = max(len(pred_continuation), len(gold_continuation))
diff_rows = []
for i in range(max_len):
    p = pred_continuation[i] if i < len(pred_continuation) else ""
    g = gold_continuation[i] if i < len(gold_continuation) else ""
    diff_rows.append({"i": i, "predicted": p, "gold": g, "match": p == g and p != ""})
with st.expander("Token-by-token diff (predicted vs gold continuation)", expanded=False):
    st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)

with st.expander("Full predicted sequence (prefix + completion)", expanded=False):
    st.code(" → ".join(completed), language="text")

# Task 3: anomaly
st.markdown("### Task 3 — anomaly detection (rule validator)")
classify_full = classify_sequence(full)
classify_pred = classify_sequence(completed)

a_cols = st.columns(2)
with a_cols[0]:
    st.markdown("**Full gold sequence**")
    if classify_full["valid"]:
        st.success("VALID — no rule violations.")
    else:
        st.error(f"INVALID — primary rule: `{classify_full['primary_rule']}`")
    st.write("Triggered rules:", classify_full["rules"] or "—")

with a_cols[1]:
    st.markdown("**Model's completion (prefix + predicted)**")
    if classify_pred["valid"]:
        st.success("VALID — no rule violations.")
    else:
        st.error(f"INVALID — primary rule: `{classify_pred['primary_rule']}`")
    st.write("Triggered rules:", classify_pred["rules"] or "—")

st.caption(
    "Task 3 uses the official Infineon rule validator "
    "(`data/raw/infineon/training_data/generate_sequences.py::validate_sequence`)."
)

with st.expander("Raw `artifacts/ngram_metrics.json`", expanded=False):
    st.json(metrics_payload)
