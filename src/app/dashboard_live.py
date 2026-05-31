"""Live interactive demo panel (Tasks 1–3 on a dev sequence)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.dashboard_data import MODELS_DIR, REPO_ROOT, load_json, repo_path
from src.data import FAMILIES, load_all_families
from src.data.step_metadata import describe_step
from src.eval.metrics import normalized_edit_distance, token_accuracy
from src.eval.rule_validator import classify_sequence
from src.ml import load_sequence_model

SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"
METRIC_CANDIDATES = {
    "ngram_baseline.pkl": REPO_ROOT / "artifacts" / "ngram_metrics.json",
}


def _available_models() -> list[Path]:
    defaults = [
        MODELS_DIR / "h_mod_nosched_mrr.pt.best",
        MODELS_DIR / "g_drop15_nosched_t2.pt.best",
        REPO_ROOT / "models" / "ngram_baseline.pkl",
    ]
    found = [p for p in defaults if p.exists()]
    for pattern in ("*.pt.best", "*.pt", "*.pkl"):
        for p in sorted(MODELS_DIR.glob(pattern)):
            if p not in found:
                found.append(p)
    return found


@st.cache_resource(show_spinner="Loading model…")
def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"Missing: {repo_path(p)}"
    try:
        return load_sequence_model(p), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _sequences():
    return load_all_families()


@st.cache_data(show_spinner=False)
def _dev_ids() -> dict[str, list[str]]:
    p = SPLITS_DIR / "dev_ids.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def render_live_demo() -> None:
    """Interactive before/after demo on one sequence."""
    paths = _available_models()
    labels = [repo_path(p) for p in paths] or ["models/ngram_baseline.pkl"]
    choice = st.selectbox("Checkpoint", labels, key="live_model")
    model_path = REPO_ROOT / choice
    model, err = _load_model(str(model_path))
    if err or model is None:
        st.error(err or "Could not load model")
        return

    sequences = _sequences()
    dev_ids = _dev_ids()
    metrics_path = None
    if model_path.suffix == ".pkl":
        metrics_path = METRIC_CANDIDATES.get(model_path.name)
    else:
        run_name = model_path.name.replace(".pt.best", "").replace(".pt", "")
        candidate = REPO_ROOT / "artifacts" / "sweeps" / f"{run_name}.json"
        metrics_path = candidate if candidate.exists() else None

    c1, c2, c3 = st.columns(3)
    with c1:
        family = st.selectbox("Family", list(FAMILIES), key="live_family")
    with c2:
        pool = dev_ids.get(family) or sorted(sequences.get(family, {}).keys())
        sid = st.selectbox(f"Sequence ({len(pool)} available)", pool, key="live_sid")
    with c3:
        prefix_pct = st.select_slider("Prefix %", 60, 80, 60, key="live_pct")

    full = sequences[family][sid] if sid else []
    if not full:
        st.warning("No sequence data.")
        return

    prefix_len = max(1, int(len(full) * prefix_pct / 100))
    prefix = full[:prefix_len]
    gold_cont = full[prefix_len:]
    gold_next = gold_cont[0] if gold_cont else None

    st.markdown("### Task 1 — next-step top-5")
    topk = model.predict_topk(family, prefix, k=5)
    rank = next((i + 1 for i, t in enumerate(topk) if t == gold_next), None)
    st.dataframe(pd.DataFrame({
        "rank": range(1, len(topk) + 1),
        "step": topk,
        "is_gold": [t == gold_next for t in topk],
    }), hide_index=True, width="stretch")
    st.metric("Gold rank in top-5", rank or "miss")

    st.markdown("### Task 2 — completion")
    completed = model.complete(family, list(prefix), max_steps=400)
    pred_cont = completed[prefix_len:]
    tok_acc = token_accuracy(pred_cont, gold_cont) if gold_cont else 0.0
    ned = normalized_edit_distance(pred_cont, gold_cont) if gold_cont else 0.0
    a, b, c = st.columns(3)
    a.metric("Token acc", f"{tok_acc:.3f}")
    b.metric("NED", f"{ned:.3f}")
    c.metric("Pred / gold len", f"{len(pred_cont)}/{len(gold_cont)}")

    st.markdown("### Task 3 — rule validator")
    cf, cp = classify_sequence(full), classify_sequence(completed)
    v1, v2 = st.columns(2)
    with v1:
        st.write("Gold sequence:", "VALID" if cf["valid"] else f"INVALID ({cf['primary_rule']})")
    with v2:
        st.write("Completion:", "VALID" if cp["valid"] else f"INVALID ({cp['primary_rule']})")
    st.caption("Task 3 submission SCORE uses T1 LM log-prob; detection uses this validator.")

    payload = load_json(metrics_path) if metrics_path else None
    if payload:
        with st.expander("Dev eval metrics for this checkpoint"):
            m = payload.get("metrics", payload)
            t1 = m.get("task1_next_step", {}).get("overall", {})
            t2 = m.get("task2_completion", {}).get("overall", {})
            t3 = m.get("task3_anomaly", {})
            st.json({"task1": t1, "task2": t2, "task3": t3})
