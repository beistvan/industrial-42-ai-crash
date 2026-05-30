"""Streamlit demo for the Infineon industrial sequence track.

Supports both implemented sequence models:

  - n-gram suffix-backoff baseline (`models/ngram_baseline.pkl`)
  - compact decoder-only Transformer (`models/transformer_*.pt`)

Run:
    make dev-split
    make train-ngram
    make train-transformer-smoke       # optional quick Transformer checkpoint
    make run-demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import FAMILIES, load_all_families  # noqa: E402
from src.eval.metrics import normalized_edit_distance, token_accuracy  # noqa: E402
from src.eval.rule_validator import _validate, classify_sequence  # noqa: E402
from src.ml import NGramBaseline, load_sequence_model  # noqa: E402

METRIC_CANDIDATES = {
    "ngram_baseline.pkl": REPO_ROOT / "artifacts" / "ngram_metrics.json",
    "transformer_small.pt": REPO_ROOT / "artifacts" / "transformer_metrics.json",
    "transformer_small_local.pt": REPO_ROOT / "artifacts" / "transformer_small_local_metrics.json",
    "transformer_small_extra_local.pt": REPO_ROOT / "artifacts" / "transformer_small_extra_local_metrics.json",
    "transformer_smoke.pt": REPO_ROOT / "artifacts" / "transformer_smoke_metrics.json",
}
DEFAULT_MODEL_PATHS = [
    REPO_ROOT / "models" / "ngram_baseline.pkl",
    REPO_ROOT / "models" / "transformer_small.pt",
    REPO_ROOT / "models" / "transformer_small_local.pt",
    REPO_ROOT / "models" / "transformer_smoke.pt",
]
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"


def _as_repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def available_model_paths() -> list[Path]:
    found = [p for p in DEFAULT_MODEL_PATHS if p.exists()]
    found.extend(p for p in sorted((REPO_ROOT / "models").glob("*.pt")) if p not in found)
    found.extend(p for p in sorted((REPO_ROOT / "models").glob("*.pkl")) if p not in found)
    found.extend(p for p in sorted((REPO_ROOT / "models" / "sweeps").glob("*.pt.best")) if p not in found)
    found.extend(p for p in sorted((REPO_ROOT / "models" / "sweeps").glob("*.pt")) if p not in found)
    return found


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading sequence model...")
def cached_load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"Model file does not exist: {p}"
    try:
        return load_sequence_model(p), None
    except Exception as exc:  # pragma: no cover - Streamlit display path.
        return None, f"Could not load model {p}: {exc}"


@st.cache_data(show_spinner="Loading real Infineon training_data...")
def cached_sequences() -> dict:
    return load_all_families()


@st.cache_data(show_spinner=False)
def cached_dev_ids(path: str) -> dict[str, list[str]]:
    p = Path(path) / "dev_ids.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def metric_path_for_model(model_path: Path) -> Path | None:
    return METRIC_CANDIDATES.get(model_path.name)


def violation_records(steps: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "rule": v.rule,
            "step_index": getattr(v, "step_index", None),
            "description": getattr(v, "description", ""),
        }
        for v in _validate(steps)
    ]


def is_ngram_model(model: Any) -> bool:
    return isinstance(model, NGramBaseline) or hasattr(model, "suffix_counts")


def explain_ngram_next_step(model: NGramBaseline, family: str, prefix: list[str], k: int = 5) -> dict:
    """Explain n-gram suffix lookup with matched suffix and counts."""
    for order in range(min(model.max_order, len(prefix)), 0, -1):
        suffix = tuple(prefix[-order:])
        counts = model.suffix_counts[family].get(suffix)
        if counts:
            top = counts.most_common(k)
            total = sum(counts.values())
            return {
                "kind": "ngram",
                "matched_suffix_order": order,
                "matched_suffix": list(suffix),
                "candidates_seen_after_suffix": total,
                "topk": [
                    {"token": t, "count": c, "prob": round(c / total, 4)}
                    for t, c in top
                ],
                "backoff": False,
            }
    counts = model.unconditional[family]
    top = counts.most_common(k)
    total = sum(counts.values()) or 1
    return {
        "kind": "ngram",
        "matched_suffix_order": 0,
        "matched_suffix": [],
        "candidates_seen_after_suffix": total,
        "topk": [
            {"token": t, "count": c, "prob": round(c / total, 4)}
            for t, c in top
        ],
        "backoff": True,
    }


def transformer_explanation(model: Any, topk: list[str]) -> dict[str, Any]:
    stats = model.stats() if hasattr(model, "stats") else {"model": type(model).__name__}
    return {
        "kind": "transformer",
        "message": (
            "Decoder-only Transformer prediction: the model conditions on "
            "<bos>, a family token, and the current prefix, then ranks the next "
            "process step by learned next-token logits."
        ),
        "topk": [{"rank": i + 1, "token": tok} for i, tok in enumerate(topk)],
        "stats": stats,
    }


def get_metrics_sections(payload: dict[str, Any]) -> tuple[dict, dict, dict, dict]:
    metrics = payload.get("metrics", payload)
    t1 = metrics.get("task1_next_step", {}).get("overall", {})
    t2 = metrics.get("task2_completion", {}).get("overall", {})
    t3 = metrics.get("task3_anomaly", {})
    return metrics, t1, t2, t3


st.set_page_config(
    page_title="Industrial: Process Unfolding Demo",
    layout="wide",
)

st.title("Industrial — Models that learn how processes unfold")
st.write(
    "Real Infineon `training_data/` demo for **next-step prediction**, "
    "**sequence completion**, and **rule-based anomaly detection**. Use the "
    "sidebar to switch between the n-gram baseline and a trained small Transformer."
)

model_paths = available_model_paths()
with st.sidebar:
    st.header("Model")
    if model_paths:
        labels = [_as_repo_path(p) for p in model_paths]
        default_idx = 0
        selected_label = st.selectbox("Checkpoint", labels, index=default_idx)
        selected_path = REPO_ROOT / selected_label
    else:
        selected_path = REPO_ROOT / "models" / "ngram_baseline.pkl"
        st.warning("No model checkpoints found under `models/`.")

    custom_path = st.text_input(
        "Custom checkpoint path",
        value="",
        placeholder="models/transformer_small.pt",
    )
    if custom_path.strip():
        selected_path = (REPO_ROOT / custom_path.strip()).resolve()

    st.caption("Useful commands")
    st.code(
        "make dev-split\n"
        "make train-ngram\n"
        "make train-transformer-smoke\n"
        "make run-demo",
        language="bash",
    )

model, load_error = cached_load_model(str(selected_path))
if load_error:
    st.error(load_error)
    st.info(
        "For the fastest local path run `make dev-split && make train-ngram`. "
        "For a CPU-safe Transformer checkpoint run `make train-transformer-smoke`."
    )
    st.stop()

sequences = cached_sequences()
dev_ids = cached_dev_ids(str(SPLITS_DIR))
model_stats = model.stats() if hasattr(model, "stats") else {"model": type(model).__name__}
metrics_path = metric_path_for_model(selected_path)
metrics_payload = load_json(str(metrics_path)) if metrics_path else {}
metrics, t1, t2, t3 = get_metrics_sections(metrics_payload)

model_name = model_stats.get("model", type(model).__name__)
st.caption(
    f"Checkpoint `{_as_repo_path(selected_path)}` · model `{model_name}`"
    + (f" · metrics `{_as_repo_path(metrics_path)}`" if metrics_path and metrics_path.exists() else "")
)

if not metrics_payload:
    st.warning(
        "Model loaded, but no matching metrics JSON was found. The live demo will still work. "
        "Run `make local-eval` for n-gram, or train/evaluate the Transformer target for metrics."
    )
else:
    st.subheader("Dev metrics — local Infineon split")
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
        a.metric("F1 invalid", f"{t3.get('f1_invalid', 0):.3f}")
        b.metric("Precision", f"{t3.get('precision_invalid', 0):.3f}")
        a.metric("Recall", f"{t3.get('recall_invalid', 0):.3f}")
        b.metric("Rule attrib.", f"{t3.get('rule_attribution_accuracy', 0):.3f}")
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
st.subheader("Live sequence demo")

ctrl = st.columns([1, 1, 2])
with ctrl[0]:
    family = st.selectbox("Family", list(FAMILIES), index=0)
with ctrl[1]:
    fam_dev = dev_ids.get(family, [])
    fam_all = sorted(sequences.get(family, {}).keys())
    seq_pool = fam_dev or fam_all
    sid = st.selectbox(
        f"Sequence ({'dev split' if fam_dev else 'all training'}, {len(seq_pool)} available)",
        seq_pool,
        index=0 if seq_pool else None,
    )
with ctrl[2]:
    full = sequences[family][sid] if sid else []
    prefix_pct = st.slider(
        "Prefix length (% of full sequence)",
        min_value=10,
        max_value=95,
        value=50,
    )
    prefix_len = max(1, int(len(full) * prefix_pct / 100)) if full else 0

if not full:
    st.warning("No sequences available for this family.")
    st.stop()

prefix = full[:prefix_len]
gold_continuation = full[prefix_len:]
st.markdown(
    f"Sequence `{sid}` · family **{family}** · total steps **{len(full)}** · "
    f"prefix **{prefix_len}**, gold continuation **{len(gold_continuation)}**"
)

# Task 1: next-step top-5.
st.markdown("### Task 1 — next-step top-5")
topk = model.predict_topk(family, prefix, k=5)
gold_next = gold_continuation[0] if gold_continuation else None
rank = next((i + 1 for i, tok in enumerate(topk) if tok == gold_next), None)

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

# Task 2: completion.
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
with st.expander("Token-by-token diff", expanded=False):
    st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)
with st.expander("Full predicted sequence", expanded=False):
    st.code(" → ".join(completed), language="text")

# Task 3: anomaly.
st.markdown("### Task 3 — anomaly detection by rule validator")
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
    st.markdown("**Model completion**")
    if classify_pred["valid"]:
        st.success("VALID — no rule violations.")
    else:
        st.error(f"INVALID — primary rule: `{classify_pred['primary_rule']}`")
    st.write("Triggered rules:", classify_pred["rules"] or "—")

st.caption(
    "The anomaly layer calls the official Infineon process-rule validator via "
    "`data/raw/infineon/training_data/generate_sequences.py::validate_sequence`."
)

# Explanation.
st.markdown("### Explanation — how this model decided")
if is_ngram_model(model):
    expl = explain_ngram_next_step(model, family, prefix, k=5)
    if expl["backoff"]:
        st.warning(
            "Backoff fired: no suffix of the prefix was seen during training, "
            f"so the model used the unconditional family distribution for `{family}`."
        )
    else:
        st.info(
            f"Matched suffix order **{expl['matched_suffix_order']}**. "
            f"There were **{expl['candidates_seen_after_suffix']}** observed "
            "training continuations after this exact suffix."
        )
    ex_l, ex_r = st.columns([1, 1])
    with ex_l:
        st.markdown("**Matched suffix**")
        st.code(" → ".join(expl["matched_suffix"]) if expl["matched_suffix"] else "(backoff)", language="text")
    with ex_r:
        st.markdown("**Top-5 with counts**")
        st.dataframe(pd.DataFrame(expl["topk"]), width="stretch", hide_index=True)
else:
    expl = transformer_explanation(model, topk)
    st.info(expl["message"])
    st.dataframe(pd.DataFrame(expl["topk"]), width="stretch", hide_index=True)
    with st.expander("Transformer stats", expanded=False):
        st.json(expl["stats"])

# Audit trace.
st.markdown("### Audit trace")
audit = {
    "input": {
        "sequence_id": sid,
        "family": family,
        "total_steps": len(full),
        "prefix_len": prefix_len,
        "gold_continuation_len": len(gold_continuation),
    },
    "model": model_stats,
    "task1_next_step": {
        "topk": topk,
        "gold_next_step": gold_next,
        "gold_rank_in_top5": rank,
    },
    "task2_completion": {
        "predicted_length": len(pred_continuation),
        "gold_length": len(gold_continuation),
        "token_accuracy": round(tok_acc, 4),
        "normalized_edit_distance": round(ned, 4),
        "exact_match": bool(exact and gold_continuation),
        "stopped_on_ship_lot": completed[-1] == "SHIP LOT" if completed else False,
    },
    "task3_anomaly": {
        "gold_sequence_valid": classify_full["valid"],
        "gold_violations": violation_records(full),
        "predicted_completion_valid": classify_pred["valid"],
        "predicted_violations": violation_records(completed),
    },
}
st.json(audit, expanded=False)

if metrics_payload:
    with st.expander("Raw metrics JSON", expanded=False):
        st.json(metrics_payload)
