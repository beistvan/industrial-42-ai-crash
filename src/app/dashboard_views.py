"""View sections for the unified Track 1 dashboard."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.app.dashboard_charts import render_training_charts
from src.app.dashboard_data import (
    DEV_EVAL_DIR,
    EVAL_PROTOCOL,
    JUDGE_EVAL_DIR,
    LEADERBOARD_MD,
    MODELS_DIR,
    NGRAM_METRICS,
    REPO_ROOT,
    SUBMISSION_DIR,
    load_json,
    metrics_sections,
    model_config_table,
    pick_winners,
    pipeline_rows,
    progress_table,
    repo_path,
    run_json_path,
    submission_csv_rows,
    training_history_df,
    wave_label,
)
from src.app.track_context import BASELINE, LEVELS, SUBMISSION, TRACK_ALIGNMENT, TRACK_DOC_URL


def render_task_metrics(t1: dict, t2: dict, t3: dict, *, caption: str) -> None:
    st.caption(caption)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Task 1 — next-step**")
        if t1:
            st.metric("MRR", f"{t1.get('mrr', 0):.4f}")
            st.metric("Top-1", f"{t1.get('top1', 0):.4f}")
            st.metric("Top-5", f"{t1.get('top5', 0):.4f}")
    with c2:
        st.markdown("**Task 2 — completion**")
        if t2:
            st.metric("Token acc", f"{t2.get('token_accuracy', 0):.4f}")
            st.metric("NED", f"{t2.get('normalized_edit_distance', 0):.4f}")
            st.metric("Exact match", f"{t2.get('exact_match_rate', 0):.4f}")
    with c3:
        st.markdown("**Task 3 — anomaly**")
        if t3:
            st.metric("F1 invalid", f"{t3.get('f1_invalid', 0):.4f}")
            st.metric("Rule attrib.", f"{t3.get('rule_attribution_accuracy', 0):.4f}")
            st.metric("Accuracy", f"{t3.get('accuracy', 0):.4f}")


def render_overview(df: pd.DataFrame | None) -> None:
    st.subheader("Approach")
    st.markdown(
        f"""
**Hybrid submission** for [Track 1]({TRACK_DOC_URL}):

1. **Decoder-only Transformer** (~4M params) on ~3k official + 750 synthetic fab sequences  
2. **Wave 3 modern stack** (RoPE + RMSNorm + SwiGLU) for Task 1 — winner **`{SUBMISSION['task1_run']}`**  
3. **Wave 2 T2 specialist** with prefix training + beam search — **`{SUBMISSION['task2_run']}`**  
4. **Task 3** — hybrid on the **T1 checkpoint**: official rule validator for `IS_VALID` + `PREDICTED_RULE` (with `pick_primary_rule` tie-break); teacher-forced LM log-prob for `SCORE` — **no separate Task-3 model or finetune**
        """
    )

    st.subheader("Winning models (dev holdout)")
    if df is None:
        st.warning("Run `make leonardo-leaderboard-final` first.")
        return

    t1_run, t2_run, t1_score, t2_score = pick_winners(df)
    t1_json = load_json(run_json_path(t1_run))
    _, _, _, t3_from_t1 = metrics_sections(t1_json) if t1_json else ({}, {}, {}, {})
    t3_rule_attr = t3_from_t1.get(
        "rule_attribution_accuracy",
        SUBMISSION.get("task3_rule_attr", 0.69),
    )

    m = st.columns(5)
    m[0].metric("T1 + anomaly", t1_run)
    m[1].metric("T1 MRR", f"{t1_score:.4f}")
    m[2].metric("T2 completion", t2_run)
    m[3].metric("T2 tok acc", f"{t2_score:.4f}")
    m[4].metric("T3 rule attrib.", f"{t3_rule_attr:.2f}")
    st.info(
        f"Hybrid: `{t1_run}` → nextstep + anomaly (validator + LM SCORE) · "
        f"`{t2_run}` → completion.csv"
    )

    for run in (t1_run, t2_run):
        ckpt = MODELS_DIR / f"{run}.pt.best"
        st.caption(f"{'✓' if ckpt.exists() else '✗'} {repo_path(ckpt)}")

    st.subheader("Three-task metrics vs baseline")
    st.dataframe(progress_table(df), hide_index=True, width="stretch")

    # Load winner JSONs for full 3-task panel
    t2_json = load_json(run_json_path(t2_run))
    if t1_json:
        _, t1, _, t3 = metrics_sections(t1_json)
        render_task_metrics(
            t1, {}, t3,
            caption=(
                f"T1 model `{t1_run}` on dev holdout — Task 3 uses same checkpoint "
                f"(validator + LM SCORE; rule attrib {t3.get('rule_attribution_accuracy', 0):.2f})"
            ),
        )
    if t2_json:
        _, _, t2, _ = metrics_sections(t2_json)
        render_task_metrics(
            {}, t2, {},
            caption=f"T2 model `{t2_run}` on dev holdout (prefix 60–80%, beam eval)",
        )

    baseline = load_json(NGRAM_METRICS)
    if baseline:
        _, bt1, bt2, bt3 = metrics_sections(baseline)
        with st.expander("N-gram baseline (Level 1)"):
            render_task_metrics(bt1, bt2, bt3, caption="Baseline reference")

    st.subheader("Evaluation protocol")
    st.markdown(
        f"""
| Eval set | Path | Used for |
|---|---|---|
| **Dev holdout** | `{repo_path(DEV_EVAL_DIR)}` | Training checkpoint selection, leaderboard, this dashboard |
| **Judge inputs** | `{repo_path(JUDGE_EVAL_DIR)}` | Official submission CSVs in `{repo_path(SUBMISSION_DIR)}/` |
        """
    )
    st.json(EVAL_PROTOCOL, expanded=False)

    st.subheader("Submission files")
    st.dataframe(pd.DataFrame(submission_csv_rows()), hide_index=True, width="stretch")

    st.subheader("Sweep pipeline (final)")
    st.dataframe(pd.DataFrame(pipeline_rows()), hide_index=True, width="stretch")

    with st.expander("Track 1 alignment"):
        st.markdown(TRACK_ALIGNMENT)


def render_leaderboard(df: pd.DataFrame) -> None:
    t1_run, t2_run, t1_score, t2_score = pick_winners(df)

    display = df.sort_values("task1_mrr", ascending=False).copy().reset_index(drop=True)
    display.insert(0, "rank", display.index + 1)
    display["wave"] = display["run"].map(wave_label)
    display["role"] = display["run"].map(
        lambda r: "T1/anomaly" if r == t1_run else ("T2" if r == t2_run else "")
    )
    for col in ("task1_top1", "task1_top5", "task1_mrr", "task2_token_acc", "task2_ned"):
        if col in display.columns:
            display[col] = display[col].round(4)

    st.subheader(f"Leaderboard ({len(df)} runs)")
    st.dataframe(
        display[["rank", "wave", "role", "run", "task1_mrr", "task1_top1",
                 "task2_token_acc", "task2_ned", "task3_rule_attr", "best_epoch", "train_seconds"]],
        hide_index=True,
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Task 1 MRR** (top 12)")
        st.bar_chart(display.set_index("run")[["task1_mrr"]].head(12))
    with c2:
        st.markdown("**Task 2 token acc** (top 12)")
        st.bar_chart(
            display.sort_values("task2_token_acc", ascending=False)
            .set_index("run")[["task2_token_acc"]].head(12)
        )

    if LEADERBOARD_MD.exists():
        with st.expander("Raw markdown"):
            st.markdown(LEADERBOARD_MD.read_text(encoding="utf-8"))


def render_training(df: pd.DataFrame | None) -> None:
    if df is None:
        st.warning("No leaderboard data.")
        return

    t1_run, t2_run, _, _ = pick_winners(df)
    choice = st.selectbox(
        "Inspect run",
        [t1_run, t2_run] + [r for r in df["run"].tolist() if r not in (t1_run, t2_run)],
    )
    payload = load_json(run_json_path(choice))
    if not payload:
        st.error(f"No metrics JSON for `{choice}`")
        return

    st.subheader(f"Architecture & config — `{choice}`")
    st.dataframe(model_config_table(payload), hide_index=True, width="stretch")

    arch = payload.get("model", {}).get("metadata", {}).get("train_cfg", {}).get("arch", "vanilla")
    if arch == "modern":
        st.markdown(
            "**Modern decoder:** RoPE positional encoding · RMSNorm · SwiGLU FFN · pre-norm blocks "
            "(see `src/ml/modern_transformer.py`)"
        )
    else:
        st.markdown(
            "**Vanilla decoder:** learned absolute positions · GELU FFN · post-norm "
            "(see `src/ml/transformer_model.py`)"
        )

    hist = training_history_df(payload)
    if hist.empty:
        st.info("No training history in JSON.")
        return

    st.subheader("Training curves")
    render_training_charts(hist, run_key=choice)

    best = payload.get("best", {})
    if best:
        st.success(
            f"Checkpoint saved at epoch **{best.get('epoch')}** "
            f"({best.get('metric')} = {best.get('value', 0):.4f})"
        )

    _, t1, t2, t3 = metrics_sections(payload)
    render_task_metrics(
        t1, t2, t3,
        caption=f"Final dev eval on `{repo_path(DEV_EVAL_DIR)}` after training",
    )

    with st.expander("Full metrics JSON"):
        st.json(payload)
