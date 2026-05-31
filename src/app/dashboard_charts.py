"""Interactive Plotly training curves with tail-focused defaults."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _find_loss_focus_start(hist: pd.DataFrame, col: str = "train_loss") -> int:
    """First epoch where loss is near its run minimum (flat tail region)."""
    if hist.empty or col not in hist.columns:
        return 1
    y_min = float(hist[col].min())
    threshold = max(y_min * 1.2, y_min + 0.2)
    for _, row in hist.iterrows():
        if float(row[col]) <= threshold:
            return int(row["epoch"])
    n = len(hist)
    return max(1, int(hist["epoch"].iloc[max(0, int(n * 0.5) - 1)]))


def _y_range(values: Iterable[float], *, pad_frac: float = 0.08) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and pd.notna(v)]
    if not vals:
        return 0.0, 1.0
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        pad = abs(vmin) * 0.05 or 0.01
        return vmin - pad, vmax + pad
    span = vmax - vmin
    return vmin - span * pad_frac, vmax + span * pad_frac


def render_epoch_controls(hist: pd.DataFrame, *, run_key: str) -> tuple[int, int]:
    """Epoch range picker — defaults to tail focus where loss/metrics stabilize."""
    emin = int(hist["epoch"].min())
    emax = int(hist["epoch"].max())
    tail_start = _find_loss_focus_start(hist)
    mid_start = max(emin, int(emax * 0.5))

    st.caption(
        "Loss drops quickly in early epochs then flattens — use **Tail focus** or drag the "
        "range slider under each chart to zoom."
    )
    preset = st.radio(
        "Epoch window",
        options=["Tail focus", "Last 50%", "Full run"],
        horizontal=True,
        index=0,
        key=f"epoch_preset_{run_key}",
    )
    if preset == "Full run":
        preset_start, preset_end = emin, emax
    elif preset == "Last 50%":
        preset_start, preset_end = mid_start, emax
    else:
        preset_start, preset_end = tail_start, emax

    preset_state_key = f"epoch_preset_applied_{run_key}"
    if st.session_state.get(preset_state_key) != preset:
        st.session_state[f"epoch_from_{run_key}"] = preset_start
        st.session_state[f"epoch_to_{run_key}"] = preset_end
        st.session_state[preset_state_key] = preset

    c1, c2 = st.columns(2)
    with c1:
        start = st.slider(
            "From epoch",
            emin,
            emax,
            preset_start,
            key=f"epoch_from_{run_key}",
        )
    with c2:
        end = st.slider(
            "To epoch",
            emin,
            emax,
            preset_end,
            key=f"epoch_to_{run_key}",
        )
    if start > end:
        start, end = end, start
    return start, end


def _slice_hist(hist: pd.DataFrame, epoch_start: int, epoch_end: int) -> pd.DataFrame:
    return hist[(hist["epoch"] >= epoch_start) & (hist["epoch"] <= epoch_end)].copy()


def _plotly_lines(
    hist: pd.DataFrame,
    *,
    columns: list[str],
    labels: dict[str, str] | None = None,
    title: str,
    y_title: str,
    epoch_start: int,
    epoch_end: int,
) -> None:
    view = _slice_hist(hist, epoch_start, epoch_end)
    if view.empty:
        st.info("No data in selected epoch range.")
        return

    labels = labels or {}
    fig = go.Figure()
    y_vals: list[float] = []
    for col in columns:
        if col not in view.columns or view[col].isna().all():
            continue
        series = view[col].astype(float)
        y_vals.extend(series.tolist())
        fig.add_trace(
            go.Scatter(
                x=view["epoch"],
                y=series,
                mode="lines+markers",
                name=labels.get(col, col),
                marker={"size": 4},
            )
        )

    if not y_vals:
        st.info(f"No `{', '.join(columns)}` data in selected epoch range.")
        return

    ymin, ymax = _y_range(y_vals)
    fig.update_layout(
        title=title,
        height=380,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={
            "title": "Epoch",
            "rangeslider": {"visible": True},
            "range": [epoch_start - 0.5, epoch_end + 0.5],
        },
        yaxis={"title": y_title, "range": [ymin, ymax]},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_training_charts(hist: pd.DataFrame, *, run_key: str) -> None:
    """Loss + metric charts with shared epoch window and zoom-friendly y-axis."""
    epoch_start, epoch_end = render_epoch_controls(hist, run_key=run_key)

    st.markdown("**Train / dev loss**")
    _plotly_lines(
        hist,
        columns=["train_loss", "dev_loss"],
        labels={"train_loss": "Train loss", "dev_loss": "Dev loss"},
        title="",
        y_title="Loss (NLL)",
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )

    if hist["task1_mrr"].notna().any():
        st.markdown("**Task 1 dev MRR per epoch**")
        _plotly_lines(
            hist,
            columns=["task1_mrr"],
            labels={"task1_mrr": "Dev MRR"},
            title="",
            y_title="MRR",
            epoch_start=epoch_start,
            epoch_end=epoch_end,
        )

    if hist["task2_tok_acc"].notna().any():
        st.markdown("**Task 2 dev token accuracy per epoch**")
        _plotly_lines(
            hist,
            columns=["task2_tok_acc"],
            labels={"task2_tok_acc": "Token accuracy"},
            title="",
            y_title="Token accuracy",
            epoch_start=epoch_start,
            epoch_end=epoch_end,
        )
