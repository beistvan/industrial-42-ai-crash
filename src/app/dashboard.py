"""Unified Track 1 dashboard — metrics, approach, training curves, live demo.

Run:
    make run-dashboard
    STREAMLIT_PORT=8502 make run-dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.dashboard_data import load_leaderboard  # noqa: E402
from src.app.dashboard_live import render_live_demo  # noqa: E402
from src.app.dashboard_views import (  # noqa: E402
    render_leaderboard,
    render_overview,
    render_training,
)
from src.app.track_context import LEVELS, TRACK_DOC_URL  # noqa: E402

st.set_page_config(
    page_title="Track 1 — Process Logic Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Track 1 — Process Logic Dashboard")
st.caption(
    f"[Zero One Track 1]({TRACK_DOC_URL}) · "
    "3 tasks · hybrid submission · dev holdout + judge eval"
)

# Level progress strip
cols = st.columns(3)
for col, lvl in zip(cols, LEVELS):
    with col:
        badge = {"done": "✅", "in_progress": "🔄"}.get(lvl["status"], "⏳")
        st.markdown(f"**{lvl['level']}** {badge}")
        st.caption(lvl["metric"])

df = load_leaderboard()

tab_overview, tab_leaderboard, tab_training, tab_live = st.tabs([
    "Overview",
    "Leaderboard",
    "Training & config",
    "Live demo",
])

with tab_overview:
    render_overview(df)

with tab_leaderboard:
    if df is not None:
        render_leaderboard(df)
    else:
        st.error("Missing `artifacts/sweeps/LEADERBOARD_FINAL.csv`")

with tab_training:
    render_training(df)

with tab_live:
    st.markdown(
        "Interactive demo on the **dev holdout** — same prefix/completion setup as Track 1 judge eval."
    )
    render_live_demo()
