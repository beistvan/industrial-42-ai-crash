"""Deprecated — use `make run-dashboard` (src/app/dashboard.py)."""
import streamlit as st

st.set_page_config(page_title="Deprecated", layout="wide")
st.error("This app was merged into the unified dashboard.")
st.code("make run-dashboard\n# or: streamlit run src/app/dashboard.py", language="bash")
st.stop()
