import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
DATA_PATH = ROOT / "data" / "raw" / "mock_process_sequences.csv"


@st.cache_data
def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


metrics = load_metrics(METRICS_PATH)
df = load_data(DATA_PATH)

st.set_page_config(
    page_title="Industrial: Process Unfolding Demo",
    layout="wide",
)

st.title("Industrial: Models that learn how processes unfold")
st.write(
    "Baseline process-monitoring demo using synthetic telemetry. "
    "Tomorrow we will replace mock data with the real case data."
)

st.info(
    "Hackathon scaffold: data loader, baseline metric evidence, scenario input, "
    "decision policy, explanation, and data preview are wired."
)

st.caption(f"Loaded metrics from: {METRICS_PATH}")
st.caption(f"Loaded data from: {DATA_PATH}")

st.subheader("Baseline evidence")
if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", metrics.get("rows", "n/a"))
    col2.metric("Accuracy", metrics.get("accuracy", "n/a"))
    col3.metric("F1 anomaly", metrics.get("f1_anomaly", "n/a"))
    st.caption(metrics.get("baseline", ""))
else:
    st.warning("No metrics found. Run: make train-baseline")

st.subheader("Scenario")

if not df.empty:
    recipes = sorted(df["recipe"].dropna().unique())
    steps = sorted(df["step_name"].dropna().unique())
else:
    recipes = ["A", "B", "C"]
    steps = ["load", "heat", "coat", "cool", "inspect", "pack"]

left, right = st.columns(2)

with left:
    recipe = st.selectbox("Recipe", recipes)
    step_name = st.selectbox("Current process step", steps)
    temperature = st.number_input("Temperature", value=180.0, step=1.0)
    pressure = st.number_input("Pressure", value=2.2, step=0.1)
    vibration = st.slider("Vibration", 0.0, 1.0, 0.32)
    tool_wear = st.slider("Tool wear", 0.0, 1.0, 0.18)
    duration_sec = st.number_input("Step duration seconds", value=60.0, step=5.0)

with right:
    st.write("### Recommendation")

    anomaly_signals = {
        "temperature_high": temperature > 185,
        "pressure_high": pressure > 2.7,
        "vibration_high": vibration > 0.55,
        "tool_wear_high": tool_wear > 0.65,
        "duration_long": duration_sec > 95,
    }
    anomaly_score = sum(anomaly_signals.values())

    if anomaly_score >= 3:
        decision = "ANOMALY REVIEW"
        confidence = "High"
    elif anomaly_score >= 1:
        decision = "WATCH"
        confidence = "Medium"
    else:
        decision = "NORMAL"
        confidence = "High"

    expected_next_step = "unknown"
    if not df.empty:
        matching = df[(df["recipe"] == recipe) & (df["step_name"] == step_name)]
        if not matching.empty:
            expected_next_step = matching["next_step"].mode().iloc[0]

    st.metric("Decision", decision)
    st.metric("Anomaly pressure", f"{anomaly_score}/5")
    st.metric("Confidence", confidence)
    st.metric("Expected next step", expected_next_step)

    st.write("### Explanation")
    st.write(
        "The rule baseline checks telemetry pressure from temperature, pressure, "
        "vibration, tool wear, and step duration. The real event data can replace "
        "these synthetic signals tomorrow without changing the demo flow."
    )

    st.write("### Policy trace")
    st.json(anomaly_signals)

st.subheader("Data sample")
if not df.empty:
    st.dataframe(df.head(20), width="stretch")
else:
    st.warning("Data sample not loaded. Check data/raw/mock_process_sequences.csv")
