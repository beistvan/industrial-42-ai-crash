#!/usr/bin/env python3
"""Generate synthetic pre-hack data for the Infineon industrial sequence track."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STEPS = ["load", "heat", "coat", "cool", "inspect", "pack"]


def make_data(runs: int = 400, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for run in range(runs):
        recipe = rng.choice(["A", "B", "C"], p=[0.45, 0.35, 0.20])
        wear = rng.uniform(0, 1)
        drift = rng.normal(0, 0.5)
        bad_run = rng.random() < 0.14 + 0.20 * wear
        for i, step in enumerate(STEPS):
            base_temp = {"load": 25, "heat": 180, "coat": 145, "cool": 70, "inspect": 35, "pack": 25}[step]
            base_pressure = {"load": 1.0, "heat": 2.4, "coat": 2.0, "cool": 1.4, "inspect": 1.0, "pack": 1.0}[step]
            anomaly = int(bad_run and (i >= rng.integers(1, len(STEPS))))
            temperature = base_temp + drift + 16 * anomaly + rng.normal(0, 4)
            pressure = base_pressure + 0.5 * anomaly + rng.normal(0, 0.1)
            vibration = 0.2 + 0.9 * wear + 0.7 * anomaly + rng.normal(0, 0.08)
            duration_sec = max(8, rng.normal(35 + 12 * i + 20 * anomaly, 6))
            next_step = STEPS[i + 1] if i + 1 < len(STEPS) else "end"
            quality_label = "fail" if bad_run else "pass"
            rows.append({
                "run_id": f"run_{run:04d}",
                "recipe": recipe,
                "step_index": i,
                "step_name": step,
                "timestamp": (start + pd.Timedelta(minutes=run * 8 + i * 2)).isoformat(),
                "temperature": round(temperature, 3),
                "pressure": round(pressure, 3),
                "vibration": round(vibration, 3),
                "tool_wear": round(wear, 3),
                "duration_sec": round(duration_sec, 3),
                "next_step": next_step,
                "anomaly": anomaly,
                "quality_label": quality_label,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_data(args.runs, args.seed)
    out = out_dir / "mock_process_sequences.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows across {args.runs} runs")


if __name__ == "__main__":
    main()
