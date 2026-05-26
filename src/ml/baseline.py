#!/usr/bin/env python3
"""Fast baseline for the industrial sequence track."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def main() -> None:
    data_path = Path("data/raw/mock_process_sequences.csv")
    if not data_path.exists():
        raise SystemExit("Missing data/raw/mock_process_sequences.csv. Run: make generate-mock-data")
    df = pd.read_csv(data_path)
    features = ["recipe", "step_index", "step_name", "temperature", "pressure", "vibration", "tool_wear", "duration_sec"]
    target = "anomaly"
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.25, random_state=7, stratify=df[target]
    )
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["recipe", "step_name"]),
        ("num", StandardScaler(), [c for c in features if c not in {"recipe", "step_name"}]),
    ])
    clf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=80, random_state=7, max_depth=8))])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    metrics = {
        "track": "industrial",
        "rows": int(len(df)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_anomaly": round(float(f1_score(y_test, pred)), 4),
        "baseline": "RandomForest anomaly classifier on synthetic process-sequence telemetry",
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
