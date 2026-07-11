"""
One anomaly detection model: Isolation Forest.

Isolation Forest works by randomly splitting the data over and over.
Points that are easy to isolate in very few splits -- because they're far
from everything else -- get a high anomaly score. No labels required.

Severity is just the anomaly score bucketed into Red / Yellow / Green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["spend_ratio", "txn_count_1h", "is_night", "merchant_mismatch"]

RED_THRESHOLD = 0.75
YELLOW_THRESHOLD = 0.40


def score_transactions(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
    model.fit(X_scaled)

    # decision_function: higher = more normal. Flip it and squeeze into [0, 1]
    # so higher anomaly_score always means "more suspicious".
    raw = -model.decision_function(X_scaled)
    lo, hi = raw.min(), raw.max()
    anomaly_score = (raw - lo) / (hi - lo) if hi > lo else np.zeros(len(raw))

    out = df.copy()
    out["anomaly_score"] = anomaly_score
    out["severity"] = classify_severity(anomaly_score)
    return out


def classify_severity(score) -> np.ndarray | str:
    is_scalar = np.isscalar(score)
    arr = np.atleast_1d(np.asarray(score, dtype=float))
    out = np.where(
        arr >= RED_THRESHOLD, "Red",
        np.where(arr >= YELLOW_THRESHOLD, "Yellow", "Green"),
    )
    return out[0] if is_scalar else out
