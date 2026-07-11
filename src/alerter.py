"""
Formats alerts for the console, logs them to JSON, prevents duplicate
alerts from a burst of transactions, and tracks simple analyst feedback.
"""
from __future__ import annotations

import json
from pathlib import Path

SEVERITY_TAG = {"Red": "[RED]", "Yellow": "[YELLOW]", "Green": "[GREEN]"}


def select_alert_candidates(scored_df):
    """
    Keeps only the highest-severity transaction per account per hour, so a
    burst of 4 rapid transactions produces 1 alert instead of 4.
    """
    rank = {"Red": 2, "Yellow": 1, "Green": 0}
    df = scored_df.copy()
    df["_rank"] = df["severity"].map(rank)
    df["_hour"] = df["timestamp"].dt.floor("h")
    df = df.sort_values(["_rank", "anomaly_score"], ascending=False)
    deduped = df.drop_duplicates(subset=["account_id", "_hour"], keep="first")
    return deduped.drop(columns=["_rank", "_hour"]).sort_values("timestamp").reset_index(drop=True)


def format_alert(alert: dict) -> str:
    tag = SEVERITY_TAG.get(alert["severity"], "[?]")
    return "\n".join([
        "-" * 60,
        f"{tag} {alert['headline']}",
        f"  Account: {alert['account_name']}  |  Amount: ${alert['amount']:,.2f}  "
        f"|  Score: {alert['anomaly_score']:.2f}  |  Urgency: {alert['urgency']}",
        f"  Tags: {', '.join(alert['tags'])}",
        f"  {alert['explanation']}",
        f"  Action: {alert['recommended_action']}",
        f"  Dispatch safe: {alert['dispatch_safe']}  ({alert['dispatch_reason']})",
        f"  Written by: {alert['generated_by']}",
        "-" * 60,
    ])


def log_alert(alert: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(path.read_text()) if path.exists() else []
    log = [a for a in log if a["txn_id"] != alert["txn_id"]]
    log.append(alert)
    path.write_text(json.dumps(log, indent=2, default=str))


def evaluate(scored_df) -> dict:
    """
    Simple precision/recall against the ground-truth is_fraud label -- used
    only for reporting how well the demo dataset was caught, not fed back
    into the model.
    """
    flagged = scored_df["severity"].isin(["Red", "Yellow"])
    truth = scored_df["is_fraud"]
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 3), "recall": round(recall, 3)}
