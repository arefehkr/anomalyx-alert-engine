#!/usr/bin/env python3
"""
run_pipeline.py -- the whole thing, start to finish.

    python run_pipeline.py                # run on the full dataset
    python run_pipeline.py --max-alerts 3  # cap how many alerts reach the agent

Steps:
  1. Load (or generate) the synthetic transaction data.
  2. Engineer 4 features per transaction.
  3. Score every transaction with an Isolation Forest -> Red/Yellow/Green.
  4. For flagged transactions: dedupe bursts, run the two-layer agent
     (Python rules engine + local Llama/template), print + log each alert.
  5. Report precision/recall against the ground-truth labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.agent import build_context, generate_alert
from src.alerter import evaluate, format_alert, log_alert, select_alert_candidates
from src.detector import score_transactions
from src.features import build_features, load_data
from src.generate_data import main as generate_data

DATA_DIR = Path(__file__).resolve().parent / "data"
ALERT_LOG_PATH = DATA_DIR / "alerts_log.json"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-alerts", type=int, default=None)
    args = parser.parse_args()

    if not (DATA_DIR / "transactions.csv").exists():
        print("No data found -- generating synthetic dataset ...")
        generate_data()

    print("Loading data ...")
    accounts, txns = load_data(DATA_DIR)
    print(f"  {len(txns)} transactions across {len(accounts)} accounts")

    print("\nEngineering features ...")
    features = build_features(accounts, txns)

    print("\nScoring with Isolation Forest ...")
    scored = score_transactions(features)
    counts = scored["severity"].value_counts()
    print(f"  Red: {counts.get('Red', 0)}  Yellow: {counts.get('Yellow', 0)}  Green: {counts.get('Green', 0)}")

    print("\nRunning the agent on flagged transactions ...")
    candidates = select_alert_candidates(scored[scored["severity"] != "Green"])
    if args.max_alerts:
        rank = {"Red": 2, "Yellow": 1}
        candidates = candidates.assign(_r=candidates["severity"].map(rank))
        candidates = candidates.sort_values(["_r", "anomaly_score"], ascending=False).drop(columns="_r")
        # Pick the top-N most severe candidates, then process them back in
        # chronological order so the account-cooldown check behaves correctly.
        candidates = candidates.head(args.max_alerts).sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(candidates)} alert candidate(s) after de-duplicating bursts.")

    ALERT_LOG_PATH.write_text("[]")
    dispatched_history: dict[str, list] = {}
    for _, row in candidates.iterrows():
        ctx = build_context(row, dispatched_history)
        alert = generate_alert(ctx)
        print(format_alert(alert))
        log_alert(alert, ALERT_LOG_PATH)
        if ctx.dispatch_safe:
            dispatched_history.setdefault(ctx.account_id, []).append(row["timestamp"])

    print(f"\n{len(candidates)} alert(s) written to {ALERT_LOG_PATH}")

    print("\nEvaluation against ground-truth labels (demo dataset only):")
    for k, v in evaluate(scored).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
