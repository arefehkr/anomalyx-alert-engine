"""
Generates a small synthetic dataset of bank transactions with a handful of
injected anomalies. Run directly: `python src/generate_data.py`

Design note: `is_fraud` is a ground-truth label used ONLY to (a) inject
realistic anomalies and (b) evaluate the pipeline afterwards. It is never
used as a feature or read by the detector or the agent -- the pipeline has
no access to the answer key.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RNG = np.random.default_rng(42)

ACCOUNTS = [
    {"account_id": "ACC001", "account_name": "Personal Chequing", "avg_amount": 45.0,
     "typical_categories": ["groceries", "dining", "utilities"]},
    {"account_id": "ACC002", "account_name": "Business Card", "avg_amount": 220.0,
     "typical_categories": ["office_supplies", "travel", "software"]},
    {"account_id": "ACC003", "account_name": "Premium Card", "avg_amount": 350.0,
     "typical_categories": ["travel", "dining", "electronics"]},
    {"account_id": "ACC004", "account_name": "Student Card", "avg_amount": 25.0,
     "typical_categories": ["groceries", "dining", "entertainment"]},
]

OTHER_CATEGORIES = ["gift_cards", "crypto_exchange", "cash_advance", "wire_transfer", "atm_withdrawal"]


def _normal_transactions(n_days: int = 90) -> list[dict]:
    rows = []
    start = pd.Timestamp("2025-01-01")
    for acc in ACCOUNTS:
        for day_offset in range(n_days):
            if RNG.random() > 0.55:  # not every account transacts every day
                continue
            n_today = RNG.poisson(1.3) + 1
            for _ in range(n_today):
                ts = start + pd.Timedelta(days=day_offset, hours=int(RNG.integers(7, 22)), minutes=int(RNG.integers(0, 60)))
                amount = round(max(2.0, RNG.lognormal(np.log(acc["avg_amount"]), 0.5)), 2)
                category = RNG.choice(acc["typical_categories"])
                rows.append({
                    "txn_id": f"TXN{len(rows) + 1:05d}", "account_id": acc["account_id"],
                    "timestamp": ts, "amount": amount, "merchant_category": category,
                    "is_fraud": False,
                })
    return rows


def _inject_anomalies(rows: list[dict], n_each: int = 6) -> list[dict]:
    start = pd.Timestamp("2025-02-01")

    for _ in range(n_each):  # large amount spike
        acc = ACCOUNTS[int(RNG.integers(0, len(ACCOUNTS)))]
        ts = start + pd.Timedelta(days=int(RNG.integers(0, 60)), hours=int(RNG.integers(9, 20)))
        rows.append({
            "txn_id": f"TXN{len(rows) + 1:05d}", "account_id": acc["account_id"], "timestamp": ts,
            "amount": round(acc["avg_amount"] * RNG.uniform(6, 12), 2),
            "merchant_category": RNG.choice(acc["typical_categories"]), "is_fraud": True,
        })

    for _ in range(n_each):  # off-hours transaction
        acc = ACCOUNTS[int(RNG.integers(0, len(ACCOUNTS)))]
        ts = start + pd.Timedelta(days=int(RNG.integers(0, 60)), hours=int(RNG.integers(1, 5)))
        rows.append({
            "txn_id": f"TXN{len(rows) + 1:05d}", "account_id": acc["account_id"], "timestamp": ts,
            "amount": round(acc["avg_amount"] * RNG.uniform(1.5, 3), 2),
            "merchant_category": "atm_withdrawal", "is_fraud": True,
        })

    for _ in range(n_each):  # burst of transactions in a short window
        acc = ACCOUNTS[int(RNG.integers(0, len(ACCOUNTS)))]
        base_ts = start + pd.Timedelta(days=int(RNG.integers(0, 60)), hours=int(RNG.integers(9, 20)))
        for m in range(4):
            rows.append({
                "txn_id": f"TXN{len(rows) + 1:05d}", "account_id": acc["account_id"],
                "timestamp": base_ts + pd.Timedelta(minutes=m * 8),
                "amount": round(acc["avg_amount"] * RNG.uniform(0.8, 1.5), 2),
                "merchant_category": RNG.choice(OTHER_CATEGORIES), "is_fraud": True,
            })

    for _ in range(n_each):  # unfamiliar merchant category
        acc = ACCOUNTS[int(RNG.integers(0, len(ACCOUNTS)))]
        ts = start + pd.Timedelta(days=int(RNG.integers(0, 60)), hours=int(RNG.integers(9, 20)))
        rows.append({
            "txn_id": f"TXN{len(rows) + 1:05d}", "account_id": acc["account_id"], "timestamp": ts,
            "amount": round(acc["avg_amount"] * RNG.uniform(1.2, 2.5), 2),
            "merchant_category": RNG.choice(OTHER_CATEGORIES), "is_fraud": True,
        })

    return rows


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts_df = pd.DataFrame(ACCOUNTS)
    accounts_df["typical_categories"] = accounts_df["typical_categories"].apply(lambda c: ",".join(c))

    rows = _normal_transactions()
    rows = _inject_anomalies(rows)
    txns_df = pd.DataFrame(rows).sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    return accounts_df, txns_df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    accounts_df, txns_df = generate()
    accounts_df.to_csv(DATA_DIR / "accounts.csv", index=False)
    txns_df.to_csv(DATA_DIR / "transactions.csv", index=False)
    print(f"Wrote {len(accounts_df)} accounts and {len(txns_df)} transactions "
          f"({txns_df['is_fraud'].sum()} labelled anomalies) to {DATA_DIR}")


if __name__ == "__main__":
    main()
