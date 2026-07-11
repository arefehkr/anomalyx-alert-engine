"""
Turns raw transactions into the 4 features the detector uses. Each one is
deliberately simple and easy to explain out loud:

  spend_ratio     -- this transaction's amount / the account's rolling
                      30-day average spend (1.0 = right on baseline)
  txn_count_1h    -- how many transactions this account made in the last hour
  is_night        -- was this between midnight and 6am
  merchant_mismatch -- is this merchant category one the account doesn't
                        normally use
"""
from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = "30D"


def load_data(data_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = pd.read_csv(data_dir / "accounts.csv")
    accounts["typical_categories"] = accounts["typical_categories"].apply(lambda s: s.split(","))
    txns = pd.read_csv(data_dir / "transactions.csv")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    return accounts, txns


def build_features(accounts: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    df = txns.merge(accounts, on="account_id", how="left")
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    parts = []
    for _, group in df.groupby("account_id", sort=False):
        g = group.copy()
        s = g.set_index("timestamp")["amount"]

        # Rolling baseline: only look at PRIOR transactions (closed="left"),
        # so a transaction never gets compared against itself.
        rolling_mean = s.rolling(ROLLING_WINDOW, closed="left").mean().to_numpy()
        g["rolling_avg"] = rolling_mean
        g["rolling_avg"] = g["rolling_avg"].fillna(g["avg_amount"])  # cold start
        g["spend_ratio"] = g["amount"] / g["rolling_avg"].clip(lower=1.0)

        g["txn_count_1h"] = s.rolling("1h", closed="both").count().to_numpy()

        parts.append(g)

    df = pd.concat(parts).sort_index()
    df["is_night"] = df["timestamp"].dt.hour.between(0, 5)
    df["merchant_mismatch"] = df.apply(
        lambda r: int(r["merchant_category"] not in r["typical_categories"]), axis=1
    )
    return df[
        ["txn_id", "account_id", "account_name", "timestamp", "amount", "merchant_category",
         "spend_ratio", "txn_count_1h", "is_night", "merchant_mismatch", "is_fraud"]
    ]
