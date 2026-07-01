from __future__ import annotations

import pandas as pd

TARGET = "isFraud"
ID = "TransactionID"
TIME = "TransactionDT"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seconds = out[TIME].to_numpy(dtype=float)
    out["dt_hour"] = ((seconds // 3600) % 24).astype("float32")
    out["dt_day"] = (seconds // 86400).astype("float32")
    return out


def add_card_delta(df: pd.DataFrame, *, card_col: str = "card1") -> pd.DataFrame:
    out = df.copy()
    ordered = out.sort_values(TIME, kind="stable")
    delta = ordered.groupby(card_col)[TIME].diff()
    out["card_dt_delta"] = delta.reindex(out.index).astype("float32")
    return out


def add_missing_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_missing"] = df.isna().sum(axis=1).astype("float32")
    return out


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(add_card_delta(add_missing_count(df)))
