from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_aml.config import get_settings

_MERGED = "ieee_merged.parquet"


def _raw_dir() -> Path:
    return get_settings().data_dir / "raw"


def _interim_dir() -> Path:
    interim = get_settings().data_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    return interim


def load_merged(*, use_cache: bool = True) -> pd.DataFrame:
    cache = _interim_dir() / _MERGED
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    raw = _raw_dir()
    transaction = pd.read_csv(raw / "train_transaction.csv")
    identity = pd.read_csv(raw / "train_identity.csv")
    merged = transaction.merge(identity, on="TransactionID", how="left")
    merged.to_parquet(cache, index=False)
    return merged
