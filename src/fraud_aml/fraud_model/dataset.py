from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fraud_aml.config import Settings
from fraud_aml.data.loaders import load_merged
from fraud_aml.data.split import time_based_split
from fraud_aml.features.build import ID, TARGET, TIME, add_all
from fraud_aml.features.encoding import FrequencyEncoder, TargetEncoder

_FREQ_COLS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
]
_TE_COLS = ["card1", "addr1", "P_emaildomain"]


@dataclass(frozen=True)
class FraudDataset:
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any
    y_test: Any
    feature_names: list[str]
    train_max_dt: float
    val_min_dt: float
    val_max_dt: float
    test_min_dt: float


def _numeric_base(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    drop = [c for c in (ID, TARGET, TIME) if c in numeric.columns]
    return numeric.drop(columns=drop)


def build_fraud_dataset(settings: Settings, *, use_cache: bool = True) -> FraudDataset:
    df = add_all(load_merged(use_cache=use_cache))
    split = time_based_split(
        df, time_col=TIME, train_frac=settings.train_frac, val_frac=settings.val_frac
    )

    y = df[TARGET].astype(int)
    freq_cols = [c for c in _FREQ_COLS if c in df.columns]
    te_cols = [c for c in _TE_COLS if c in df.columns]

    freq = FrequencyEncoder(freq_cols).fit(df.loc[split.train_idx])
    te = TargetEncoder(te_cols).fit(df.loc[split.train_idx], y.loc[split.train_idx])

    features = pd.concat([_numeric_base(df), freq.transform(df), te.transform(df)], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    feature_names = [str(c) for c in features.columns]

    dt = df[TIME]
    return FraudDataset(
        X_train=features.loc[split.train_idx],
        y_train=y.loc[split.train_idx],
        X_val=features.loc[split.val_idx],
        y_val=y.loc[split.val_idx],
        X_test=features.loc[split.test_idx],
        y_test=y.loc[split.test_idx],
        feature_names=feature_names,
        train_max_dt=float(dt.loc[split.train_idx].max()),
        val_min_dt=float(dt.loc[split.val_idx].min()),
        val_max_dt=float(dt.loc[split.val_idx].max()),
        test_min_dt=float(dt.loc[split.test_idx].min()),
    )
