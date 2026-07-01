from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_idx: Any
    val_idx: Any
    test_idx: Any


def time_based_split(
    df: pd.DataFrame,
    *,
    time_col: str = "TransactionDT",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> TimeSplit:
    order = np.argsort(df[time_col].to_numpy(), kind="stable")
    ordered = df.index.to_numpy()[order]
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return TimeSplit(
        train_idx=pd.Index(ordered[:n_train]),
        val_idx=pd.Index(ordered[n_train : n_train + n_val]),
        test_idx=pd.Index(ordered[n_train + n_val :]),
    )
