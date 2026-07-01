from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_aml.data.split import time_based_split


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"TransactionDT": np.arange(n) * 10, "x": rng.random(n)})
    return df.sample(frac=1.0, random_state=1)


def test_no_time_leak() -> None:
    df = _frame()
    split = time_based_split(df, train_frac=0.6, val_frac=0.2)
    train = df.loc[split.train_idx, "TransactionDT"]
    val = df.loc[split.val_idx, "TransactionDT"]
    test = df.loc[split.test_idx, "TransactionDT"]
    assert train.max() < val.min()
    assert val.max() < test.min()
    assert train.max() < test.min()


def test_deterministic_and_disjoint() -> None:
    df = _frame()
    first = time_based_split(df)
    second = time_based_split(df)
    assert list(first.train_idx) == list(second.train_idx)
    all_idx = set(first.train_idx) | set(first.val_idx) | set(first.test_idx)
    assert all_idx == set(df.index)
    assert set(first.train_idx).isdisjoint(first.val_idx)
    assert set(first.val_idx).isdisjoint(first.test_idx)
