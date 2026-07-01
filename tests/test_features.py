from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_aml.features.build import add_card_delta, add_time_features


def test_time_features() -> None:
    df = pd.DataFrame({"TransactionDT": [0, 3600, 93600], "card1": [1, 1, 1]})
    out = add_time_features(df)
    assert list(out["dt_hour"]) == [0.0, 1.0, 2.0]
    assert list(out["dt_day"]) == [0.0, 0.0, 1.0]


def test_card_delta() -> None:
    df = pd.DataFrame({"TransactionDT": [0, 10, 25], "card1": [1, 1, 1]})
    deltas = list(add_card_delta(df)["card_dt_delta"])
    assert np.isnan(deltas[0])
    assert deltas[1] == 10.0
    assert deltas[2] == 15.0
