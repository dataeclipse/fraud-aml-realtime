from __future__ import annotations

import pandas as pd

from fraud_aml.features.encoding import FrequencyEncoder, TargetEncoder


def test_target_encoder_fit_train_only() -> None:
    train = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    target = pd.Series([1, 1, 0, 0])
    encoder = TargetEncoder(["c"], smoothing=0.0).fit(train, target)
    val = pd.DataFrame({"c": ["a", "b", "z"]})
    out = encoder.transform(val)["c_te"]
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 0.0
    assert abs(out.iloc[2] - 0.5) < 1e-6


def test_frequency_encoder() -> None:
    train = pd.DataFrame({"c": ["a", "a", "b"]})
    encoder = FrequencyEncoder(["c"]).fit(train)
    out = encoder.transform(pd.DataFrame({"c": ["a", "b", "z"]}))["c_freq"]
    assert list(out) == [2.0, 1.0, 0.0]
