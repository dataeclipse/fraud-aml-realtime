from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_aml.config import Settings
from fraud_aml.fraud_model.train import evaluate, fit_lightgbm


def _split(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"f0": rng.random(n), "f1": rng.random(n)})
    y = pd.Series((x["f0"] + rng.normal(0, 0.1, n) > 0.65).astype(int))
    return x, y


@pytest.mark.fraud
def test_lightgbm_deterministic() -> None:
    x, y = _split(400, 0)
    x_train, y_train = x.iloc[:300], y.iloc[:300]
    x_val, y_val = x.iloc[300:], y.iloc[300:]
    first = fit_lightgbm(x_train, y_train, x_val, y_val, seed=7).predict_proba(x_val)[:, 1]
    second = fit_lightgbm(x_train, y_train, x_val, y_val, seed=7).predict_proba(x_val)[:, 1]
    assert np.allclose(first, second)


@pytest.mark.fraud
def test_evaluate_reasonable() -> None:
    x, y = _split(600, 1)
    x_train, y_train = x.iloc[:400], y.iloc[:400]
    x_val, y_val = x.iloc[400:500], y.iloc[400:500]
    x_test, y_test = x.iloc[500:], y.iloc[500:]
    model = fit_lightgbm(x_train, y_train, x_val, y_val, seed=1)
    metrics = evaluate(model, x_test, y_test, Settings(_env_file=None))
    assert metrics["pr_auc"] > 0.6
    assert metrics["roc_auc"] > 0.7
