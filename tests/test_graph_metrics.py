from __future__ import annotations

import numpy as np

from fraud_aml.graph_aml.metrics import illicit_metrics


def test_illicit_metrics_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8])
    m = illicit_metrics(y, pred, score)
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["pr_auc"] > 0.99


def test_illicit_metrics_misses() -> None:
    y = np.array([0, 1, 1, 1])
    pred = np.array([0, 0, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9])
    m = illicit_metrics(y, pred, score)
    assert m["recall"] < 0.5
