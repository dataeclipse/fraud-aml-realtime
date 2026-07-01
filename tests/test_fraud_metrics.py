from __future__ import annotations

import numpy as np

from fraud_aml.fraud_model.metrics import cost_optimal_threshold, pr_auc, recall_at_precision


def test_pr_auc_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y, proba) > 0.99


def test_recall_at_precision() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert recall_at_precision(y, proba, precision_target=0.9) == 1.0


def test_cost_threshold_prefers_recall_when_fn_expensive() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.4, 0.45, 0.5, 0.6])
    threshold, cost = cost_optimal_threshold(y, proba, fn_cost=100.0, fp_cost=1.0)
    assert threshold <= 0.5
    assert cost == 0.0
