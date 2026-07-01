from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    return float(roc_auc_score(y_true, y_score))


def recall_at_precision(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, *, precision_target: float
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision[:-1] >= precision_target
    return float(recall[:-1][mask].max()) if mask.any() else 0.0


def precision_at_recall(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, *, recall_target: float
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall[:-1] >= recall_target
    return float(precision[:-1][mask].max()) if mask.any() else 0.0


def cost_optimal_threshold(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, *, fn_cost: float, fp_cost: float
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    scores_sorted = scores[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    positives = int(y.sum())
    fn = positives - tp
    cost = fn * fn_cost + fp * fp_cost
    base_cost = positives * fn_cost
    best_k = int(np.argmin(cost))
    if base_cost <= float(cost[best_k]):
        return 1.0, float(base_cost)
    return float(scores_sorted[best_k]), float(cost[best_k])
