from __future__ import annotations

import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def illicit_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, y_score: npt.ArrayLike
) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
