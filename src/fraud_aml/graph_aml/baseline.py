from __future__ import annotations

import numpy as np


def train_baseline(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        scale_pos_weight=negatives / max(positives, 1),
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba
