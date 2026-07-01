from __future__ import annotations

import numpy as np


def temporal_split(
    timestep: np.ndarray, y: np.ndarray, *, cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    labeled = y >= 0
    train_idx = np.where(labeled & (timestep <= cutoff))[0]
    test_idx = np.where(labeled & (timestep > cutoff))[0]
    return train_idx, test_idx
