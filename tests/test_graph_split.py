from __future__ import annotations

import numpy as np

from fraud_aml.graph_aml.split import temporal_split


def test_temporal_no_leak() -> None:
    timestep = np.array([1, 2, 3, 40, 41, 42])
    y = np.array([1, 0, 1, 0, 1, -1])
    train, test = temporal_split(timestep, y, cutoff=34)
    assert timestep[train].max() < timestep[test].min()
    assert 5 not in train and 5 not in test
    assert set(train).isdisjoint(test)


def test_determinism() -> None:
    timestep = np.array([1, 2, 40])
    y = np.array([1, 0, 1])
    first = temporal_split(timestep, y, cutoff=34)
    second = temporal_split(timestep, y, cutoff=34)
    assert list(first[0]) == list(second[0])
    assert list(first[1]) == list(second[1])
