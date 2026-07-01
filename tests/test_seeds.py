from __future__ import annotations

import numpy as np

from fraud_aml.seeds import set_seeds


def test_seeds_deterministic() -> None:
    set_seeds(123)
    first = np.random.rand(5)
    set_seeds(123)
    second = np.random.rand(5)
    assert np.allclose(first, second)
