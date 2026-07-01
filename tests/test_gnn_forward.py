from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.graph


def test_sage_forward_deterministic() -> None:
    from fraud_aml.graph_aml.gnn import run_forward

    x = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.1, 0.0]], dtype=np.float32
    )
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    first = run_forward(x, edge_index, kind="sage", hidden=8, seed=1)
    second = run_forward(x, edge_index, kind="sage", hidden=8, seed=1)
    assert first.shape == (4, 2)
    assert np.allclose(first, second)
