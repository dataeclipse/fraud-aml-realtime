from __future__ import annotations

import pandas as pd

from fraud_aml.graph_aml.data import build_graph


def test_build_graph() -> None:
    feats = pd.DataFrame([[10, 1, 0.5, 0.6], [20, 2, 0.1, 0.2], [30, 40, 0.9, 0.8]])
    classes = pd.DataFrame({"txId": [10, 20, 30], "class": ["1", "2", "unknown"]})
    edges = pd.DataFrame({"a": [10, 20], "b": [20, 30]})

    gd = build_graph(feats, classes, edges)
    assert gd.x.shape == (3, 2)
    assert list(gd.y) == [1, 0, -1]
    assert list(gd.timestep) == [1, 2, 40]
    assert gd.edge_index.shape == (2, 2)
    assert gd.edge_index[0, 0] == 0
    assert gd.edge_index[1, 0] == 1
