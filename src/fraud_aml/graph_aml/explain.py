from __future__ import annotations

from typing import Any

import numpy as np


def suspicious_subgraph(
    edge_index: np.ndarray, node: int, proba: np.ndarray, *, hops: int = 1, top: int = 20
) -> dict[str, Any]:
    import torch
    from torch_geometric.utils import k_hop_subgraph

    ei = torch.tensor(edge_index, dtype=torch.long)
    nodes, _, _, _ = k_hop_subgraph(node, hops, ei, relabel_nodes=False)
    neighbors = [int(n) for n in nodes.tolist() if int(n) != node]
    neighbors.sort(key=lambda n: float(proba[n]), reverse=True)
    top_neighbors = neighbors[:top]
    return {
        "node": int(node),
        "score": round(float(proba[node]), 4),
        "n_neighbors": len(neighbors),
        "top_neighbors": top_neighbors,
        "top_neighbor_scores": [round(float(proba[n]), 4) for n in top_neighbors],
    }
