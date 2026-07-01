from __future__ import annotations

from typing import Any

import numpy as np


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build(kind: str, in_channels: int, hidden: int) -> Any:
    from torch_geometric.nn import GAT, GraphSAGE

    model_cls = GraphSAGE if kind == "sage" else GAT
    return model_cls(in_channels=in_channels, hidden_channels=hidden, num_layers=2, out_channels=2)


def run_forward(
    x: np.ndarray, edge_index: np.ndarray, *, kind: str = "sage", hidden: int = 8, seed: int = 42
) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    model = _build(kind, x.shape[1], hidden)
    model.train(False)
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long))
    result: np.ndarray = out.numpy()
    return result


def train_gnn(
    x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    kind: str = "sage",
    hidden: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    seed: int = 42,
) -> tuple[Any, np.ndarray, np.ndarray]:
    import torch

    torch.manual_seed(seed)
    device = device_name()
    xt = torch.tensor(x, dtype=torch.float, device=device)
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    tr = torch.tensor(train_idx, dtype=torch.long, device=device)

    model = _build(kind, x.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    positives = int((y[train_idx] == 1).sum())
    negatives = int(len(train_idx) - positives)
    weight = torch.tensor([1.0, negatives / max(positives, 1)], dtype=torch.float, device=device)

    model.train(True)
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(xt, ei)
        loss = torch.nn.functional.cross_entropy(out[tr], yt[tr], weight=weight)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    model.train(False)
    with torch.no_grad():
        proba = torch.softmax(model(xt, ei), dim=1)[:, 1].cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    return model, pred, proba
