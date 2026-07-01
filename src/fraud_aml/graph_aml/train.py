from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from fraud_aml.config import Settings, get_settings
from fraud_aml.graph_aml.baseline import train_baseline
from fraud_aml.graph_aml.data import GraphData, load_elliptic
from fraud_aml.graph_aml.explain import suspicious_subgraph
from fraud_aml.graph_aml.gnn import device_name, train_gnn
from fraud_aml.graph_aml.metrics import illicit_metrics
from fraud_aml.graph_aml.split import temporal_split
from fraud_aml.logging_config import configure_logging, get_logger


def _cols(gd: GraphData, local: bool) -> list[int]:
    return list(range(gd.n_local)) if local else list(range(gd.x.shape[1]))


def _tabular(
    gd: GraphData, train_idx: Any, test_idx: Any, *, local: bool, seed: int
) -> dict[str, float]:
    x = gd.x[:, _cols(gd, local)]
    pred, proba = train_baseline(x[train_idx], gd.y[train_idx], x[test_idx], seed=seed)
    return illicit_metrics(gd.y[test_idx], pred, proba)


def _gnn(
    gd: GraphData, train_idx: Any, test_idx: Any, *, local: bool, kind: str, settings: Settings
) -> tuple[dict[str, float], np.ndarray]:
    x = gd.x[:, _cols(gd, local)]
    _, pred, proba = train_gnn(
        x,
        gd.edge_index,
        gd.y,
        train_idx,
        kind=kind,
        hidden=settings.gnn_hidden,
        epochs=settings.gnn_epochs,
        lr=settings.gnn_lr,
        seed=settings.random_seed,
    )
    return illicit_metrics(gd.y[test_idx], pred[test_idx], proba[test_idx]), proba


def _row(name: str, m: dict[str, float]) -> str:
    return (
        f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['pr_auc']:.3f} |"
    )


def _write_docs(
    path: Path,
    gd: GraphData,
    train_idx: Any,
    test_idx: Any,
    full: dict[str, dict[str, float]],
    local: dict[str, dict[str, float]],
    subgraph: dict[str, Any],
    device: str,
    cutoff: int,
) -> None:
    labeled = int((gd.y >= 0).sum())
    illicit = int((gd.y == 1).sum())
    top_scores = subgraph["top_neighbor_scores"][:5]
    full_best_gnn = max(full["sage"]["f1"], full["gat"]["f1"])
    full_verdict = "beats" if full["tabular"]["f1"] >= full_best_gnn else "is beaten by"
    local_verdict = (
        "still beats" if local["tabular"]["f1"] >= local["sage"]["f1"] else "is beaten by"
    )
    lines = [
        "# Graph AML on Elliptic (Phase 4)",
        "",
        f"Elliptic: {gd.x.shape[0]} nodes, {gd.x.shape[1]} features, "
        f"{gd.edge_index.shape[1]} edges, timesteps "
        f"{int(gd.timestep.min())}-{int(gd.timestep.max())}.",
        f"Labeled {labeled} ({illicit} illicit = {illicit / labeled:.1%}); the rest unknown and "
        f"masked. Trained on {device}.",
        "",
        "## Temporal split (no leak)",
        f"Train timestep <= {cutoff}, test > {cutoff} (Weber et al. split). "
        f"Train nodes {len(train_idx)}, test nodes {len(test_idx)}.",
        f"train.max(ts)={int(gd.timestep[train_idx].max())} < "
        f"test.min(ts)={int(gd.timestep[test_idx].min())}. Asserted in tests/test_graph_split.py.",
        "Unknown nodes still pass messages but are not in the loss.",
        "",
        "## Full features (165): tabular vs GNN",
        "| Model | illicit F1 | precision | recall | PR-AUC |",
        "|---|---|---|---|---|",
        _row("LightGBM (tabular)", full["tabular"]),
        _row("GraphSAGE", full["sage"]),
        _row("GAT", full["gat"]),
        "",
        "## Local features (94, no aggregates): tabular vs GNN",
        "| Model | illicit F1 | precision | recall | PR-AUC |",
        "|---|---|---|---|---|",
        _row("LightGBM (tabular)", local["tabular"]),
        _row("GraphSAGE", local["sage"]),
        "",
        "## Main conclusion (the point of this project)",
        f"On the full feature set the tabular model {full_verdict} the GNN "
        f"(F1 {full['tabular']['f1']:.3f} vs {full_best_gnn:.3f}): ~71 features are already",
        "neighborhood aggregates, so the graph structure is baked into the node features",
        "(Weber et al. 2019: RandomForest illicit-F1 0.79 vs GCN 0.42).",
        "",
        f"On local features (aggregates removed) the tabular model {local_verdict} the GNN "
        f"(F1 {local['tabular']['f1']:.3f} vs {local['sage']['f1']:.3f}). The GNN trades precision",
        "for recall (high recall, low precision) - it flags many nodes and catches most illicit,",
        "but noisily.",
        "",
        "Takeaway (methodological, not a failure): a GNN is not a default win. Validate it against",
        "a strong gradient-boosted tabular baseline. On Elliptic the signal a vanilla 2-layer",
        "GraphSAGE/GAT extracts is already captured by hand-crafted features and boosting; beating",
        "the tabular model needs a stronger graph model (temporal GNNs like EvolveGCN, tuned",
        "nets). Knowing when NOT to reach for a GNN is the result.",
        "",
        "## Suspicious subgraph (compliance report)",
        f"Top illicit-scored test node {subgraph['node']} (score {subgraph['score']}): "
        f"{subgraph['n_neighbors']} neighbors, top neighbor scores {top_scores}.",
        "GNNExplainer or GAT attention weights can refine which edges drive the flag.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Elliptic graph AML: tabular baseline vs GNN.")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    log = get_logger("graph")
    settings = get_settings()

    gd = load_elliptic()
    train_idx, test_idx = temporal_split(gd.timestep, gd.y, cutoff=settings.elliptic_cutoff)
    device = device_name()

    tab_full = _tabular(gd, train_idx, test_idx, local=False, seed=settings.random_seed)
    sage_full, sage_proba = _gnn(
        gd, train_idx, test_idx, local=False, kind="sage", settings=settings
    )
    gat_full, _ = _gnn(gd, train_idx, test_idx, local=False, kind="gat", settings=settings)
    tab_local = _tabular(gd, train_idx, test_idx, local=True, seed=settings.random_seed)
    sage_local, _ = _gnn(gd, train_idx, test_idx, local=True, kind="sage", settings=settings)

    top_node = int(test_idx[int(np.argmax(sage_proba[test_idx]))])
    subgraph = suspicious_subgraph(gd.edge_index, top_node, sage_proba)

    full = {"tabular": tab_full, "sage": sage_full, "gat": gat_full}
    local = {"tabular": tab_local, "sage": sage_local}
    _write_docs(
        Path(args.docs_dir) / "graph_aml.md",
        gd,
        train_idx,
        test_idx,
        full,
        local,
        subgraph,
        device,
        settings.elliptic_cutoff,
    )

    print(f"DEVICE {device}")
    print(f"FULL  tab={tab_full['f1']:.3f} sage={sage_full['f1']:.3f} gat={gat_full['f1']:.3f}")
    print(f"LOCAL tab={tab_local['f1']:.3f} sage={sage_local['f1']:.3f}")
    print(
        f"SPLIT train.max_ts={int(gd.timestep[train_idx].max())} "
        f"test.min_ts={int(gd.timestep[test_idx].min())}"
    )
    log.info(
        "graph_done",
        full_tabular_f1=round(tab_full["f1"], 3),
        full_sage_f1=round(sage_full["f1"], 3),
        local_tabular_f1=round(tab_local["f1"], 3),
        local_sage_f1=round(sage_local["f1"], 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
