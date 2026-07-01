from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_aml.config import get_settings
from fraud_aml.logging_config import configure_logging, get_logger

DATASET = "ellipticco/elliptic-data-set"
N_LOCAL = 94


@dataclass(frozen=True)
class GraphData:
    x: np.ndarray
    y: np.ndarray
    timestep: np.ndarray
    edge_index: np.ndarray
    node_ids: np.ndarray
    n_local: int


def _api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _human(size: float) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def manifest(api: Any | None = None) -> list[tuple[str, str]]:
    api = api or _api()
    listing = api.dataset_list_files(DATASET)
    items = getattr(listing, "files", listing)
    out: list[tuple[str, str]] = []
    for item in items:
        total_bytes = getattr(item, "totalBytes", None)
        if isinstance(total_bytes, int) and total_bytes > 0:
            size = _human(total_bytes)
        else:
            size = str(getattr(item, "size", "?"))
        out.append((str(item.name), size))
    return out


def download(raw_dir: Path, *, yes: bool, api: Any | None = None) -> None:
    if not yes:
        return
    api = api or _api()
    raw_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET, path=str(raw_dir), unzip=True)


def _find(raw_dir: Path, name: str) -> Path:
    matches = list(raw_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {raw_dir}")
    return matches[0]


def build_graph(feats: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame) -> GraphData:
    node_ids = feats.iloc[:, 0].to_numpy()
    timestep = feats.iloc[:, 1].to_numpy(dtype=int)
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)

    class_map = dict(zip(classes.iloc[:, 0], classes.iloc[:, 1].astype(str), strict=False))
    y = np.array(
        [
            1 if class_map.get(nid) == "1" else 0 if class_map.get(nid) == "2" else -1
            for nid in node_ids
        ],
        dtype=np.int64,
    )

    id_to_idx = {nid: index for index, nid in enumerate(node_ids)}
    src = edges.iloc[:, 0].map(id_to_idx).to_numpy()
    dst = edges.iloc[:, 1].map(id_to_idx).to_numpy()
    valid = ~(pd.isna(src) | pd.isna(dst))
    edge_index = np.vstack([src[valid].astype(np.int64), dst[valid].astype(np.int64)])
    return GraphData(x, y, timestep, edge_index, node_ids, N_LOCAL)


def load_elliptic(raw_dir: Path | None = None) -> GraphData:
    raw = raw_dir or (get_settings().data_dir / "raw")
    feats = pd.read_csv(_find(raw, "elliptic_txs_features.csv"), header=None)
    classes = pd.read_csv(_find(raw, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(_find(raw, "elliptic_txs_edgelist.csv"))
    return build_graph(feats, classes, edges)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the Elliptic dataset (nodes/edges/labels)."
    )
    parser.add_argument(
        "--yes", action="store_true", help="actually download (otherwise manifest only)"
    )
    parser.add_argument("--raw-dir", default=None)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    log = get_logger("graph-ingest")
    settings = get_settings()
    raw_dir = Path(args.raw_dir) if args.raw_dir else settings.data_dir / "raw"

    api = _api()
    print(f"Elliptic ({DATASET}) - files (to {raw_dir}):")
    for name, size in manifest(api):
        print(f"  {name:44s} {size:>12s}")
    print()

    if not args.yes:
        print("Dry-run. Add --yes to download.")
        return 0

    download(raw_dir, yes=True, api=api)
    log.info("graph_ingest_done", raw_dir=str(raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
