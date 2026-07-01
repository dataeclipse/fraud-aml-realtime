from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[3] / "feast_repo"
_FEATURES = [
    "card_window:win_count",
    "card_window:win_sum",
    "card_window:win_velocity",
    "card_window:win_delta",
]


def get_store(repo_path: Path = _REPO) -> Any:
    from feast import FeatureStore

    return FeatureStore(repo_path=str(repo_path))


def push_features(store: Any, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(
        [
            {
                "card1": str(row["card1"]),
                "win_count": row["win_count"],
                "win_sum": row["win_sum"],
                "win_velocity": row["win_velocity"],
                "win_delta": row["win_delta"],
                "event_timestamp": pd.Timestamp(time.time(), unit="s"),
            }
            for row in rows
        ]
    )
    store.push("card_window_push", frame)


def read_online(store: Any, card1: str) -> dict[str, Any]:
    result = store.get_online_features(features=_FEATURES, entity_rows=[{"card1": str(card1)}])
    values: dict[str, Any] = result.to_dict()
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read online window features for a card from Redis."
    )
    parser.add_argument("--card", required=True)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    store = get_store()
    print(read_online(store, args.card))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
