from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_aml.streaming.producer import playback_delay, replay


class _Sink:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def produce(self, topic: str, value: bytes) -> None:
        self.messages.append(json.loads(value))

    def flush(self) -> None:
        pass


def test_playback_delay_scales() -> None:
    assert playback_delay(1000, 0, 1000.0) == 1.0
    assert playback_delay(0, 0, 1000.0) == 0.0
    assert playback_delay(500, 1000, 1000.0) == 0.0


def test_replay_orders_by_time(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [30, 10, 20],
            "card1": [100, 100, 200],
            "TransactionAmt": [5.0, 6.0, 7.0],
        }
    )
    parquet = tmp_path / "merged.parquet"
    df.to_parquet(parquet)
    sink = _Sink()
    count = replay(
        parquet,
        topic="t",
        speedup=1e9,
        sink=sink,
        clock=lambda: 0.0,
        sleep=lambda _s: None,
        wall=lambda: 0.0,
    )
    assert count == 3
    assert [m["TransactionDT"] for m in sink.messages] == [10, 20, 30]
