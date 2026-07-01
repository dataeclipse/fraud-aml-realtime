from __future__ import annotations

from typing import Any

import pandas as pd

from fraud_aml.streaming.features import add_stream_features
from fraud_aml.streaming.processor import run_events

W = 50
_EVENTS = [
    ("A", 100, 10.0),
    ("B", 100, 5.0),
    ("A", 100, 20.0),
    ("A", 150, 1.0),
    ("A", 151, 1.0),
    ("A", 100000, 7.0),
]
_FIELDS = ("win_count", "win_sum", "win_velocity", "win_delta")


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": list(range(len(_EVENTS))),
            "card1": [c for c, _, _ in _EVENTS],
            "TransactionDT": [d for _, d, _ in _EVENTS],
            "TransactionAmt": [a for _, _, a in _EVENTS],
        }
    )


def _events() -> list[dict[str, Any]]:
    return [
        {"TransactionID": i, "card1": c, "TransactionDT": d, "TransactionAmt": a, "publish_ts": 0.0}
        for i, (c, d, a) in enumerate(_EVENTS)
    ]


def test_online_offline_equivalence() -> None:
    offline = add_stream_features(_df(), window_seconds=W)
    online = run_events(_events(), window_seconds=W)
    for i in range(len(_EVENTS)):
        for field in _FIELDS:
            assert offline.iloc[i][field] == online[i][field], (i, field)


def test_window_boundary_inclusive_then_exclusive() -> None:
    online = run_events(_events(), window_seconds=W)
    assert online[3]["win_count"] == 3.0
    assert online[4]["win_count"] == 2.0


def test_tie_break_same_dt() -> None:
    online = run_events(_events(), window_seconds=W)
    assert online[2]["win_count"] == 2.0
    assert online[2]["win_sum"] == 30.0
    assert online[2]["win_delta"] == 0.0


def test_unseen_entity_and_sparse_gap() -> None:
    online = run_events(_events(), window_seconds=W)
    assert online[0]["win_delta"] == -1.0
    assert online[1]["win_delta"] == -1.0
    assert online[5]["win_count"] == 1.0
    assert online[5]["win_delta"] == float(100000 - 151)
