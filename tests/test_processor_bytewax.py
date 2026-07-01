from __future__ import annotations

from typing import Any

import pytest

from fraud_aml.streaming.processor import make_mapper, run_events

pytestmark = pytest.mark.stream

_RAW = [("A", 100, 10.0), ("A", 120, 5.0), ("B", 130, 3.0), ("A", 200, 1.0)]
_EVENTS = [
    {"TransactionID": i, "card1": c, "TransactionDT": d, "TransactionAmt": a, "publish_ts": 0.0}
    for i, (c, d, a) in enumerate(_RAW)
]


def test_bytewax_dataflow_matches_reducer() -> None:
    import bytewax.operators as op
    from bytewax.dataflow import Dataflow
    from bytewax.testing import TestingSource, run_main

    collected: list[dict[str, Any]] = []
    flow = Dataflow("test")
    stream = op.input("in", flow, TestingSource(_EVENTS))
    keyed = op.key_on("key", stream, lambda event: str(event["card1"]))
    featured = op.stateful_map("win", keyed, make_mapper(50))
    op.inspect("out", featured, lambda _step, item: collected.append(item[1]))
    run_main(flow)

    expected = {(r["card1"], r["TransactionID"]): r for r in run_events(_EVENTS, window_seconds=50)}
    got = {(r["card1"], r["TransactionID"]): r for r in collected}
    assert got.keys() == expected.keys()
    for key in expected:
        for field in ("win_count", "win_sum", "win_velocity", "win_delta"):
            assert got[key][field] == expected[key][field]
