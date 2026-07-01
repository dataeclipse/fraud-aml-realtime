from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from fraud_aml.features.build import TIME

AMOUNT = "TransactionAmt"
ENTITY = "card1"
NO_PREVIOUS = -1.0
_KEYS = ("win_count", "win_sum", "win_velocity", "win_delta")


@dataclass
class WindowState:
    events: deque[tuple[int, float]] = field(default_factory=deque)
    prev_dt: int | None = None


def update_window(
    state: WindowState, dt: int, amount: float, *, window_seconds: int
) -> dict[str, float]:
    while state.events and dt - state.events[0][0] > window_seconds:
        state.events.popleft()
    state.events.append((dt, amount))
    count = float(len(state.events))
    amount_sum = float(sum(value for _, value in state.events))
    velocity = amount_sum / float(window_seconds)
    delta = NO_PREVIOUS if state.prev_dt is None else float(dt - state.prev_dt)
    state.prev_dt = dt
    return {"win_count": count, "win_sum": amount_sum, "win_velocity": velocity, "win_delta": delta}


def add_stream_features(
    df: pd.DataFrame, *, window_seconds: int, entity_col: str = ENTITY
) -> pd.DataFrame:
    ordered = df.sort_values(TIME, kind="stable")
    states: dict[str, WindowState] = {}
    rows: dict[str, list[float]] = {key: [] for key in _KEYS}
    index: list[object] = []
    for row in ordered.itertuples(index=True):
        key = str(getattr(row, entity_col))
        state = states.setdefault(key, WindowState())
        feats = update_window(
            state,
            int(getattr(row, TIME)),
            float(getattr(row, AMOUNT)),
            window_seconds=window_seconds,
        )
        index.append(row.Index)
        for name in _KEYS:
            rows[name].append(feats[name])
    return pd.DataFrame(rows, index=index).reindex(df.index)
