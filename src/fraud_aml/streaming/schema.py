from __future__ import annotations

import json
from typing import Any

FIELDS = ("TransactionID", "TransactionDT", "card1", "TransactionAmt", "publish_ts")


def serialize(event: dict[str, Any]) -> bytes:
    return json.dumps(event, separators=(",", ":")).encode()


def deserialize(raw: bytes | str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(raw)
    return data
