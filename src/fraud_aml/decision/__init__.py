from __future__ import annotations

from typing import Literal

Decision = Literal["allow", "review", "block"]
_ORDER: dict[str, int] = {"allow": 0, "review": 1, "block": 2}


def decide(score: float, *, review_at: float = 0.3, block_at: float = 0.7) -> Decision:
    if score >= block_at:
        return "block"
    if score >= review_at:
        return "review"
    return "allow"


def stricter(current: Decision | None, other: Decision | None) -> Decision | None:
    if current is None:
        return other
    if other is None:
        return current
    return current if _ORDER[current] >= _ORDER[other] else other
