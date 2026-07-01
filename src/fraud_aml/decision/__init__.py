from __future__ import annotations

from typing import Literal

Decision = Literal["allow", "review", "block"]


def decide(score: float, *, review_at: float = 0.3, block_at: float = 0.7) -> Decision:
    if score >= block_at:
        return "block"
    if score >= review_at:
        return "review"
    return "allow"
