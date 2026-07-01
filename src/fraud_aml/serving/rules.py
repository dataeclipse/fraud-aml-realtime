from __future__ import annotations

from dataclasses import dataclass

from fraud_aml.config import Settings
from fraud_aml.decision import Decision, stricter


@dataclass(frozen=True)
class RuleResult:
    fired: list[str]
    forced: Decision | None


def apply_rules(online: dict[str, float], settings: Settings) -> RuleResult:
    count = float(online.get("win_count", 0.0) or 0.0)
    total = float(online.get("win_sum", 0.0) or 0.0)
    velocity = float(online.get("win_velocity", 0.0) or 0.0)

    fired: list[str] = []
    forced: Decision | None = None
    if count > settings.win_count_limit:
        fired.append(f"count_over_{settings.win_count_limit:g}")
        forced = stricter(forced, "review")
    if velocity > settings.win_velocity_limit:
        fired.append(f"velocity_over_{settings.win_velocity_limit:g}")
        forced = stricter(forced, "review")
    if total > settings.win_sum_limit:
        fired.append(f"sum_over_{settings.win_sum_limit:g}")
        forced = stricter(forced, "block")
    return RuleResult(fired=fired, forced=forced)
