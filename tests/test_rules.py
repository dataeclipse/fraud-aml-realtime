from __future__ import annotations

from fraud_aml.config import Settings
from fraud_aml.serving.rules import apply_rules


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_no_rules_when_quiet() -> None:
    result = apply_rules({"win_count": 1.0, "win_sum": 10.0, "win_velocity": 0.1}, _settings())
    assert result.fired == []
    assert result.forced is None


def test_count_forces_review() -> None:
    settings = _settings()
    result = apply_rules({"win_count": settings.win_count_limit + 1}, settings)
    assert result.forced == "review"
    assert any("count_over" in f for f in result.fired)


def test_sum_forces_block() -> None:
    settings = _settings()
    result = apply_rules({"win_sum": settings.win_sum_limit + 1}, settings)
    assert result.forced == "block"


def test_rules_only_escalate() -> None:
    settings = _settings()
    result = apply_rules(
        {"win_count": settings.win_count_limit + 1, "win_sum": settings.win_sum_limit + 1},
        settings,
    )
    assert result.forced == "block"
