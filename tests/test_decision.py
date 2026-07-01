from __future__ import annotations

from fraud_aml.decision import decide


def test_decision_thresholds() -> None:
    assert decide(0.1) == "allow"
    assert decide(0.3) == "review"
    assert decide(0.5) == "review"
    assert decide(0.7) == "block"
    assert decide(0.95) == "block"
