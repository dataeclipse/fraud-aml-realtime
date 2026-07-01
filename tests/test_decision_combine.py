from __future__ import annotations

from fraud_aml.config import Settings
from fraud_aml.decision import stricter
from fraud_aml.serving.scoring import ScoringService


def _service() -> ScoringService:
    bundle = {"model": None, "freq": None, "te": None, "feature_order": [], "meta": {}}
    return ScoringService(bundle, Settings(_env_file=None))


def test_ml_decision_bands() -> None:
    service = _service()
    assert service._ml_decision(0.005) == "allow"
    assert service._ml_decision(0.05) == "review"
    assert service._ml_decision(0.2) == "block"


def test_rules_escalate_only() -> None:
    assert stricter("allow", "review") == "review"
    assert stricter("block", "review") == "block"
    assert stricter("review", None) == "review"
    assert stricter("allow", None) == "allow"
