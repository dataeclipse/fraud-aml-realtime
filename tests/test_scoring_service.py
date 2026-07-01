from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_aml.config import Settings
from fraud_aml.features.build import TARGET, add_all
from fraud_aml.fraud_model.dataset import assemble_features, fit_encoders
from fraud_aml.fraud_model.train import fit_lightgbm
from fraud_aml.serving.schemas import ScoreRequest
from fraud_aml.serving.scoring import ScoringService

pytestmark = pytest.mark.fraud


class _Result:
    def to_dict(self) -> dict[str, Any]:
        return {
            "win_count": [25.0],
            "win_sum": [100.0],
            "win_velocity": [0.1],
            "win_delta": [80.0],
        }


class _FakeStore:
    def get_online_features(self, features: Any, entity_rows: Any) -> _Result:
        return _Result()


def test_end_to_end_score() -> None:
    rng = np.random.default_rng(0)
    n = 200
    raw = pd.DataFrame(
        {
            "TransactionID": range(n),
            "card1": rng.integers(1, 20, n),
            "TransactionAmt": rng.random(n) * 100,
            "TransactionDT": np.arange(n) * 10,
            "addr1": rng.integers(100, 300, n).astype(float),
            "isFraud": (rng.random(n) > 0.8).astype(int),
        }
    )
    df = add_all(raw)
    y = df[TARGET].astype(int)
    freq, te = fit_encoders(df, y)
    feats = assemble_features(df, freq, te)
    order = [str(c) for c in feats.columns]
    model = fit_lightgbm(feats.iloc[:150], y.iloc[:150], feats.iloc[150:], y.iloc[150:], seed=1)
    bundle = {
        "model": model,
        "freq": freq,
        "te": te,
        "feature_order": order,
        "meta": {"model_version": "test"},
    }
    service = ScoringService(bundle, Settings(_env_file=None), feast_store=_FakeStore())

    request = ScoreRequest(
        TransactionID=999,
        card1=5,
        TransactionAmt=50.0,
        TransactionDT=2000,
        features={"addr1": 200.0},
    )
    response = service.score(request)
    assert 0.0 <= response.score <= 1.0
    assert response.decision in ("review", "block")
    assert any("count_over" in f for f in response.fired_rules)
    assert response.model_version == "test"
