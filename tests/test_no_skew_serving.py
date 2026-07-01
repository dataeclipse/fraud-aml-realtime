from __future__ import annotations

import pandas as pd

from fraud_aml.config import Settings
from fraud_aml.features.build import TARGET, add_all
from fraud_aml.fraud_model.dataset import assemble_features, fit_encoders
from fraud_aml.serving.schemas import ScoreRequest
from fraud_aml.serving.scoring import ScoringService


def test_serving_vector_matches_training() -> None:
    raw = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "card1": [100, 100, 100],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "TransactionDT": [0, 50, 130],
            "addr1": [200.0, 200.0, 200.0],
            "isFraud": [0, 0, 1],
        }
    )
    df = add_all(raw)
    y = df[TARGET].astype(int)
    freq, te = fit_encoders(df, y)
    train_features = assemble_features(df, freq, te)
    order = [str(c) for c in train_features.columns]

    bundle = {"model": None, "freq": freq, "te": te, "feature_order": order, "meta": {}}
    service = ScoringService(bundle, Settings(_env_file=None))

    request = ScoreRequest(
        TransactionID=3,
        card1=100,
        TransactionAmt=30.0,
        TransactionDT=130,
        features={"addr1": 200.0},
    )
    serving = service.build_vector(request, {"win_delta": float(130 - 50)})

    expected = train_features.iloc[[2]].reset_index(drop=True)[order]
    got = serving.reset_index(drop=True)[order]
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
