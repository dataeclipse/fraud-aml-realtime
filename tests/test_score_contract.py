from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fraud_aml.serving.app import app, get_service
from fraud_aml.serving.schemas import ScoreRequest, ScoreResponse

_VALID = {
    "TransactionID": 1,
    "card1": 12345,
    "TransactionAmt": 100.0,
    "TransactionDT": 86400,
    "features": {"addr1": 200.0},
}


class _StubService:
    def score(self, request: ScoreRequest) -> ScoreResponse:
        return ScoreResponse(
            score=0.2,
            decision="review",
            fired_rules=["velocity_over_5"],
            top_reason="card1_freq (increases risk)",
            model_version="test",
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_service] = _StubService
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_score_valid(client: TestClient) -> None:
    response = client.post("/score", json=_VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("allow", "review", "block")
    assert isinstance(body["fired_rules"], list)
    assert body["model_version"] == "test"


def test_score_rejects_unknown_field(client: TestClient) -> None:
    assert client.post("/score", json={**_VALID, "surprise": 1}).status_code == 422


def test_score_rejects_bad_amount(client: TestClient) -> None:
    assert client.post("/score", json={**_VALID, "TransactionAmt": -5.0}).status_code == 422


def test_score_rejects_missing_card(client: TestClient) -> None:
    bad = {k: v for k, v in _VALID.items() if k != "card1"}
    assert client.post("/score", json=bad).status_code == 422
