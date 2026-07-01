from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from fraud_aml.serving.app import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
