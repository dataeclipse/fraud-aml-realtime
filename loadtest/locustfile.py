from __future__ import annotations

import random
from typing import Any

from locust import HttpUser, between, task


def _payload() -> dict[str, Any]:
    return {
        "TransactionID": random.randint(1, 10**9),
        "card1": random.randint(1000, 18000),
        "TransactionAmt": round(random.uniform(10, 500), 2),
        "TransactionDT": random.randint(86400, 15000000),
        "features": {
            "addr1": float(random.randint(100, 500)),
            "C1": float(random.randint(0, 30)),
            "D1": float(random.randint(0, 600)),
        },
    }


class ScoreUser(HttpUser):
    wait_time = between(0.0, 0.05)

    @task
    def score(self) -> None:
        self.client.post("/score", json=_payload())
