from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

SCORE_REQUESTS = Counter("fraud_score_requests_total", "/score requests", ["status"])
DECISIONS = Counter("fraud_decisions_total", "Decisions", ["decision"])
SCORE_LATENCY = Histogram("fraud_score_latency_seconds", "/score latency")
SCORE_HIST = Histogram(
    "fraud_predicted_score",
    "Predicted fraud score",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
