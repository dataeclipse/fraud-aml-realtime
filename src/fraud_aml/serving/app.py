from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from fraud_aml import __version__
from fraud_aml.config import get_settings
from fraud_aml.logging_config import get_logger
from fraud_aml.serving.metrics import (
    DECISIONS,
    SCORE_HIST,
    SCORE_LATENCY,
    SCORE_REQUESTS,
    metrics_payload,
)
from fraud_aml.serving.schemas import ScoreRequest, ScoreResponse

if TYPE_CHECKING:
    from fraud_aml.serving.scoring import ScoringService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log = get_logger("service")
    settings = get_settings()
    try:
        from fraud_aml.serving.model_loader import get_feast_store, load_bundle
        from fraud_aml.serving.scoring import ScoringService

        bundle = load_bundle()
        store = None
        try:
            store = get_feast_store(settings)
        except Exception as exc:
            log.warning("feast_unavailable", error=str(exc))
        app.state.service = ScoringService(bundle, settings, feast_store=store)
        log.info("model_loaded", version=bundle["meta"].get("model_version"))
    except Exception as exc:
        app.state.service = None
        log.error("model_load_failed", error=str(exc))
    yield


app = FastAPI(title="Fraud and AML service", version=__version__, lifespan=lifespan)


def get_service(request: Request) -> ScoringService:
    service: ScoringService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return service


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest, service: ScoringService = Depends(get_service)) -> ScoreResponse:
    start = time.perf_counter()
    try:
        result = service.score(payload)
        SCORE_REQUESTS.labels(status="ok").inc()
        DECISIONS.labels(decision=result.decision).inc()
        SCORE_HIST.observe(result.score)
        return result
    except Exception:
        SCORE_REQUESTS.labels(status="error").inc()
        raise
    finally:
        SCORE_LATENCY.observe(time.perf_counter() - start)


@app.get("/healthz")
def healthz(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "service", None)
    return {
        "status": "ok" if service is not None else "degraded",
        "model_loaded": service is not None,
        "model_version": service.meta.get("model_version") if service is not None else None,
        "redis": service is not None and service.feast_store is not None,
        "version": __version__,
    }


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)
