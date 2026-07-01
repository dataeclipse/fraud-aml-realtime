from __future__ import annotations

from fastapi import FastAPI

from fraud_aml import __version__

app = FastAPI(title="Fraud and AML service", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"status": "ok", "version": __version__}
