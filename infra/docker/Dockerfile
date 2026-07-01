# syntax=docker/dockerfile:1
# Multi-stage, non-root. Lean api stack (serve + Feast/Redis client), no bytewax/kafka/torch.
# The fraud model is baked into deploy/ (fraud-aml-export-model). Build context is the project root.

# ---------- builder ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api --no-dev

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

# LightGBM links against the OpenMP runtime (libgomp), absent in python:3.12-slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
COPY --chown=appuser:appuser deploy ./deploy
COPY --chown=appuser:appuser feast_repo ./feast_repo

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FRAUD_REDIS_URL=redis://redis:6379/0

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "fraud_aml.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
