# Deploy (brief)

The api image is self-contained: the fraud model is baked into `deploy/` (joblib) and the runtime is
the lean api stack (serve + Feast/Redis client), no bytewax/kafka/torch. Secrets via env only.

## Build
```bash
uv run fraud-aml-export-model                                   # bake the model -> deploy/
docker build -f infra/docker/Dockerfile -t fraud-aml-api:1.0.0 .
```

## VPS
```bash
docker compose -f infra/compose.yaml up -d      # redpanda + redis + api
```
Point `FRAUD_REDIS_URL` and the Feast online store at the Redis host. A HEALTHCHECK on `/healthz` is
built in. The producer and processor run separately via `uv` (see docs/serving.md), not in the image.

## Fly.io
```bash
fly launch --no-deploy            # internal_port 8000, health-check GET /healthz
fly deploy --dockerfile infra/docker/Dockerfile
```
Provision Redis (`fly redis create`) and set `FRAUD_REDIS_URL`.

## Notes
- A new model = re-run `fraud-aml-export-model` and rebuild, or mount `deploy/` as a volume.
- If the online store is unreachable, `/score` still returns an ML decision with empty online
  features (the rule engine is a no-op) - it degrades rather than failing.
