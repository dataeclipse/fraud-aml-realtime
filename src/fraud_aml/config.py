from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    random_seed: int = 42
    log_level: str = "INFO"

    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "fraud-aml"

    kafka_bootstrap: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"

    psi_threshold: float = 0.2
    review_at: float = 0.3
    block_at: float = 0.7

    train_frac: float = 0.6
    val_frac: float = 0.2
    fraud_fn_cost: float = 10.0
    fraud_fp_cost: float = 1.0
    target_precision: float = 0.9
    target_recall: float = 0.5

    stream_topic: str = "transactions"
    window_seconds: int = 3600
    speedup: float = 1000.0
    online_ttl_seconds: int = 86400


@lru_cache
def get_settings() -> Settings:
    return Settings()
