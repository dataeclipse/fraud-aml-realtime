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


@lru_cache
def get_settings() -> Settings:
    return Settings()
