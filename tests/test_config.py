from __future__ import annotations

from pathlib import Path

from fraud_aml.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.random_seed == 42
    assert settings.data_dir == Path("data")
    assert 0.0 < settings.review_at < settings.block_at < 1.0
