from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from fraud_aml.config import Settings


def load_bundle(deploy_dir: Path = Path("deploy")) -> dict[str, Any]:
    encoders = joblib.load(deploy_dir / "encoders.joblib")
    feature_order = json.loads((deploy_dir / "feature_order.json").read_text(encoding="utf-8"))
    meta = json.loads((deploy_dir / "meta.json").read_text(encoding="utf-8"))
    return {
        "model": joblib.load(deploy_dir / "model.joblib"),
        "freq": encoders["freq"],
        "te": encoders["te"],
        "feature_order": feature_order,
        "meta": meta,
    }


def get_feast_store(settings: Settings) -> Any:
    from fraud_aml.streaming.online_store import get_store

    return get_store()
