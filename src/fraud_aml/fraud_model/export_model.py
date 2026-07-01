from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from fraud_aml.config import get_settings
from fraud_aml.data.loaders import load_merged
from fraud_aml.data.split import time_based_split
from fraud_aml.features.build import TARGET, TIME, add_all
from fraud_aml.fraud_model.dataset import assemble_features, fit_encoders
from fraud_aml.fraud_model.metrics import cost_optimal_threshold
from fraud_aml.fraud_model.train import fit_lightgbm
from fraud_aml.logging_config import configure_logging, get_logger


def build_and_bake(
    out_dir: Path,
    *,
    seed: int,
    fn_cost: float,
    fp_cost: float,
    train_frac: float,
    val_frac: float,
    version: str,
) -> dict[str, Any]:
    df = add_all(load_merged())
    split = time_based_split(df, time_col=TIME, train_frac=train_frac, val_frac=val_frac)
    y = df[TARGET].astype(int)
    freq, te = fit_encoders(df.loc[split.train_idx], y.loc[split.train_idx])
    features = assemble_features(df, freq, te)
    feature_order = [str(c) for c in features.columns]

    model = fit_lightgbm(
        features.loc[split.train_idx],
        y.loc[split.train_idx],
        features.loc[split.val_idx],
        y.loc[split.val_idx],
        seed=seed,
    )
    proba = model.predict_proba(features.loc[split.test_idx])[:, 1]
    threshold, _ = cost_optimal_threshold(
        y.loc[split.test_idx], proba, fn_cost=fn_cost, fp_cost=fp_cost
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump({"freq": freq, "te": te}, out_dir / "encoders.joblib")
    (out_dir / "feature_order.json").write_text(json.dumps(feature_order), encoding="utf-8")
    meta = {
        "model_version": version,
        "n_features": len(feature_order),
        "cost_threshold": float(threshold),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Phase 1 model and bake it for serving.")
    parser.add_argument("--out", default="deploy")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_logging()
    log = get_logger("export")
    settings = get_settings()
    meta = build_and_bake(
        Path(args.out),
        seed=settings.random_seed,
        fn_cost=settings.fraud_fn_cost,
        fp_cost=settings.fraud_fp_cost,
        train_frac=settings.train_frac,
        val_frac=settings.val_frac,
        version=settings.model_version_label,
    )
    log.info("baked", **meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
