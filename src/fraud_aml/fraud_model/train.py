from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fraud_aml.config import Settings, get_settings
from fraud_aml.fraud_model.metrics import (
    cost_optimal_threshold,
    pr_auc,
    precision_at_recall,
    recall_at_precision,
    roc_auc,
)


def fit_lightgbm(x_train: Any, y_train: Any, x_val: Any, y_val: Any, *, seed: int = 42) -> Any:
    import lightgbm as lgb

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale = negatives / max(positives, 1)
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def evaluate(model: Any, x: Any, y: Any, settings: Settings) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1]
    threshold, min_cost = cost_optimal_threshold(
        y, proba, fn_cost=settings.fraud_fn_cost, fp_cost=settings.fraud_fp_cost
    )
    return {
        "pr_auc": pr_auc(y, proba),
        "roc_auc": roc_auc(y, proba),
        "recall_at_precision": recall_at_precision(
            y, proba, precision_target=settings.target_precision
        ),
        "precision_at_recall": precision_at_recall(y, proba, recall_target=settings.target_recall),
        "cost_threshold": threshold,
        "min_cost": min_cost,
    }


def _pr_curve_png(y: Any, proba: Any, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y, proba)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (test)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _write_report(
    path: Path,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    data: Any,
    settings: Settings,
) -> None:
    lines = [
        "# Fraud baseline (Phase 1)",
        "",
        "LightGBM on IEEE-CIS with a strict time-based split (train early, validation middle, test",
        f"late). scale_pos_weight handles the ~1:27 imbalance. Threshold by expected cost "
        f"(FN={settings.fraud_fn_cost}, FP={settings.fraud_fp_cost}).",
        "",
        "## Time-based split (no leak)",
        f"- train.max(TransactionDT) = {data.train_max_dt:.0f}",
        f"- val range = [{data.val_min_dt:.0f}, {data.val_max_dt:.0f}]",
        f"- test.min(TransactionDT) = {data.test_min_dt:.0f}",
        f"- train.max < test.min: {data.train_max_dt < data.test_min_dt}",
        "",
        "## Metrics",
        "| Metric | Validation | Test (late period) |",
        "|---|---|---|",
        f"| PR-AUC | {val_metrics['pr_auc']:.4f} | {test_metrics['pr_auc']:.4f} |",
        f"| ROC-AUC | {val_metrics['roc_auc']:.4f} | {test_metrics['roc_auc']:.4f} |",
        f"| recall @ precision {settings.target_precision} | "
        f"{val_metrics['recall_at_precision']:.4f} | {test_metrics['recall_at_precision']:.4f} |",
        f"| precision @ recall {settings.target_recall} | "
        f"{val_metrics['precision_at_recall']:.4f} | {test_metrics['precision_at_recall']:.4f} |",
        "",
        f"Cost-optimal threshold on test: {test_metrics['cost_threshold']:.4f} "
        f"(FN cost {settings.fraud_fn_cost}, FP cost {settings.fraud_fp_cost}).",
        "",
        "Precision-recall curve (test): `img/pr_curve.png`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the IEEE-CIS fraud baseline (time-based split)."
    )
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    from fraud_aml.fraud_model.dataset import build_fraud_dataset
    from fraud_aml.logging_config import configure_logging, get_logger

    configure_logging()
    log = get_logger("fraud-baseline")
    settings = get_settings()

    data = build_fraud_dataset(settings, use_cache=not args.no_cache)
    model = fit_lightgbm(
        data.X_train, data.y_train, data.X_val, data.y_val, seed=settings.random_seed
    )
    val_metrics = evaluate(model, data.X_val, data.y_val, settings)
    test_metrics = evaluate(model, data.X_test, data.y_test, settings)

    docs = Path(args.docs_dir)
    proba = model.predict_proba(data.X_test)[:, 1]
    _pr_curve_png(data.y_test, proba, docs / "img" / "pr_curve.png")
    _write_report(docs / "fraud_baseline.md", val_metrics, test_metrics, data, settings)

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri or "sqlite:///mlflow.db")
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="fraud-baseline-lgbm"):
        mlflow.log_params(
            {
                "scale_pos_weight": "neg/pos",
                "seed": settings.random_seed,
                "n_features": len(data.feature_names),
                "split": "time-based",
                "fn_cost": settings.fraud_fn_cost,
                "fp_cost": settings.fraud_fp_cost,
            }
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(docs / "img" / "pr_curve.png"))

    print(f"VAL  pr_auc={val_metrics['pr_auc']:.4f} roc_auc={val_metrics['roc_auc']:.4f}")
    print(f"TEST pr_auc={test_metrics['pr_auc']:.4f} roc_auc={test_metrics['roc_auc']:.4f}")
    print(
        f"SPLIT train.max_dt={data.train_max_dt:.0f} test.min_dt={data.test_min_dt:.0f} "
        f"train.max<test.min={data.train_max_dt < data.test_min_dt}"
    )
    print(
        f"COST fn={settings.fraud_fn_cost} fp={settings.fraud_fp_cost} "
        f"threshold={test_metrics['cost_threshold']:.4f} min_cost={test_metrics['min_cost']:.0f}"
    )
    log.info("fraud_baseline_done", test_pr_auc=round(test_metrics["pr_auc"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
