from __future__ import annotations

import argparse
from pathlib import Path

from fraud_aml.data.loaders import load_merged


def build_eda_report(path: Path) -> None:
    df = load_merged()
    rate = float(df["isFraud"].mean())
    n_rows, n_cols = df.shape
    missing = df.isna().mean().sort_values(ascending=False)
    high_missing = int((missing > 0.5).sum())
    dt = df["TransactionDT"]
    span_days = float((dt.max() - dt.min()) / 86400)
    top_missing = missing.head(8)
    product = df.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False)

    lines = [
        "# IEEE-CIS EDA (Phase 1)",
        "",
        f"Merged shape: {n_rows} rows x {n_cols} columns (train_transaction left-joined with",
        "train_identity on TransactionID; identity is present for a minority of transactions).",
        "",
        "## Class balance",
        f"- Fraud rate: {rate:.4f} (~1:{(1 - rate) / rate:.0f}) - strong imbalance, handled with",
        "  scale_pos_weight, not blind oversampling.",
        "",
        "## Missingness",
        f"- {high_missing} columns are over 50% missing (identity and many V-features are sparse).",
        "  LightGBM keeps native NaN; a per-row n_missing feature is added.",
        "- Top missing columns:",
    ]
    lines += [f"  - {name}: {frac:.2f}" for name, frac in top_missing.items()]
    lines += [
        "",
        "## Time axis",
        f"- TransactionDT spans ~{span_days:.0f} days. Split is by time (docs/split.md):",
        "  train early, test late - no future leakage.",
        "",
        "## Categoricals",
        "- High cardinality (card1, addr1, email domains) - frequency and target encoding fit on",
        "  train only, not one-hot.",
        "- Fraud rate by ProductCD:",
    ]
    lines += [f"  - {name}: {value:.4f}" for name, value in product.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the IEEE-CIS EDA report to docs/eda.md.")
    parser.add_argument("--out", default="docs/eda.md")
    args = parser.parse_args(argv)
    build_eda_report(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
