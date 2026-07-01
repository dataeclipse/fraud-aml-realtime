# IEEE-CIS EDA (Phase 1)

Merged shape: 590540 rows x 434 columns (train_transaction left-joined with
train_identity on TransactionID; identity is present for a minority of transactions).

## Class balance
- Fraud rate: 0.0350 (~1:28) - strong imbalance, handled with
  scale_pos_weight, not blind oversampling.

## Missingness
- 214 columns are over 50% missing (identity and many V-features are sparse).
  LightGBM keeps native NaN; a per-row n_missing feature is added.
- Top missing columns:
  - id_24: 0.99
  - id_25: 0.99
  - id_07: 0.99
  - id_08: 0.99
  - id_21: 0.99
  - id_26: 0.99
  - id_27: 0.99
  - id_23: 0.99

## Time axis
- TransactionDT spans ~182 days. Split is by time (docs/split.md):
  train early, test late - no future leakage.

## Categoricals
- High cardinality (card1, addr1, email domains) - frequency and target encoding fit on
  train only, not one-hot.
- Fraud rate by ProductCD:
  - C: 0.1169
  - S: 0.0590
  - H: 0.0477
  - R: 0.0378
  - W: 0.0204
