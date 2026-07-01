# Time-based split

Fraud data is a time series: `TransactionDT` is seconds from a fixed origin. A random split would
let the model learn from future transactions and score past ones, which leaks and overstates
quality. The split here is strictly by time.

## Scheme
1. Sort all rows by `TransactionDT` (stable).
2. Take the earliest `train_frac` (default 0.6) as train, the next `val_frac` (0.2) as validation,
   the latest remainder (0.2) as test.
3. Guarantee: `train.max(TransactionDT) <= val.min <= val.max <= test.min`, so no future row is in
   train or validation relative to test.

`fraud_aml.data.split.time_based_split` returns disjoint index sets and is deterministic. The test
`tests/test_split.py` asserts `train.max < val.min < test.min` (no time leak), determinism, and
that the three sets partition the data. This test is unmarked, so it runs in the PR gate.

## Leakage controls beyond the split
- Frequency and target encoding are fit on the train rows only; validation and test are transformed
  with train statistics, and unseen categories fall back to the global prior (`fraud_aml.features.encoding`).
- `TransactionDT`, `TransactionID`, and the target are dropped from the feature matrix; only relative
  time features (`dt_hour`, `dt_day`, `card_dt_delta`) are kept, so the model cannot key on the
  absolute time boundary between train and test.
