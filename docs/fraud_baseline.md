# Fraud baseline (Phase 1)

LightGBM on IEEE-CIS with a strict time-based split (train early, validation middle, test
late). scale_pos_weight handles the ~1:27 imbalance. Threshold by expected cost (FN=10.0, FP=1.0).

## Time-based split (no leak)
- train.max(TransactionDT) = 8745772
- val range = [8745798, 12192842]
- test.min(TransactionDT) = 12192900
- train.max < test.min: True

## Metrics
| Metric | Validation | Test (late period) |
|---|---|---|
| PR-AUC | 0.3942 | 0.2632 |
| ROC-AUC | 0.8790 | 0.8447 |
| recall @ precision 0.9 | 0.0002 | 0.0027 |
| precision @ recall 0.5 | 0.3501 | 0.2195 |

Cost-optimal threshold on test: 0.0948 (FN cost 10.0, FP cost 1.0).

Precision-recall curve (test): `img/pr_curve.png`.
