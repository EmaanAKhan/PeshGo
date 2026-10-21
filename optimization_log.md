# Optimization log

Everything here was measured on the **validation** period; the test set was scored once at the end. The numbers are copied from `reports/metrics.json`.

## Reference points (validation, MAE in µg/m³)
| Predictor | MAE |
|---|---|
| Persistence baseline (declared before results) | 46.05 |
| CAMS forecast | 88.4 |
| **Chosen model** | **38.35** |

## 1. Hyper-parameter search (median model, trained on the training period only)
| # | Settings | Validation MAE | Chosen |
|---|---|---|---|
| 1 | learning rate 0.1, 300 rounds, 31 leaves, min 40 per leaf | 39.392 | no |
| 2 | learning rate 0.1, 400 rounds, 15 leaves, min 80 per leaf | 38.354 | yes |
| 3 | learning rate 0.1, 200 rounds, 63 leaves, min 20 per leaf | 38.945 | no |

## 2. Calibrating the 80% range (conformal adjustment)
The raw 10th and 90th percentile models covered **54%** of validation outcomes. Widening the range by 25.75 µg/m³, the margin that reaches 80% on validation, gave **80%** on validation and **92%** on the test period (too wide there, because validation is winter only). Kept, because narrow ranges that miss are worse than wide ranges that hold.

## 3. Efficiency
| Item | Value |
|---|---|
| Training time (all fits) | 285.7 s on a busy laptop CPU |
| Peak memory during training | 24.7 MB |
| Median hourly pipeline run | 178 s |

