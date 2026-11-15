# 20261022-064454_r102_gmm2_block_bag20

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Bagging reflects parameter uncertainty; changes boundary posteriors by < 0.02 and the ranking slightly.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Bagging re-orders the boundary rows a little (Kendall 0.98, 2 side flips, log-loss 0.0174) without gain: R103 replicas -2.4e-8 +- 0.4e-8 expected AUC vs S003 (slightly worse, negligible). Rejected.
