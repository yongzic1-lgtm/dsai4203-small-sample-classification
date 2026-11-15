# 20261022-064749_r102_supsel13_gmm2

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Supervised selection misses jointly-informative block features (R100), so the GMM on the selected set is worse than the label-free block (lower CV log-loss quality, lower expected AUC).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Negative: supervised in-fold selection (top-13 marginal LRT on 200 labelled rows) is the only transductive GMM below CV AUC 1 (0.99997 +- 0.0001; log-loss 0.025, 702 test rows in (0.1, 0.9), Kendall 0.70 vs S003): it picks redundant features (17, 6) and misses 15/26/30. R103 -3.5e-4 test AUC vs S003. Label-free selection wins.
