# 20261022-074251_r251_tabicl_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A tabular foundation model (tabicl) used as a 200-row in-context learner should match or beat the best supervised baselines on the same inputs (L1 all-features 0.847; QDA block 0.996) because its synthetic prior covers covariance-type class differences; predicted CV AUC 0.85-0.95 on all 150 features, still below the transductive GMM (1.000). Refuted if below L1 (all) / below L2 logistic 0.898 (block).

## Result

(see summary.json)

## Conclusion (write after reading the results)

TabICL (checkpoint tabicl-classifier-v2-20260212, BSD-3, anonymous download, R250), 8 estimators, raw 150 features: CV AUC 0.9929 +/- 0.0021, log-loss 0.120. vs TabPFN v2 (all) about +0.008; vs QDA(reg 0.1) -0.003 (p 0.42, within noise in the data-sampling view); vs GMM -0.007. Hypothesis (0.85-0.95) refuted upwards: the strongest model that uses no feature selection and no unlabeled data, yet below Bayesian QDA on the block (1.000). Refit test predictions; pseudo-AUC vs S003 0.995.
