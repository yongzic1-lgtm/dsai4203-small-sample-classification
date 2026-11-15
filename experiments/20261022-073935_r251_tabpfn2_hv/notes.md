# 20261022-073935_r251_tabpfn2_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A tabular foundation model (tabpfn) used as a 200-row in-context learner should match or beat the best supervised baselines on the same inputs (L1 all-features 0.847; QDA block 0.996) because its synthetic prior covers covariance-type class differences; predicted CV AUC 0.97-0.999 on the block, still below the transductive GMM (1.000). Refuted if below L1 (all) / below L2 logistic 0.898 (block).

## Result

(see summary.json)

## Conclusion (write after reading the results)

TabPFN v2 on the 13-column block: 0.9888 +/- 0.0031 (vs QDA -0.007, p 0.07; vs GMM -0.011). Hypothesis (0.97-0.999) confirmed, but below the classical/Bayesian Gaussian models. Refit test predictions.
