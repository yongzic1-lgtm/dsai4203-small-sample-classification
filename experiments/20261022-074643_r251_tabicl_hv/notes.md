# 20261022-074643_r251_tabicl_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A tabular foundation model (tabicl) used as a 200-row in-context learner should match or beat the best supervised baselines on the same inputs (L1 all-features 0.847; QDA block 0.996) because its synthetic prior covers covariance-type class differences; predicted CV AUC 0.97-0.999 on the block, still below the transductive GMM (1.000). Refuted if below L1 (all) / below L2 logistic 0.898 (block).

## Result

(see summary.json)

## Conclusion (write after reading the results)

TabICL on the block: 0.9989 +/- 0.0008, log-loss 0.047: +0.0031 vs QDA(reg 0.1) (split-SE 0.0003, corrected p 0.20), -0.0011 vs GMM; comparable to lightly regularised QDA (r256 reg 0.01: 0.9998) and below Bayesian QDA (r255: 1.0000, log-loss 0.009). Hypothesis (0.97-0.999) confirmed. Boundary side agreement with S003 57%: its boundary ranking is essentially unrelated to the GMM's (Spearman 0.38).
