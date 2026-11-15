# 20261022-073743_r251_tabpfn2_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A tabular foundation model (tabpfn) used as a 200-row in-context learner should match or beat the best supervised baselines on the same inputs (L1 all-features 0.847; QDA block 0.996) because its synthetic prior covers covariance-type class differences; predicted CV AUC 0.85-0.95 on all 150 features, still below the transductive GMM (1.000). Refuted if below L1 (all) / below L2 logistic 0.898 (block).

## Result

(see summary.json)

## Conclusion (write after reading the results)

TabPFN v2 weights (ModelVersion.V2, Prior Labs License; downloaded anonymously, see R250), 8 estimators, raw 150 features, in-context on the fold's 200 rows: CV AUC 0.9851 +/- 0.0041, log-loss 0.167. vs L1 +0.138 (p 2e-7), vs L2-block +0.087, vs QDA(reg 0.1) -0.011 (split-SE 0.001, p 0.041), vs GMM -0.015. Hypothesis (0.85-0.95 on all features) refuted upwards: without any feature selection the prior-fitted network finds the covariance signal far better than the linear models, but stays below QDA / Bayesian QDA (1.000). Test predictions = refit on all 250 rows (test_pred=refit). Test: pseudo-AUC vs S003 labels 0.986; 23% of test rows have scores in (0.1, 0.9) (S003: 1%). Not proposed for submission (rule-4 question, and worse than in-house models).
