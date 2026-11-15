# 20261022-071541_r201_gnb_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Gaussian naive Bayes keeps class-specific variances but drops correlations. Prediction: between LDA (0.878, means only) and QDA (0.996, full covariance): 0.92-0.98 on the block; worse on all features.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Gaussian naive Bayes (class-specific variances, no correlations) on the in-fold block: 0.8668 +/- 0.0050, i.e. no better than LDA (means only, ~0.86-0.88). The hypothesis (0.92-0.98) is refuted: the class difference sits in the correlations (off-diagonal covariance), not in per-feature variances or means. Key ablation for the 'covariance signal' story. (sup)
