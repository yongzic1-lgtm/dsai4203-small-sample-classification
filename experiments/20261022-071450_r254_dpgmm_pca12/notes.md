# 20261022-071450_r254_dpgmm_pca12

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A DP mixture fitted on the 20,000 unlabeled rows is as good as the K=2 GMM on CV (AUC 1.0, saturated) and differs from S003 only on the ~200 boundary rows; CV log-loss within 10% of the GMM K=2 reference.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 1.0000, log-loss 0.0192 (slightly worse than the raw-block versions), n_eff = 2; boundary side agreement with S003 99.5%. Same conclusion as dpgmm_hv.
