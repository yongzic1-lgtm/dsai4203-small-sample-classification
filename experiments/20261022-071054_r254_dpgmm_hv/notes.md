# 20261022-071054_r254_dpgmm_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A DP mixture fitted on the 20,000 unlabeled rows is as good as the K=2 GMM on CV (AUC 1.0, saturated) and differs from S003 only on the ~200 boundary rows; CV log-loss within 10% of the GMM K=2 reference.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 1.0000 (saturated), CV log-loss 0.01707 vs 0.01711 for the GMM K=2 reference (S003 config): indistinguishable. n_eff = 2 (ELBO 178,510). Test: side agreement with S003 99.0% on the 204 boundary rows, Spearman 0.999, mean |diff| 0.004. A valid, redundant alternative to S003; no reason to submit it separately.
