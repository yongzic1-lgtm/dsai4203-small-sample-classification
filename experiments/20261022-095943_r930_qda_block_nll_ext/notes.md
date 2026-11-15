# 20261022-095943_r930_qda_block_nll_ext

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Written before running. sklearn QDA's rank check (tol 1e-4) makes reg < 1e-4 fail on the rank-12 block, so the inner log-loss can only move to 3e-4 or 1e-4; prediction: it does (r931 smoke test on 10 folds chose 1e-4/3e-4), outer AUC stays within 1e-4 of r210 (0.99998) and outer OOF log-loss changes by < 0.003. The AUC-tuned version keeps 0.001-0.01 (ties go to the first grid entry). Either way the CV comparison with the GMM stays non-significant because both are saturated; D009's 'indistinguishable in CV' is not driven by the grid edge.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Grid edge confirmed but harmless for AUC: with the grid extended below r210's edge, inner log-loss picks 1e-4 in 53, 3e-4 in 36 and 0.001 in 11 folds (1e-4 is again an edge: sklearn's rank check tol 1e-4 makes reg < 1e-4 fail on the rank-12 block, those configs score NaN). Outer CV AUC 0.99998 +/- 0.00007 (identical to r210, 1 fold below 1.0), OOF log-loss 0.0076 vs r210's 0.0087. Paired vs TransductiveGMM: -1.6e-5, corrected p 0.84, 95% interval with corrected SE [-1.8e-4, +1.5e-4]. Prediction confirmed (AUC within 1e-4, log-loss -0.0011). This is a post-hoc variant (grid widened after seeing r210); not an honest estimate of a pre-registered procedure. (cand)
