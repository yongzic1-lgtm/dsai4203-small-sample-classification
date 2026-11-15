# 20261022-100014_r930_qda_block_auc_ext

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Written before running. sklearn QDA's rank check (tol 1e-4) makes reg < 1e-4 fail on the rank-12 block, so the inner log-loss can only move to 3e-4 or 1e-4; prediction: it does (r931 smoke test on 10 folds chose 1e-4/3e-4), outer AUC stays within 1e-4 of r210 (0.99998) and outer OOF log-loss changes by < 0.003. The AUC-tuned version keeps 0.001-0.01 (ties go to the first grid entry). Either way the CV comparison with the GMM stays non-significant because both are saturated; D009's 'indistinguishable in CV' is not driven by the grid edge.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Inner-AUC tuning over the extended grid chooses 0.003 (40), 0.001 (30), 3e-4 (17), 0.01 (10), 1e-4 (3): ties still go to the first (largest) reg, smaller values only win when strictly better. Outer AUC 0.99990 +/- 0.00021, log-loss 0.0144, identical AUC to r201 (4 folds below 1.0); vs GMM -9.7e-5, corrected p 0.71, sign test 4-0 (p 0.125), 95% corrected interval [-6.1e-4, +4.2e-4]. Prediction confirmed. (cand)
