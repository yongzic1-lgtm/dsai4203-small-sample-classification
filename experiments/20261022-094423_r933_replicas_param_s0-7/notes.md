# 20261022-094423_r933_replicas_param_s0-7

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Predictions (written before running): on parametric replicas the GMM family (S003, r105, r155, r102) is within 1e-6 of the Bayes private AUC and within 1e-7 of each other (R103/R153); BQDA (r255) and self-training (r209) lose 1e-5 to 1e-4; strictly supervised nested QDA (r201, r210) loses 1e-4 to 1e-3 (R153: QDA reg 0.1 loses 7e-3, tuning removes most of it). Label noise lowers every procedure's private AUC by ~flip_y/2 and dominates its spread (sd ~4e-4 at flip 0.005), so private AUCs are correlated > 0.95 across procedures and the probability that any candidate beats S003 on the private LB is < 0.5 for all, near 0.5 only inside the GMM family where most replicas tie. The extended reg grid (x_qda_block_nll_ext) changes r210 by < 1e-4.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Preliminary (8 replicas, draw 0; pooled verdicts in r934/r935). Clean labels (flip 0): GMM family within 7e-9 of S003, S003 7.5e-7 below Bayes; every other candidate loses in 8/8 replicas: BQDA -1.4e-4, self-training -7.2e-5, r210 -2.0e-4, r201 -2.8e-4, ext-grid QDA -6.6e-5 (hypothesis ranges confirmed, BQDA slightly worse than predicted). With label noise applied to train AND test labels, r210 degrades to -4.4e-4 / -6.6e-4 / -1.3e-3 at flip 0.0025 / 0.005 / 0.01 (a mislabelled training row hurts the reg-at-grid-edge QDA), r201 only to -2.7e-4 / -3.0e-4 / -3.8e-4; but the real training labels show no confident contradiction (GMM CV 1.0 in 100/100 folds), so these train-noise replicas overstate the supervised loss -> follow-up runs with --train-noise 0 (test-label noise only) and --round-decimals 8 (the CSV format, which merges confident rows into ties). r102's mean private diff (-5e-5 / -1.2e-4 at 0.005 / 0.01) is tie-structure variance, not bias (expected-AUC diff -2e-10). Private sd across replicas ~4e-4 at flip >= 0.0025, dominated by test-label noise. (cand)
