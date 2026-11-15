# 20261022-160545_r933_replicas_param_s8-71

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Predictions (written before running): on parametric replicas the GMM family (S003, r105, r155, r102) is within 1e-6 of the Bayes private AUC and within 1e-7 of each other (R103/R153); BQDA (r255) and self-training (r209) lose 1e-5 to 1e-4; strictly supervised nested QDA (r201, r210) loses 1e-4 to 1e-3 (R153: QDA reg 0.1 loses 7e-3, tuning removes most of it). Label noise lowers every procedure's private AUC by ~flip_y/2 and dominates its spread (sd ~4e-4 at flip 0.005), so private AUCs are correlated > 0.95 across procedures and the probability that any candidate beats S003 on the private LB is < 0.5 for all, near 0.5 only inside the GMM family where most replicas tie. The extended reg grid (x_qda_block_nll_ext) changes r210 by < 1e-4.

## Result

(see summary.json)

## Conclusion (write after reading the results)

64 replicas (draw 0; pooled verdicts in r936), consistent with s0-7. Clean labels: GMM family = S003 to < 1e-9 (r105 / r155 / r102 tie or differ only through tie-breaking), S003 3e-7 below Bayes; every other candidate loses in 64/64 replicas: BQDA -1.16e-4, self-training -6.4e-5, r210 -1.56e-4, r201 -2.37e-4, ext-grid -5.2e-5. With train+test label noise r210 degrades to -3.3e-4 / -5.7e-4 / -1.32e-3 at flip 0.0025 / 0.005 / 0.01 (r201 -2.8e-4 / -3.0e-4 / -3.3e-4), confirming that log-loss tuning to the smallest reg is fragile to mislabelled training rows; no candidate's mean is above S003 at any level (r102 +8e-6 at 0.01 is tie-breaking noise; expected-AUC diff 0). Hypothesis confirmed except that r210 is worse than r201 once training labels are noisy. (cand)
