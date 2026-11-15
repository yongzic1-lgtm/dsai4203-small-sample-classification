# 20261022-161642_r933_replicas_param_s72-135

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Predictions (written before running): on parametric replicas the GMM family (S003, r105, r155, r102) is within 1e-6 of the Bayes private AUC and within 1e-7 of each other (R103/R153); BQDA (r255) and self-training (r209) lose 1e-5 to 1e-4; strictly supervised nested QDA (r201, r210) loses 1e-4 to 1e-3 (R153: QDA reg 0.1 loses 7e-3, tuning removes most of it). Label noise lowers every procedure's private AUC by ~flip_y/2 and dominates its spread (sd ~4e-4 at flip 0.005), so private AUCs are correlated > 0.95 across procedures and the probability that any candidate beats S003 on the private LB is < 0.5 for all, near 0.5 only inside the GMM family where most replicas tie. The extended reg grid (x_qda_block_nll_ext) changes r210 by < 1e-4.

## Result

(see summary.json)

## Conclusion (write after reading the results)

64 replicas (draw 0; pooled verdicts in r936), consistent with s0-7 and s8-71. Clean labels: GMM family = S003 (r102 ties exactly in 27/64), every other candidate loses in 64/64: BQDA -1.34e-4, self-training -6.6e-5, r210 -1.67e-4, r201 -2.67e-4, ext-grid -6.3e-5. With train+test label noise r210 degrades to -4.0e-4 / -7.3e-4 / -1.65e-3 at flip 0.0025 / 0.005 / 0.01 (sd up to 3.6e-3: a few replicas with a flipped training row break it), r201 only to -2.7e-4 / -3.0e-4 / -3.5e-4; BQDA and self-training are insensitive to train noise. No candidate's mean exceeds S003 at any level. Hypothesis confirmed except the r210-vs-r201 ordering under train-label noise. (cand)
