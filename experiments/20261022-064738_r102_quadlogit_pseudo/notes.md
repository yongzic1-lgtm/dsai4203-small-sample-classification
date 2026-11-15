# 20261022-064738_r102_quadlogit_pseudo

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Distils the GMM log-odds; ranking close to S003 (Kendall > 0.95), no gain.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Negative: the regularised quadratic logistic model on soft pseudo-labels cannot reproduce the sharp mixture log-odds (2,407 test rows in (0.1, 0.9), Kendall 0.36 vs S003, log-loss 0.051); R103 -4.5e-4 test AUC vs S003. Rejected.
