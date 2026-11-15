# 20261022-070207_r255_gpc_ard_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A GP classifier with an ARD RBF kernel can represent the quadratic (covariance) boundary; predicted CV AUC between L2 logistic (0.898) and QDA (0.996).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Negative result: GP classifier (Laplace, ARD RBF) on the block 0.9778 +/- 0.0046, clearly below QDA (-0.018, split-SE 0.001, corrected p 0.019, win rate 0.055). A stationary RBF kernel does not capture the quadratic (covariance) boundary as efficiently as a Gaussian class-conditional model with 200 rows. Hypothesis (between 0.898 and 0.996) confirmed.
