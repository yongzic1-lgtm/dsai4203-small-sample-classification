# 20261022-193028_r422_linear_ceiling_r921

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. R921's converged fit and R154's fit describe the same 12-dim Gaussian pair, so the population best-linear AUC is 0.947 +- 0.002 under both (R154: formula 0.9468, MC 0.9466), the quadratic Bayes AUC 0.99995, the Bayes error 0.33 % +- 0.02 %, and ~186 +- 5 test rows are expected with 0.1 < p < 0.9. On the real test rows the converged posterior gives 189 (R922), the S003 mixture ~203 (r321) and S003's fold-mean score ~204 (r932): the definitions differ by ~10 %, not the model. Across R921 replicas the true count has sd ~13 (binomial), so 189 is typical; the refitted posterior's count is within a few rows of the true one. The refitted ceiling has sampling sd < 0.005. Linear models trained on 250 rows reach ~0.85-0.92 test AUC.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported. R921's converged fit and R154's fit give the same population best-linear AUC, 0.946843 (formula; MC 0.94665 vs 0.94663; LDA direction identical), Mahalanobis 2.2839, quadratic Bayes AUC 0.99995; the R154 MC values are reproduced exactly (same seed). Parametric bootstrap with R921's generator (200 replicas, refitted): ceiling 0.94696 +- 0.0014 (90 % 0.9445-0.9493); the population-optimal direction scores 0.9468 +- 0.0015 on a 19,750-row test set. Bayes error: expected 0.311 % +- 0.021 % per replica test set (60.6 Bayes errors of 19,750; MC 0.319 % for R921's fit vs R154's 0.327 %, both within ~2.5 MC SE). Boundary rows (0.1 < p < 0.9): population expectation 182.5 (R921) / 186.1 (R154, MC SE ~2), replica true counts 184.4 +- 14.1, the refitted posterior counts 0.6 +- 3.8 fewer rows than the truth; the real data's 189 sits at the 68th percentile. The circulating 203 (r321) and 204 (r932) use S003's score, which is a compressed rate (range 0.0061-0.9864), not a calibrated posterior: a different definition, not a different model. Linear models trained on 250 rows (high-variance block) reach 0.894 +- 0.006 (shrinkage LDA) and 0.912 +- 0.007 (L2 logistic regression) on replicas, 0.035-0.053 below the ceiling. (r2_noise)
