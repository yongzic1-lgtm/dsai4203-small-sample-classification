# 20261022-064735_r102_selftrain_qda99

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Fitting on the confident core shrinks the class covariances: a small but measurable loss in expected AUC.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Self-training QDA on confident pseudo-labels: Kendall 0.98, 4 side flips; R103 -2.1e-8 (clean) / -2.9e-7 (label noise) vs S003, as expected from fitting on the truncated confident core. Negative result.
