# 20261022-064500_r102_semisup_eps01

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

250 labels vs 20,000 rows: parameters move negligibly; boundary Kendall vs S003 > 0.98.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Semi-supervised EM (soft constraints) ~ unsupervised + mapping: Kendall 0.9996 vs S003; better CV log-loss (0.0077 vs 0.0171) only because it outputs the posterior instead of rate-compressed scores (calibration, irrelevant for AUC). R103: +1.6e-10 (n.s.). Equivalent to S003, not worth a submission.
