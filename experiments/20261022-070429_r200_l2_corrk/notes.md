# 20261022-070429_r200_l2_corrk

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

In-fold top-k |corr| + L2, nested: 0.8478 +/- 0.0138 (k = 10 in 89/100 folds); within split noise of R003 L1-all. Negative result. (sup)
