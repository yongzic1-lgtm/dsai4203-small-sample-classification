# 20261022-070353_r200_l1_corrk

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

In-fold top-k |corr| + L1, nested k x C: 0.8562 +/- 0.0138 (k = 10 in 70/100 folds). Supervised univariate filtering cannot find the informative subspace: below the label-free variance block (0.932). Negative result for in-fold correlation selection. (sup)
