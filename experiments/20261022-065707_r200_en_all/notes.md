# 20261022-065707_r200_en_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.8552 +/- 0.0129, nested (C, l1_ratio) chose l1_ratio 0.8 in 97/100 folds: elastic net behaves like L1 and is not better than L1 (0.8693). Negative result. (sup)
