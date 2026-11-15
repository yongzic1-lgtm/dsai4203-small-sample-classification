# 20261022-070235_r200_lda_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Shrinkage LDA on all 150 features: 0.7962 +/- 0.0124, the worst linear model; Ledoit-Wolf shrinkage of a 150x150 covariance from 200 rows still lets the noise columns dominate. Negative result. (sup)
