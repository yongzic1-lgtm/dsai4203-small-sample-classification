# 20261022-070309_r200_en_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.9250 +/- 0.0095, (C=3, l1_ratio=0.8) in 100/100 folds; C=3 is the top of the EN grid (grid edge). Slightly below L1/L2 block with larger C. +0.047 vs R003 LDA block (p_corr < 0.001). (sup)
