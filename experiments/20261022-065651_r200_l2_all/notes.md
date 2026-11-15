# 20261022-065651_r200_l2_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.7870 +/- 0.0122; the inner CV picked the smallest C (0.001-0.01) in 93/100 folds, i.e. the most shrunken (near mean-difference) rule; -0.060 vs R003 L1 all (p_corr 0.002). L2 on 150 features cannot ignore the 137 noise columns. Negative result; grid lower edge reached (0.001), but even stronger shrinkage converges to a fixed direction, so no further gain is expected. (sup)
