# 20261022-065631_r200_l1_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.8693 +/- 0.0131 (std across repeats), nested C chose 0.2/0.5 in 95/100 folds. +0.022 vs R003's untuned L1 C=0.1 (split SE 0.0027, win rate 0.75) but corrected p = 0.22: tuning C helps a little and is not significant. Consistent with the hypothesis (linear 0.83-0.91). Decision: linear on all features is a negative result. (sup)
