# 20261022-070352_r200_lda_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Shrinkage LDA on the in-fold block: 0.8592 +/- 0.0073, -0.019 vs R003's LDA on the 13-column train+test block (0.8779, p_corr 0.007): for LDA the extra borderline columns hurt, unlike for logistic regression with weak penalty. LDA (shared covariance) is the clearest illustration that the class signal is not in the means. (sup)
