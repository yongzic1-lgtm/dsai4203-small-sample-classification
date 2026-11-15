# 20261022-070254_r200_l2_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.9317 +/- 0.0074, C = 30 (grid top) in 100/100 folds -> grid edge, rechecked in R207. Same as L1 block (diff -0.0004, split SE 0.0004). Still 0.06 below QDA (R003 0.9958): the covariance signal is out of reach of a linear rule. (sup)

Addendum after R207 (20261022-074147_r207_l2_block_tt_wide): L2 logistic on the exact 13-column train+test block with C tuned up to 1e4 reaches 0.9383, so the gain over R003 comes from R003's too-strong penalty (C = 1), not from the borderline columns; the R200 grid edge was harmless (R207 wide grid: 0.9318 / 0.9319). (sup)
