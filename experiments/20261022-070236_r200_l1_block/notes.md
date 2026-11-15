# 20261022-070236_r200_l1_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; block >= all features; nested tuning changes R003's untuned values by < 0.015. Refuted if any linear model reaches > 0.93.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.9321 +/- 0.0078 on the in-fold variance block (mean 19.1 columns: the 13 train+test block columns in every fold plus 3-10 borderline columns). C chosen at the grid top (10) in 78/100 folds -> wide-grid recheck in R207. +0.034 vs R003 L2 logistic on the 13-column train+test block (0.8977; split SE 0.0012, p_corr 0.013). This REFUTES the linear hypothesis (> 0.93): either the weaker penalty or the extra borderline columns carry signal (R207 separates the two). (sup)

Addendum after R207 (20261022-074147_r207_l2_block_tt_wide): L2 logistic on the exact 13-column train+test block with C tuned up to 1e4 reaches 0.9383, so the gain over R003 comes from R003's too-strong penalty (C = 1), not from the borderline columns; the R200 grid edge was harmless (R207 wide grid: 0.9318 / 0.9319). (sup)
