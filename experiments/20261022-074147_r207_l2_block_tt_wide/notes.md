# 20261022-074147_r207_l2_block_tt_wide

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R200 L2/L1 on the in-fold block reached 0.932 with C at the grid edge. Prediction: the wider grid moves the AUC by < 0.005 (C beyond 30 is almost unpenalised for 16-23 columns and 200 rows). L2 on the exact 13-column train+test block stays near R003's 0.898 (C=1) even with tuned C, i.e. the gain of R200's block models comes from the borderline columns, not from the penalty. Refuted if l2_block_tt_wide > 0.92.

## Result

(see summary.json)

## Conclusion (write after reading the results)

L2 logistic on R003's exact 13-column train+test block with nested C up to 1e4: 0.9383 +/- 0.0068 vs R003's fixed C = 1: 0.8977. Prediction (stays near 0.898) REFUTED: R200's gain over R003 came from R003's too-strong penalty, not from the borderline columns. Linear ceiling of this problem ~0.93-0.94 whatever the column set. (sup)
