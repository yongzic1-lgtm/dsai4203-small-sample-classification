# 20261022-074108_r207_l1_block_wide

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R200 L2/L1 on the in-fold block reached 0.932 with C at the grid edge. Prediction: the wider grid moves the AUC by < 0.005 (C beyond 30 is almost unpenalised for 16-23 columns and 200 rows). L2 on the exact 13-column train+test block stays near R003's 0.898 (C=1) even with tuned C, i.e. the gain of R200's block models comes from the borderline columns, not from the penalty. Refuted if l2_block_tt_wide > 0.92.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Grid-edge check for L1: 0.9319 vs R200 0.9321, unchanged; C = 10 still the modal choice (50/100). (sup)
