# 20261022-070545_r256_qda_reg0.03_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Classical QDA's AUC depends on reg_param; if a smaller reg already reaches the Bayesian QDA AUC, the gain is mostly shrinkage strength/target, not Bayesian averaging.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Classical QDA reg_param 0.03: 0.99895 +/- 0.00063. Monotone in reg: 0.01 > 0.03 > 0.1 (0.9958) > 0.3 (0.9832).
