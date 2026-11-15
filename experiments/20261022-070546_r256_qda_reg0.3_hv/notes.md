# 20261022-070546_r256_qda_reg0.3_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Classical QDA's AUC depends on reg_param; if a smaller reg already reaches the Bayesian QDA AUC, the gain is mostly shrinkage strength/target, not Bayesian averaging.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Classical QDA reg_param 0.3: 0.98317 +/- 0.00255 (-0.0126 vs reg 0.1, corrected p 0.02). Strong identity shrinkage destroys the covariance signal.
