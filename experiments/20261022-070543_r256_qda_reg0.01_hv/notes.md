# 20261022-070543_r256_qda_reg0.01_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Classical QDA's AUC depends on reg_param; if a smaller reg already reaches the Bayesian QDA AUC, the gain is mostly shrinkage strength/target, not Bayesian averaging.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Classical QDA with reg_param 0.01: 0.99976 +/- 0.00033 (+0.0040 vs reg 0.1, split-SE 0.0002). The R003 QDA (reg 0.1) was over-regularised towards the identity; a lightly regularised plug-in QDA already nearly closes the gap to the GMM. Sensitivity grid only (not tuned on these folds).
