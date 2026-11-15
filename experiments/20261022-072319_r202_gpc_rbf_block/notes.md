# 20261022-072319_r202_gpc_rbf_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A GP classifier with an RBF kernel is a flexible smooth boundary; on the 13-d block it can represent the quadratic class boundary but must learn it from 200 rows. Prediction: ARD block 0.985-0.996 (<= QDA 0.996, because QDA has the correct parametric form), isotropic block a bit lower, all 150 features < 0.90 (isotropic distance dominated by noise features). ARD length scales should be shortest on block features with the largest class variance difference.

## Result

(see summary.json)

## Conclusion (write after reading the results)

GPC (Laplace), isotropic C*RBF, ML-II hyperparameters, in-fold block: 0.9558 +/- 0.0089, -0.044 vs strict QDA (p_corr 0.004). A generic smooth kernel learns the quadratic boundary poorly from 200 rows. (sup)
