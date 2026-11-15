# 20261022-072454_r202_gpc_rbf_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A GP classifier with an RBF kernel is a flexible smooth boundary; on the 13-d block it can represent the quadratic class boundary but must learn it from 200 rows. Prediction: ARD block 0.985-0.996 (<= QDA 0.996, because QDA has the correct parametric form), isotropic block a bit lower, all 150 features < 0.90 (isotropic distance dominated by noise features). ARD length scales should be shortest on block features with the largest class variance difference.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Isotropic GPC on all 150 features: 0.7855 +/- 0.0141. Negative result, as predicted (noise columns dominate the distance). (sup)
