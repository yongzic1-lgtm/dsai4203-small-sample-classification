# 20261022-072331_r202_gpc_ard_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A GP classifier with an RBF kernel is a flexible smooth boundary; on the 13-d block it can represent the quadratic class boundary but must learn it from 200 rows. Prediction: ARD block 0.985-0.996 (<= QDA 0.996, because QDA has the correct parametric form), isotropic block a bit lower, all 150 features < 0.90 (isotropic distance dominated by noise features). ARD length scales should be shortest on block features with the largest class variance difference.

## Result

(see summary.json)

## Conclusion (write after reading the results)

GPC with ARD length scales: 0.9757 +/- 0.0047 (log-loss 0.258), -0.024 vs strict QDA (split SE 0.0011, p_corr 0.006). ARD switches off ~8 of the 19 in-fold columns (length scales at the 1e4 bound). Better than isotropic, but the hypothesis (0.985-0.996) is refuted: QDA's correct parametric form beats the flexible GP at n = 200. (sup)
