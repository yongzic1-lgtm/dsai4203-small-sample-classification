# 20261022-072014_r204_labelprop_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Graph label spreading on a kNN graph of the 20,000 block rows exploits cluster structure but not the global Gaussian shape. Prediction: 0.95-0.995 on the block, nested >= default; on all 150 features the graph is dominated by noise: < 0.80. All below TransductiveGMM.

## Result

(see summary.json)

## Conclusion (write after reading the results)

LabelPropagation(knn 7): 0.9313 +/- 0.0039 (1000 iterations, not converged to tol). Same conclusion. (sup)
