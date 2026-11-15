# 20261022-072008_r204_labelspread_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Graph label spreading on a kNN graph of the 20,000 block rows exploits cluster structure but not the global Gaussian shape. Prediction: 0.95-0.995 on the block, nested >= default; on all 150 features the graph is dominated by noise: < 0.80. All below TransductiveGMM.

## Result

(see summary.json)

## Conclusion (write after reading the results)

LabelSpreading(knn 7, alpha 0.2, sklearn defaults) on 20,000 block rows: 0.9118 +/- 0.0081. Graph diffusion ignores the Gaussian shape; below supervised QDA by ~0.09. Prediction (0.95-0.995) refuted. (sup)
