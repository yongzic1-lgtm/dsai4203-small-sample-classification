# 20261022-072038_r204_labelspread_nested

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Graph label spreading on a kNN graph of the 20,000 block rows exploits cluster structure but not the global Gaussian shape. Prediction: 0.95-0.995 on the block, nested >= default; on all 150 features the graph is dominated by noise: < 0.80. All below TransductiveGMM.

## Result

(see summary.json)

## Conclusion (write after reading the results)

LabelSpreading nested k x alpha: 0.9438 +/- 0.0052, chose alpha 0.8 / k 30 (grid corner) in 98/100 folds; wider grid in R209. Still 0.056 below supervised QDA (p_corr < 0.001). (sup)
