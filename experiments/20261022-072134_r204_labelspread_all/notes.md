# 20261022-072134_r204_labelspread_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Graph label spreading on a kNN graph of the 20,000 block rows exploits cluster structure but not the global Gaussian shape. Prediction: 0.95-0.995 on the block, nested >= default; on all 150 features the graph is dominated by noise: < 0.80. All below TransductiveGMM.

## Result

(see summary.json)

## Conclusion (write after reading the results)

LabelSpreading on all 150 features: 0.6225 +/- 0.0201: a kNN graph built in 150 dimensions is dominated by the 137 noise columns. Strong negative result: semi-supervised learning needs the right representation. (sup)
