# 20261022-073222_r253_hs_quadhv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

With squares/products of the block the sparse model can express the class covariance difference: predicted CV AUC 0.97-0.995, i.e. close to but below QDA on the block (0.996).

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.9673 +/- 0.0054 with 150 linear + 91 block squares/products: +0.120 vs L1, but -0.028 vs QDA(reg 0.1) (p 0.0015). Hypothesis (0.97-0.995) narrowly refuted: a sparse logistic model on second-order terms is a less efficient description of the covariance difference than the Gaussian class model (QDA/Bayes-QDA) with 200 rows.
