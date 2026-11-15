# 20261022-071544_r201_poly2lr_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A linear logistic model on degree-2 features of the block can represent the QDA boundary; with 104 features and 200 rows it needs strong L2. Prediction: 0.97-0.995, below QDA.

## Result

(see summary.json)

## Conclusion (write after reading the results)

L2 logistic on degree-2 polynomial features of the in-fold block (~190 columns), nested C (0.01-0.3): 0.9303 +/- 0.0143. Hypothesis (0.97-0.995) refuted: a quadratic-feature linear model has the right function class but 200 rows are not enough to fit ~190 coefficients; the 2x(19+190) parameters of QDA are estimated far more efficiently through the Gaussian likelihood. (sup)
