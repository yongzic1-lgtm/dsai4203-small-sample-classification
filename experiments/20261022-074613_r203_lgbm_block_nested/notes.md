# 20261022-074613_r203_lgbm_block_nested

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Axis-aligned splits approximate a quadratic (covariance) boundary in 13 correlated dimensions badly with 200 rows, and on all features the 137 noise columns dilute the splits. Prediction: block 0.80-0.93, all features 0.70-0.88, all well below QDA (0.996); boosting not better than RF; the nested LightGBM grid gains < 0.02 over the fixed one. Expected negative result.

## Result

(see summary.json)

## Conclusion (write after reading the results)

LightGBM on the in-fold block with a small nested grid (8 configs; chose 7 leaves, min_child 5, 400 trees = grid corner in 97/100 folds): 0.9136 +/- 0.0113, +0.006 over the fixed LightGBM (split SE 0.0010, p_corr 0.21). Tuning does not rescue trees: prediction (gain < 0.02) confirmed. Trees are the expected negative result of the track. (sup)
