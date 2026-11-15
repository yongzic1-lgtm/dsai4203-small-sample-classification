# 20261022-074610_r203_xgb_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Axis-aligned splits approximate a quadratic (covariance) boundary in 13 correlated dimensions badly with 200 rows, and on all features the 137 noise columns dilute the splits. Prediction: block 0.80-0.93, all features 0.70-0.88, all well below QDA (0.996); boosting not better than RF; the nested LightGBM grid gains < 0.02 over the fixed one. Expected negative result.

## Result

(see summary.json)

## Conclusion (write after reading the results)

XGBoost (depth 3, fixed a priori) on the in-fold block: 0.9033 +/- 0.0109, same level as RF/ET/LightGBM on the block (0.897-0.908); ~0.097 below strict QDA. Expected negative result. (sup)
