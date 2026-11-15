# 20261022-074525_r203_rf_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Axis-aligned splits approximate a quadratic (covariance) boundary in 13 correlated dimensions badly with 200 rows, and on all features the 137 noise columns dilute the splits. Prediction: block 0.80-0.93, all features 0.70-0.88, all well below QDA (0.996); boosting not better than RF; the nested LightGBM grid gains < 0.02 over the fixed one. Expected negative result.

## Result

(see summary.json)

## Conclusion (write after reading the results)

RandomForest on the in-fold block: 0.8965 +/- 0.0074. Axis-aligned splits approximate the correlation (covariance) boundary poorly from 200 rows. As predicted (0.80-0.93). (sup)
