# 20261022-074421_r203_rf_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Axis-aligned splits approximate a quadratic (covariance) boundary in 13 correlated dimensions badly with 200 rows, and on all features the 137 noise columns dilute the splits. Prediction: block 0.80-0.93, all features 0.70-0.88, all well below QDA (0.996); boosting not better than RF; the nested LightGBM grid gains < 0.02 over the fixed one. Expected negative result.

## Result

(see summary.json)

## Conclusion (write after reading the results)

RandomForest (500, min_leaf 3, fixed a priori) on all 150 features: 0.8659 +/- 0.0070. Expected negative result (prediction 0.70-0.88). (sup)
