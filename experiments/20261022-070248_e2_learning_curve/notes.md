# 20261022-070248_e2_learning_curve

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

The mean CV AUC rises with n and flattens; the spread across the 10 subsets shrinks with n. Because 250 rows give only a few points on the curve, the extrapolated AUC at 500 training rows has a wide interval, and the predicted gain from 250 to 500 rows is small relative to that interval.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Real data, L1 logistic C=0.1 all features: CV mean 0.659/0.735/0.810/0.830/0.847 (n=50..250), std 0.100/0.040/0.028/0.021/0.006; 24.5% constant (all-zero) folds at n=50. AUC at 500 training rows: power law 0.906 (90% boot [0.881, 0.916]), drop-50 0.869, log-linear 0.945, envelope [0.857, 0.974]; predicted gain 250->500 +0.040 [+0.022, +0.047]. Hypothesis confirmed (still rising, wide interval); the curve-form spread is 3x the bootstrap interval. (ext12)
