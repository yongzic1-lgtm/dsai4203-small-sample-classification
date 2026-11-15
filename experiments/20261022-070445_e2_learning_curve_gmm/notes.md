# 20261022-070445_e2_learning_curve_gmm

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

The mixture is fitted without labels, so the labelled rows only have to tell which of the two components is the positive class. With 40 training rows (n = 50) that mapping is already certain, so the CV AUC equals 1.0 at every size with zero spread, the learning curve is flat and the extrapolated AUC at 500 rows is 1.0: unlabeled structure moves the label requirement from hundreds of rows (supervised QDA / L1) to a handful.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Real data, TransductiveGMM K=2 block: CV and out-of-subset AUC = 1.0000 in all 50 subsets at every size (std 0). Hypothesis confirmed: with the mixture fitted on unlabeled features, 40 labelled rows already saturate; AUC(500) = 1.0 trivially. (ext12)
