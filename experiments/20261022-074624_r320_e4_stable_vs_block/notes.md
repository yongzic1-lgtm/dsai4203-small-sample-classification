# 20261022-074624_r320_e4_stable_vs_block

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The bootstrap-stable |corr| set is NOT the set worth keeping: it is mostly a subset of the label-free high-variance block (plus feature 17), and the block features that are never stable carry class signal through their variance (covariance), not their mean, so QDA / the transductive GMM on the block beat the same models on the stable set, while L2 logreg is indifferent or prefers the stable set.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(ext34, 2026-10-22) The top-10 stable set {17, 46, 49, 77, 80, 98, 100, 112} is 7 block features plus feature 17, which is an exact linear combination of the block (label-free R^2 = 1 - 3e-14; so are 6, 52, 57, 61, 142, 144: 20 features span the 12-dim block subspace = R001's 8 zero eigenvalues). Six block features (8, 15, 26, 30, 64, 67) are never stable, yet they carry class signal in the variance (label-free GMM components: d ~ 0, log var ratio up to -1.0/+0.83) and alone give QDA 0.917 / GMM 0.932. Honest CV: block > stable for L2 logreg (0.862 vs 0.832, p = 0.018), QDA (0.996 vs 0.961, p = 0.0006) and GMM (1.000 vs 0.983, p = 0.009); block | stable = block. Stable != worth keeping for a covariance signal.
