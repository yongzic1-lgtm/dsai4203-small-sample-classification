# 20261022-064606_r150_taxonomy

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

make_classification(n_samples=20000, n_features=150, n_informative=12, n_redundant=8, n_repeated=0, n_clusters_per_class=1, weights=None, hypercube=True, flip_y small) followed by a per-column affine map (noise std 0.5-3, means spread over about +-5). Predictions: 20 structured columns of rank 12 (8 exact dependencies, no duplicates); 130 independent Gaussian noise columns; one Gaussian per class (BIC within class prefers K=1); some informative columns with exactly zero class-mean difference; balanced classes; the 13-column high-variance block spans all 12 informative dimensions.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed on every point: 20 structured columns of rank 12 (informative 6,15,30,46,52,61,64,80,100,112,142,144; redundant 8,17,26,49,57,67,77,98 = exact linear combinations, min R^2 1-4e-14; no repeated columns) and 130 independent Gaussian noise columns with std ~ U(0.5,3) (KS p 0.26) and mean ~ U(-5,5) (KS p 0.87), independent of each other, i.e. make_classification(shift=0, scale=1) followed by x*a+b; one Gaussian per class (BIC best K=2 overall, K=1 within each class), balanced (10,016 vs 9,984), hypercube=True (8 informative columns with zero class-mean difference, max |z| 2.08, the 4 active ones |z| >= 47), raw class_sep estimate 0.709 (hamming 4), implied B within [-1.08, 1.11]. No train label contradicts a confident posterior (flip_y MLE 0, 95% upper 0.015 from train labels). The 13-column block already spans the 12 informative dimensions: the GMM posterior from the 20 structured columns is identical to the block's up to EM tolerance (max posterior difference 0.013 with sklearn's default tol=1e-3, 1 side change). Note: the hypothesis was formed from <30 s local smoke looks at the same data before this run.
