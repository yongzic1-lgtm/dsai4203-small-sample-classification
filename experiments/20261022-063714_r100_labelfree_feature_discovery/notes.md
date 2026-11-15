# 20261022-063714_r100_labelfree_feature_discovery

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Label-free conditional likelihood-ratio tests (soft labels from the block GMM) find a few features outside the 13-feature block that are class-dependent given the block; all other features are independent noise (no correlation with anything). Refitting on the enlarged set changes the ranking of the ~200 boundary rows.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Hypothesis only partly confirmed. Label-free structure: exactly 20 of the 150 features are correlated with anything (|r| > 0.05 at n = 20,000; the other 130 have max |r| 0.029 with any feature = independent noise); the 20 span a 12-dim space with 8 exact linear relations (8 zero eigenvalues), i.e. the 7 non-block correlated features (6, 17, 52, 57, 61, 142, 144) are exact linear combinations of the 13-feature block (R^2 = 1.0000; their conditional LRT ~127 is a rounding artefact, flagged `exact_redundant`). The conditional LRT given the block finds NO further class-dependent feature (best non-redundant p = 0.012 among 130 = null), so discovery stops at iteration 0 with S = block; the K=2 full GMM components are Gaussian (Mahalanobis KS p = 0.73 / 0.36, |skew|, |ex. kurtosis| < 0.1), BIC picks K = 2 by > 850. Supervised in-fold marginal selection (top-13 by LRT on fold labels) overlaps the label-free set by only 77 % per fold: it always picks redundant 17 and often 6 and misses 15, 26, 30, which are informative only jointly. Training row 9 (y = 0) has posterior 0.79: a boundary case, not label noise. Decision: no feature-set change; the block already spans the whole informative subspace. Code bundle: b229bfbf58b5900a (job ran with --no-sync because remote.py's prepare() returns None).
