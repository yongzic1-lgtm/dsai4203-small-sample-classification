# 20261022-064201_r101_bic_model_choice

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

BIC on the 20,000 rows picks K = 2 full covariance with negligible regularisation; tied/diag/spherical are far worse (the classes differ in covariance). The joint 142-dim comparison prefers 'block mixture x independent noise' over a 142-dim mixture. Restarts of K = 2 all reach the same optimum, so restart ensembles cannot change the ranking of boundary rows; the expected-AUC proxy differences between K=2 variants are < 1e-4.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed. BIC on the 20,000 rows (block, 12-dim PCA basis) picks K = 2 full, reg 1e-6 (343,911; K=3 +851, K=4 +1,697; tied/diag/spherical worse by > 118,000), and the regularisation is NOT negligible: reg 1e-4 costs +1,724 BIC and already re-orders the S003 boundary rows (Kendall 0.77), reg 1e-2 Kendall 0.33. Joint 142-dim comparison: "block mixture x independent Gaussians on the 130 isolated features" (BIC 7.725e6) beats a 142-dim K=2 mixture (7.904e6) and a full-covariance noise block (7.800e6), so the gmm2_all-type model is rejected label-free. 30 restarts of K=2 reach the same optimum (log-lik range 1e-9; restart ensemble = reference, Kendall 1.0); K=3/4 restarts differ (multiple optima). Expected test AUC if labels were Bernoulli(reference posterior): 0.999948 for the reference AND for S003 (identical to 6 decimals): under the fitted model the whole ranking headroom is 5e-5 AUC, and the observed public loss of S003 (1 - 0.99923 = 7.7e-4) is 15x larger than the model's own Bayes error, which points to label noise rather than boundary-row ordering. Code bundle: latest sync before the job (see logs/code_bundles, 06:42 JST).
