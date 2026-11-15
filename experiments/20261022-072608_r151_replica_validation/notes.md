# 20261022-072608_r151_replica_validation

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Parametric replicas reproduce every real summary statistic within their 5-95% band and the R003 model ranking (GMM > QDA >> logreg_hv > LDA > L1) with CV AUCs within ~0.02; make_classification replicas (class_sep 0.71, hamming 4) reproduce the structure counts exactly but show a wider spread of difficulty because A_k, B and the scales are redrawn.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported for the parametric replica, partly for make_classification. sklearn bit-exactness of the captured internals: True for 3 seeds. Parametric replicas (30) reproduce the real data: std profile (q10/median/q90/max all within 5-95%), the correlation eigen-spectrum (14 of the top 15 eigenvalues within 5-95%), 13 high-variance columns of rank 12, 20 structured columns of rank 12, 8 zero eigenvalues, BIC(K=3)-BIC(K=2) per row 0.0496 vs 0.0495 real, boundary fraction 0.96% vs 0.95% real, GMM train purity; and the R003 ranking (gmm2_hv > qda_hv > logreg_hv > lda_hv > l1_all in 97% of replicas) with CV AUCs (5x4) L1 0.855+-0.024 (real 0.849), LDA 0.887+-0.018 (0.875), QDA 0.992+-0.003 (0.995), L2-logreg 0.903+-0.016 (0.895), GMM 0.99997 (1.0). make_classification replicas (class_sep 0.71, hamming 4) match the structure counts exactly but vary strongly in difficulty and in the high-variance block (13.9-35.7 columns, 5-95%), so the R003 order is reproduced in only 57%. Real-data outliers vs parametric: Mardia skewness (real 1.10 < replica 1.13-1.40, i.e. the real classes are if anything closer to Gaussian than the truncation-biased replica statistic) and the max |corr(noise, target)| (0.151, at the 3rd percentile; chance).
