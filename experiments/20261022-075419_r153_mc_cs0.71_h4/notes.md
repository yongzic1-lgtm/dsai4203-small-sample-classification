# 20261022-075419_r153_mc_cs0.71_h4

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

(i) Bayes AUC on the private test is ~0.9999-1.0 for flip_y=0 and drops by about flip_y/2 per unit label noise (~0.995 at flip_y=0.01); (ii) gmm2_hv is within ~1e-4 of Bayes, gmm2_all and gmm4_hv lose more, qda_hv ~0.99, l1_all ~0.85; (iii) at AUC ~0.999 the public-LB sd is ~5e-4-1e-3 and transductive variants swap or tie on the public LB in most draws; (iv) adding the informative columns outside the block does not change the private AUC when the block already spans the informative space (as in the real data).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Generalises R153_param with one important exception. On 100 make_classification replicas (class_sep 0.71, hamming 4, new A_k/B/scales each) the Bayes private AUC is 0.99998 (flip 0), 0.99750 (0.005), 0.99497 (0.01). The high-variance block spans the informative space in 95% of replicas; there gmm2_hv equals the structured-column / true-informative mixtures within 3e-7. In the other 5% (the variance rule misses an informative direction) gmm2_hv loses up to 0.007 private AUC and the label-free structured-column mixture gmm2_struct_pca recovers +0.0026 on average (better in 5/5), reaching Bayes within 2e-7. So "all linearly structured columns, PCA to rank" is the robust column rule; on the real data it changes nothing because the block already has rank 12 (R150). gmm2_all is close to Bayes here (-1.6e-5) and better than gmm2_hv in the non-spanning replicas; gmm4_hv is again slightly worse (-3.4e-5). Public-LB sd: 0.0001 (flip 0), 0.0027 (0.005), 0.0038 (0.01); ABC for S003's public score again favours flip_y > 0 (posterior 0.20 / 0.50 / 0.30 for 0 / 0.005 / 0.01), predictive private AUC of S003 0.9970 (90%: 0.9944-0.9993).
