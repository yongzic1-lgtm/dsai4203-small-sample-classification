# 20261022-073058_r153_param

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

(i) Bayes AUC on the private test is ~0.9999-1.0 for flip_y=0 and drops by about flip_y/2 per unit label noise (~0.995 at flip_y=0.01); (ii) gmm2_hv is within ~1e-4 of Bayes, gmm2_all and gmm4_hv lose more, qda_hv ~0.99, l1_all ~0.85; (iii) at AUC ~0.999 the public-LB sd is ~5e-4-1e-3 and transductive variants swap or tie on the public LB in most draws; (iv) adding the informative columns outside the block does not change the private AUC when the block already spans the informative space (as in the real data).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Mostly supported (100 parametric replicas, 4 label-noise levels). (i) Bayes AUC on the 19,355-row private part: 0.99995 +- 0.00001 without label noise, 0.99869 / 0.99742 / 0.99494 at flip_y 0.0025 / 0.005 / 0.01 (about 1 - flip_y/2: label noise, not class overlap, sets the ceiling). (ii) gmm2_hv (= S003) is Bayes-optimal to within 3e-7 (flip 0); gmm2_hv_tight / _pca / struct_pca / inf_oracle are identical to it within 1e-8; gmm2_all (-3.4e-5, worse in 100% of replicas) and gmm4_hv (-3.5e-5, worse in 94%) lose a little; qda_hv 0.9931 (-0.0068), l1_all 0.860. CV AUC (5x4) is 0.99997 for all transductive variants, so CV cannot rank them. (iii) Public LB (395 rows): sd of the public AUC 0.00009 at flip 0 (52% of draws exactly 1.0) but 0.0019-0.0039 at flip 0.0025-0.01; transductive K=2 variants tie on the public LB in 60-100% of draws; gmm2_hv vs gmm2_all: tie 42%, wrong order 12% (flip 0) up to 47% (flip 0.01) although gmm2_hv wins on the private set in 64-100%. (iv) Adding the 7 structured columns outside the block (or using the true informative columns) changes the private AUC by <= 1.8e-7 at flip 0 (mean 1e-8) because the block already spans the informative space (100% of replicas, as in the real data). ABC: a public score within +-0.00015 of S003's 0.99923 has probability 0.0025 at flip 0 vs 0.017-0.022 at flip 0.0025-0.01, so label noise is likely present; posterior predictive of S003's private AUC (uniform prior over the 4 flip levels) mean 0.9972, 90% interval 0.9944-0.9991.
