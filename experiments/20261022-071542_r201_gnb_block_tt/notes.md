# 20261022-071542_r201_gnb_block_tt

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Gaussian naive Bayes keeps class-specific variances but drops correlations. Prediction: between LDA (0.878, means only) and QDA (0.996, full covariance): 0.92-0.98 on the block; worse on all features.

## Result

(see summary.json)

## Conclusion (write after reading the results)

GNB on the 13-column train+test block: 0.8721 +/- 0.0052, same conclusion as gnb_block. (sup)
