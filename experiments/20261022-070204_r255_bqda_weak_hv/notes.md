# 20261022-070204_r255_bqda_weak_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A Bayesian (NIW, Student-t predictive) QDA averages over the class covariances instead of plugging in noisy estimates from ~90 rows per class; predicted: CV AUC above QDA(reg 0.1) 0.9958 by more than 2 split-SE, but still below the transductive GMM (1.0000); refuted if not above QDA.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.9994 +/- 0.0005 with a fixed weak isotropic prior: most of the gain over QDA(reg 0.1) (+0.0037) needs no empirical Bayes. R256 (weak prior + plug-in, 0.99945) shows the Student-t predictive adds nothing here: the gain comes from weaker covariance shrinkage than reg_param 0.1.
