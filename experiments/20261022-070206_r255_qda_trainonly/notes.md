# 20261022-070206_r255_qda_trainonly

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A Bayesian (NIW, Student-t predictive) QDA averages over the class covariances instead of plugging in noisy estimates from ~90 rows per class; predicted: CV AUC above QDA(reg 0.1) 0.9958 by more than 2 split-SE, but still below the transductive GMM (1.0000); refuted if not above QDA.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Classical QDA(reg 0.1) with the fold-only block/scaling: 0.9928 +/- 0.0030, slightly below the R003 version (0.9958; the train-only block contains extra near-threshold noise columns). Paired reference for bqda_eb_trainonly (+0.0063 for the Bayesian version).
