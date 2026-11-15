# 20261022-070202_r255_bqda_eb_trainonly

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A Bayesian (NIW, Student-t predictive) QDA averages over the class covariances instead of plugging in noisy estimates from ~90 rows per class; predicted: CV AUC above QDA(reg 0.1) 0.9958 by more than 2 split-SE, but still below the transductive GMM (1.0000); refuted if not above QDA.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.9990 +/- 0.0011, log-loss 0.041: with NO unlabeled row anywhere (block rule + scaling on the fold's 200 rows; that rule picks 18-20 columns, i.e. the 13-block plus near-threshold features 31, 63, 84, 89, 105, 122) Bayesian QDA still beats QDA(reg 0.1) by +0.0032 (split-SE 0.0004, corrected p 0.29) and trails the GMM by 0.0010 (p 0.46). The fully inductive pipeline nearly closes the gap.
