# 20261022-070200_r255_bqda_eb_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A Bayesian (NIW, Student-t predictive) QDA averages over the class covariances instead of plugging in noisy estimates from ~90 rows per class; predicted: CV AUC above QDA(reg 0.1) 0.9958 by more than 2 split-SE, but still below the transductive GMM (1.0000); refuted if not above QDA.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 1.0000 +/- 0.0000 on all 100 folds (supervised, 250 labels; block columns/scaling label-free as in R003), CV log-loss 0.0094 (GMM S003 config 0.0171). Paired vs QDA(reg 0.1): +0.0042, split-SE 0.00025, win rate 0.79, corrected p 0.12 (> 2 split-SE but not significant in the conservative data-sampling view). Hypothesis 'above QDA' confirmed in the split-noise view; 'below GMM' refuted: CV cannot separate it from the transductive GMM. EB chose kappa0=1 (grid edge), nu0=d+5, pooled prior in all folds; R256 shows the gain is the shrinkage target/strength, not the Student-t predictive. On test: pseudo-AUC vs S003 non-boundary labels 0.99998, but only 73.5% side agreement on the 204 S003 boundary rows.
