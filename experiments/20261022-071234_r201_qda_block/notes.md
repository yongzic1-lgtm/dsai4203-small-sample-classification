# 20261022-071234_r201_qda_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R003: QDA(reg 0.1) on the train+test block = 0.9958. Prediction: nested reg on the in-fold block gives 0.994-0.997 (within 0.002 of R003). A smoke check showed that R001's ratio-1.5 rule on 200 rows returns a superset (16-23 columns, always containing the 13); prediction: the extra noise columns cost 0.001-0.005, so qda_blockratio (ratio tuned in the inner CV) and qda_block_tt are slightly better than qda_block; PCA basis within split noise; extra in-fold |corr| features change AUC by < 0.002; QDA on all 150 features or on corr-top-k is clearly worse (< 0.97) because covariance estimation in high dimension with 200 rows is poor. Refuted if nested QDA on the block < 0.99.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.9999 +/- 0.0002, OOF log-loss 0.0146. Strictly supervised (in-fold block of 19.1 columns on average, no test features), nested reg_param chose 0.001-0.01 in 100/100 folds (0.003: 49, 0.001: 41, 0.01: 10; 0.001 is the grid bottom, but R201 qda_pcablock with reg 0 available made the same choices). +0.0041 vs R003's untuned QDA reg 0.1 (split SE 0.0002, win rate 0.79, p_corr 0.12) and -0.0001 vs TransductiveGMM (p_corr 0.71): the hypothesis (0.994-0.997) is REFUTED upwards: R003's supervised baseline was over-regularised, and 250 labels alone nearly saturate CV AUC. Best strictly supervised model of the track: the 'honest supervised' candidate. (sup)
