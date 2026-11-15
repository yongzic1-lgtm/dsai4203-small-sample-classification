# 20261022-071252_r201_qda_blockratio

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R003: QDA(reg 0.1) on the train+test block = 0.9958. Prediction: nested reg on the in-fold block gives 0.994-0.997 (within 0.002 of R003). A smoke check showed that R001's ratio-1.5 rule on 200 rows returns a superset (16-23 columns, always containing the 13); prediction: the extra noise columns cost 0.001-0.005, so qda_blockratio (ratio tuned in the inner CV) and qda_block_tt are slightly better than qda_block; PCA basis within split noise; extra in-fold |corr| features change AUC by < 0.002; QDA on all 150 features or on corr-top-k is clearly worse (< 0.97) because covariance estimation in high dimension with 200 rows is poor. Refuted if nested QDA on the block < 0.99.

## Result

(see summary.json)

## Conclusion (write after reading the results)

AUC 0.9996 +/- 0.0006: tuning the variance ratio (1.5/1.6/1.7) in the inner CV is slightly WORSE than the fixed a-priori rule 1.5 (-0.0003, split SE 0.0001, p_corr 0.69): the 3-10 borderline columns that the 1.5 rule adds help a little, or at least do not hurt. Within the data-sampling noise. (sup)
