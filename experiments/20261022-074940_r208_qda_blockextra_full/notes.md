# 20261022-074940_r208_qda_blockextra_full

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201: nested QDA on the in-fold block = 0.9999 with reg 0.001-0.01; the ratio-tuned variant (fewer columns) was slightly lower (0.9996), suggesting that the 3-10 borderline columns carry signal. Predictions: with the full reg grid, block+extra is within split noise of qda_block (the in-fold block already contains the useful borderline columns); corr-top-k stays clearly worse (< 0.99: univariate |corr| misses variance-only signal); QDA-LW on all 150 features < 0.95; tuning-free QDA-LW on the block 0.99-0.999 (LW over-shrinks a little); the exact 13-column in-fold block (ratio 1.65) is slightly below qda_block (0.9990-0.9998).

## Result

(see summary.json)

## Conclusion (write after reading the results)

With the full reg grid, in-fold block + k extra |corr| columns: 0.99987 +/- 0.00022, identical to qda_block (diff -0.00003, p_corr 0.78); the inner CV mostly picks extra_k = 0 (60/100 folds) or 1. Extra correlated columns add nothing once the borderline variance columns are in. R201 qda_blockextra's lower score was a grid artefact. Prediction confirmed. (sup)
