# 20261022-075228_r208_qda_block_lw

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201: nested QDA on the in-fold block = 0.9999 with reg 0.001-0.01; the ratio-tuned variant (fewer columns) was slightly lower (0.9996), suggesting that the 3-10 borderline columns carry signal. Predictions: with the full reg grid, block+extra is within split noise of qda_block (the in-fold block already contains the useful borderline columns); corr-top-k stays clearly worse (< 0.99: univariate |corr| misses variance-only signal); QDA-LW on all 150 features < 0.95; tuning-free QDA-LW on the block 0.99-0.999 (LW over-shrinks a little); the exact 13-column in-fold block (ratio 1.65) is slightly below qda_block (0.9990-0.9998).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Tuning-free QDA-LW on the in-fold block: 0.9863 +/- 0.0047 (log-loss 0.148): analytic Ledoit-Wolf shrinkage over-shrinks relative to the nested reg 0.001-0.01 (-0.0136, p_corr 0.020). Not a good tuning-free substitute here. Prediction (0.99-0.999) refuted slightly. (sup)
