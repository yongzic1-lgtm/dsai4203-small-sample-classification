# 20261022-075225_r208_qda_all_lw

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201: nested QDA on the in-fold block = 0.9999 with reg 0.001-0.01; the ratio-tuned variant (fewer columns) was slightly lower (0.9996), suggesting that the 3-10 borderline columns carry signal. Predictions: with the full reg grid, block+extra is within split noise of qda_block (the in-fold block already contains the useful borderline columns); corr-top-k stays clearly worse (< 0.99: univariate |corr| misses variance-only signal); QDA-LW on all 150 features < 0.95; tuning-free QDA-LW on the block 0.99-0.999 (LW over-shrinks a little); the exact 13-column in-fold block (ratio 1.65) is slightly below qda_block (0.9990-0.9998).

## Result

(see summary.json)

## Conclusion (write after reading the results)

QDA with per-class Ledoit-Wolf shrinkage on all 150 features: 0.8753 +/- 0.0152 (-0.125 vs block QDA). Column selection is essential for QDA at n = 200. Prediction (< 0.95) confirmed. (sup)
