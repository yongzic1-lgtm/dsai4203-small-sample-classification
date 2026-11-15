# 20261022-082044_r210_sup_qda_tt_lw

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Predictions: (1) supervised QDA-LW on the train+test block scores ~0.995 (R206 at n = 200: 0.9956), so self-training adds ~+0.004 AUC and lowers log-loss by > 0.05 for this tuning-free base; (2) with inner log-loss the nested reg moves to 0.001-0.003 and the outer AUC is >= the AUC-tuned versions (qda_block 0.9999, selftrain fullgrid 0.9996), with lower outer log-loss.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supervised QDA-LW (tuning-free) on the train+test block: 0.9962 +/- 0.0010, log-loss 0.0817. Pair C: self-training the same base on the 20,000 rows (R209 selftrain_qda_lw) gives 1.0000 / log-loss 0.0071, i.e. +0.0038 AUC and -0.075 log-loss: for a tuning-free (over-shrunk) base the unlabeled rows clearly help. Prediction (~+0.004, log-loss -0.05) confirmed. (sup)
