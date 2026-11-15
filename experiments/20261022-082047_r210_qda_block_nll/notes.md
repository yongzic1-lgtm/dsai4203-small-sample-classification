# 20261022-082047_r210_qda_block_nll

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Predictions: (1) supervised QDA-LW on the train+test block scores ~0.995 (R206 at n = 200: 0.9956), so self-training adds ~+0.004 AUC and lowers log-loss by > 0.05 for this tuning-free base; (2) with inner log-loss the nested reg moves to 0.001-0.003 and the outer AUC is >= the AUC-tuned versions (qda_block 0.9999, selftrain fullgrid 0.9996), with lower outer log-loss.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Strict supervised QDA on the in-fold block with reg chosen by inner LOG-LOSS: reg 0.001 in 100/100 folds (grid bottom), outer AUC 1.0000 +/- 0.0001, log-loss 0.0087 (AUC-tuned R201 qda_block: 0.9999 / 0.0146). Tie-break hypothesis confirmed: with saturated inner AUCs, AUC-based nested tuning keeps the first (most regularised) of many tied configurations; a proper scoring rule resolves them. Caveat: this inner criterion was chosen after seeing R201/R209, so it is a second-level choice (counted in the track's configuration total). (sup)
