# 20261022-082102_r210_selftrain_qda_nll

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Predictions: (1) supervised QDA-LW on the train+test block scores ~0.995 (R206 at n = 200: 0.9956), so self-training adds ~+0.004 AUC and lowers log-loss by > 0.05 for this tuning-free base; (2) with inner log-loss the nested reg moves to 0.001-0.003 and the outer AUC is >= the AUC-tuned versions (qda_block 0.9999, selftrain fullgrid 0.9996), with lower outer log-loss.

## Result

(see summary.json)

## Conclusion (write after reading the results)

SelfTraining(QDA) nested by inner log-loss: reg 0.001 in 100/100 folds, AUC 1.0000, log-loss 0.0071 (AUC-tuned R209 fullgrid: 0.9996 / 0.0287). Same tie-break effect as for the supervised QDA. (sup)
