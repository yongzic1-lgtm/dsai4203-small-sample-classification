# 20261022-080546_r209_sup_qda_tt_r003

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 (the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supervised QDA with FIXED reg 0.003 on the train+test block: AUC 1.0000 (all folds), log-loss 0.0109. The reg value was taken from R201's nested choice (informed by earlier CV results on the same rows), so this is not an honest estimate; its nested counterpart R201 qda_block_tt scored 0.9997. Base of pair A. (sup)
