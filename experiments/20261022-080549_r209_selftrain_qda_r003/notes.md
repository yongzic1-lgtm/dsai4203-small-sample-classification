# 20261022-080549_r209_selftrain_qda_r003

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 (the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Pair A: SelfTraining(QDA reg 0.003, threshold 0.99) = 1.0000 vs its supervised base 1.0000 (no AUC difference), log-loss 0.0120 vs 0.0109 (slightly worse): with a well-tuned base, self-training adds nothing. Prediction (< 0.001 AUC change) confirmed. (sup)
