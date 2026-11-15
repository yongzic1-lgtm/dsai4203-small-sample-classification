# 20261022-080634_r209_labelspread_widegrid

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 (the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).

## Result

(see summary.json)

## Conclusion (write after reading the results)

LabelSpreading with k up to 120 and alpha up to 0.99: 0.9566 +/- 0.0070 (k = 120, alpha 0.95/0.99 = grid edge again). Still 0.043 below supervised QDA (p_corr 0.002): graph diffusion does not capture the Gaussian/covariance structure. Prediction (< 0.97) confirmed. Not pursued further. (sup)
