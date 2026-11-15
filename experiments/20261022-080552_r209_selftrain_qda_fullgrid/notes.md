# 20261022-080552_r209_selftrain_qda_fullgrid

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 (the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).

## Result

(see summary.json)

## Conclusion (write after reading the results)

SelfTraining(QDA) nested over reg down to 0.001: 0.9996 +/- 0.0003; the inner CV chose reg 0.01-0.03 (not 0.003), i.e. LARGER than the best fixed value. Explanation tested in R210: inner AUCs saturate at 1.0, GridSearchCV breaks ties toward the first (most regularised) grid entry. (sup)
