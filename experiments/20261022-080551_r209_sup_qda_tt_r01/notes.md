# 20261022-080551_r209_sup_qda_tt_r01

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 (the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Recomputation of R003 sup_qda_hv (QDA reg 0.1, train+test block): 0.9958 +/- 0.0011, identical to R003 (0.99578): the pipeline is reproducible across sessions. Base of pair B (partner: R204 selftrain_qda_default, +0.0009). (sup)
