# 20261022-080632_r209_selftrain_qda_lw

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 (the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).

## Result

(see summary.json)

## Conclusion (write after reading the results)

SelfTraining(QDA with Ledoit-Wolf shrinkage, threshold 0.99), no tuning: AUC 1.0000, log-loss 0.0071 (best calibrated model of the track, better than TransductiveGMM's 0.0171), test pseudo-AUC vs GMM labels 1.0000. Self-training lifts the over-shrunk LW base (supervised QDA-LW ~0.986-0.996) to the GMM solution: here the unlabeled rows clearly help. Base archived in R210 (sup_qda_tt_lw). (sup)
