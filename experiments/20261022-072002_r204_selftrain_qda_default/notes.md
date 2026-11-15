# 20261022-072002_r204_selftrain_qda_default

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Self-training adds confidently pseudo-labelled test rows; with a QDA base the pseudo-labels in the confident regions are almost always right, so the refitted QDA moves towards the mixture solution. Prediction: SelfTraining-QDA >= supervised QDA (0.996) and close to TransductiveGMM (1.000), 0.997-1.000. With a linear base (LR) self-training cannot fix the wrong model family: 0.85-0.92.

## Result

(see summary.json)

## Conclusion (write after reading the results)

SelfTraining(QDA reg 0.1, threshold 0.75): 0.9966 +/- 0.0008 vs its own supervised base (R003 sup_qda_hv, same reg and block) 0.9958: +0.0009 (split SE 0.0003, win rate 0.55, p_corr 0.72). Self-training helps an over-regularised base a little, but far less than simply tuning reg (+0.0041). (sup)
