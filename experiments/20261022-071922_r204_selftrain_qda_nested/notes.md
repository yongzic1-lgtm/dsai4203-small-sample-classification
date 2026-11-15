# 20261022-071922_r204_selftrain_qda_nested

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Self-training adds confidently pseudo-labelled test rows; with a QDA base the pseudo-labels in the confident regions are almost always right, so the refitted QDA moves towards the mixture solution. Prediction: SelfTraining-QDA >= supervised QDA (0.996) and close to TransductiveGMM (1.000), 0.997-1.000. With a linear base (LR) self-training cannot fix the wrong model family: 0.85-0.92.

## Result

(see summary.json)

## Conclusion (write after reading the results)

SelfTraining(QDA) on 20,000 rows, nested reg x threshold: 0.9995 +/- 0.0002, log-loss 0.041; reg chosen 0.03 = bottom of a grid fixed before R201 (confound), threshold 0.99/0.9. -0.0004 vs strict supervised QDA (p_corr 0.47): the semi-supervised baseline does NOT beat the tuned supervised model. Prediction (>= QDA) refuted. Full-grid recheck in R209. (sup)
