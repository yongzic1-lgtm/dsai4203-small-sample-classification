# 20261022-072003_r204_selftrain_lr

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Self-training adds confidently pseudo-labelled test rows; with a QDA base the pseudo-labels in the confident regions are almost always right, so the refitted QDA moves towards the mixture solution. Prediction: SelfTraining-QDA >= supervised QDA (0.996) and close to TransductiveGMM (1.000), 0.997-1.000. With a linear base (LR) self-training cannot fix the wrong model family: 0.85-0.92.

## Result

(see summary.json)

## Conclusion (write after reading the results)

SelfTraining(L2 logistic): 0.8777 +/- 0.0074 (log-loss 1.58: confident wrong pseudo-labels), below the supervised L2 logistic on the same block (R003 0.8977). Self-training cannot repair the wrong model family and can hurt. As predicted (0.85-0.92). (sup)
