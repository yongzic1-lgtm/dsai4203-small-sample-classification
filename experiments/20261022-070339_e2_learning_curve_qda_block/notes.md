# 20261022-070339_e2_learning_curve_qda_block

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

Supervised QDA (reg_param 0.1) on the 13-column high-variance block (columns and standardisation from train+test features, labels only from the subset) needs many labels to estimate two 13x13 covariance matrices: its CV AUC is clearly below the 250-row value 0.996 (R003) at n = 50 (about 18 rows per class per training fold) and rises steeply, so the curve is still rising at n = 250 but close to the ceiling 1; the extrapolated AUC at 500 training rows is above 0.996 and the predicted gain is small in absolute terms (< 0.005) because of the ceiling.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Real data, QDA reg 0.1 on the block: CV mean 0.973/0.990/0.992/0.995/0.996, std 0.022/0.005/0.003/0.0014/0.0005. AUC(500) power law 0.9969 [0.9952, 0.9986], envelope [0.9952, 1.0] (log-linear exceeds 1); gain 250->500 +0.0008. Partly refutes the hypothesis: QDA is already at 0.973 with 40 training rows and flat from ~100 rows, so it is NOT data-hungry on this problem; the predicted gain < 0.005 was right. (ext12)
