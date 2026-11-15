# 20261022-071605_r201_svm_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

RBF-SVM can approximate the quadratic boundary. Prediction: block 0.97-0.995 (<= QDA), all features < 0.90 (RBF distances dominated by 137 noise features).

## Result

(see summary.json)

## Conclusion (write after reading the results)

RBF-SVM on the in-fold block, nested C x gamma (C=100, gamma=0.01 in 87/100 folds: grid corner): 0.9505 +/- 0.0105, -0.049 vs QDA (p_corr 0.002). Hypothesis (0.97-0.995) refuted. (sup)
