# 20261022-071633_r201_svm_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

RBF-SVM can approximate the quadratic boundary. Prediction: block 0.97-0.995 (<= QDA), all features < 0.90 (RBF distances dominated by 137 noise features).

## Result

(see summary.json)

## Conclusion (write after reading the results)

RBF-SVM on all 150 features: 0.8052 +/- 0.0108. Negative result (noise columns dominate the RBF distance), as predicted. (sup)
