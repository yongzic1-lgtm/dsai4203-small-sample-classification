# 20261022-065605_r154_class_structure

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

One Gaussian per class in the 12-dim informative space; the classes differ mainly in covariance: the best linear projection reaches only ~0.88-0.92 AUC (explaining LDA/logreg/L1 in R003) while the quadratic Bayes rule reaches ~0.9999; a covariance-only version of the fitted model is still near-perfectly separable.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported. In the 12 informative columns the fitted model has one Gaussian per class with very different shapes (generalised eigenvalues of S0^-1 S1 from 0.002 to 26.7); the class means differ only in the 4 active coordinates (unscaled differences -1.38, 1.34, 1.62, 1.34 = 2 x class_sep ~ 1.42; inactive |diff| <= 0.06). Population AUCs from the fitted Gaussians (400k rows per class): quadratic Bayes 0.99995, best linear projection 0.947 (formula 0.9468; LDA direction identical), mean-only model 0.947, covariance-only model 0.9998. So linear models are capped at 0.947 even with infinite data (R003's 0.85-0.90 adds estimation error on 200 rows), and almost all separability is in the covariance difference. Bayes error 0.33%; about 186 of the 19,750 test rows are expected with 0.1 < p < 0.9.
