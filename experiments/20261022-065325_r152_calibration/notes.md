# 20261022-065325_r152_calibration

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The label-free identification recovers the informative/redundant/noise roles exactly in >= 90% of make_classification replicas with class_sep >= 0.6; the class_sep estimator is nearly unbiased with sd ~0.05 at hamming 4, so the real data's class_sep is about 0.7 (between 0.6 and 0.8).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported. On 176 make_classification replicas (class_sep 0.5-1.0; same generator seeds across the grid, i.e. common random numbers) the label-free identification recovered the structured set, the rank (12), the informative set and the redundant set in 100% of replicas and the zero-difference set / hamming distance in 95%. The class_sep estimator is unbiased (cs_hat = 0.0015 + 0.9998 cs, residual sd 0.050 at hamming 4), so the real raw estimate 0.709 calibrates to class_sep = 0.71 (95% interval 0.61-0.81; grid posterior mass 0.41 at 0.70, 0.28 at 0.75, 0.20 at 0.65). The real data have more boundary rows (0.95% with 0.1<p<0.9) than a typical make_classification draw at class_sep 0.7 (0.37% +- 0.34%): the overlap depends mostly on the random A_k, so the parametric replica (fitted A_k^T A_k) is the better model of our particular data; make_classification replicas show the population spread.
