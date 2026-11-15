# 20261022-065046_r152_calibration

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The label-free identification recovers the informative/redundant/noise roles exactly in >= 90% of make_classification replicas with class_sep >= 0.6; the class_sep estimator is nearly unbiased with sd ~0.05 at hamming 4, so the real data's class_sep is about 0.7 (between 0.6 and 0.8).

## Result

(see summary.json)

## Conclusion (write after reading the results)

INVALID: failed before any result (joblib/loky could not unpickle the scipy.special.expit ufunc referenced by the worker defined in __main__; BrokenProcessPool). Fixed by computing the boundary fraction without expit; superseded by the next r152_calibration run.
