# 20261022-070539_r256_bqda_eb_widekappa_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

A wider kappa0 grid moves the EB optimum beyond 1 but changes CV AUC / log-loss by less than 2 split-SE (the result does not hinge on the grid edge).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Wider kappa0 grid (to 300): EB moves to kappa0=3 (95/100 folds, all-250 fit too), AUC 1.0000 unchanged, log-loss 0.0102 (vs 0.0094). Hypothesis confirmed: the R255 result does not hinge on the grid edge.
