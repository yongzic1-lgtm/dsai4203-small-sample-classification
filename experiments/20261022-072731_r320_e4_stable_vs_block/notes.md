# 20261022-072731_r320_e4_stable_vs_block

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The bootstrap-stable |corr| set is NOT the set worth keeping: it is mostly a subset of the label-free high-variance block (plus feature 17), and the block features that are never stable carry class signal through their variance (covariance), not their mean, so QDA / the transductive GMM on the block beat the same models on the stable set, while L2 logreg is indifferent or prefers the stable set.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(ext34) Numbers valid, figure superseded: re-run 20261022-074624_r320_e4_stable_vs_block has identical numbers (deterministic) plus an in_block_span column and a readable figure; cite the re-run.
