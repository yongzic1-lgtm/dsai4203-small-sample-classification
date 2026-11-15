# 20261022-070518_r307_e2_model_comparison

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Labels needed for the same AUC differ by orders of magnitude between the three models: the L1 baseline is still rising at 200 training rows, supervised QDA approaches the ceiling only near 200 rows, and the transductive GMM is flat at 1.0 from the smallest size.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Comparison of the three E2 curves on identical subsets: reports/figures/r307_e2_model_comparison.png, reports/tables/r307_e2_model_comparison.md. Labels needed for AUC >= 0.97: GMM <= 40, QDA ~40, L1 not reached by 200 (and extrapolated ~0.91 at 500). (ext12)
