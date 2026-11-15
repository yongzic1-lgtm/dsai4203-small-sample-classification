# 20261022-073433_r253_hs_inclusion

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The horseshoe posterior concentrates its inclusion mass on a few block features and on the target-correlated features outside the block found in R002; the L1 solution selects a superset with many noise features.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Full-data fits (4 chains x 1000/1000). hs_quadhv mixed (R-hat 1.02, ESS 218, 0 divergences): top terms 6, 100, 49x64, 46, 80, 67x80, 30, 80x98, 8x8 (squares/products of block features carry the class signal). hs_all did NOT mix (R-hat 4.6, ESS 2, 0 divergences): per-feature inclusion for the linear model is not reliable; the top terms (6, 80, 46, 30, 142, 100, ...) are only indicative. Follow-up R257 tests whether the 8 exact linear dependencies cause it. L1 (C=0.1, all rows) selects 22 features, 6 in the block.
