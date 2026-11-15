# 20261022-073018_r253_hs_all

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Horseshoe shrinkage separates the few relevant linear features from noise better than L1 on 200 rows: predicted CV AUC above L1 logistic (0.847) by > 2 split-SE but below L2 on the block (0.898).

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.9261 +/- 0.0070 (supervised, all 150 features): vs L1 logistic C=0.1 +0.079 (split-SE 0.0025, win rate 0.99, corrected p 0.0005) and vs L2 logistic on the block +0.028 (p 0.028): the best LINEAR model so far, a clear win by the team's rule. Hypothesis 'above L1' confirmed, 'below the block L2 (0.898)' refuted: horseshoe shrinkage recovers linear signal outside the block (consistent with R002). Still far below QDA/BQDA (-0.070): the main signal is quadratic. Test pseudo-AUC vs S003 labels 0.934.
