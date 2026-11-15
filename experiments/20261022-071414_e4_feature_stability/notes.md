# 20261022-071414_e4_feature_stability

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

A few features enter the bootstrap top-k in nearly every resample, the rest at about the chance rate k/150; the sorted frequency curve drops sharply after them and the drop position does not move with k. Whether those stable features are also worth keeping is decided only by honest CV (selection repeated inside each fold); the naive comparison (stable set chosen on all 250 rows) is biased upward. Real data (written before the full run, ext34 2026-10-22): R001 found 7 features with |corr| 0.34-0.40 (98, 17, 100, 112, 80, 77, 46), the next one 0.24 (49), the rest <= 0.18, so the drop should sit at m* ~ 7 for every k >= 7 (at k = 5 the 7 strong features share 5 places). Six of the seven are in the label-free 13-feature high-variance block, so the stable set should be a strict subset of the block plus feature 17; with L2 logistic regression the stable set alone should beat all 150.


## Result

(see summary.json)

## Conclusion (write after reading the results)

(ext34, 2026-10-22) Sharp drop confirmed: 7 features (98, 17, 100, 112, 80, 77, 46; |corr| 0.34-0.40) are in the top-k in >= 98.6% of the 2000 resamples for k >= 10, feature 49 is intermediate (0.04/0.65/0.79/0.92 for k = 5/10/13/20), the rest <= 0.28 at k = 10; gap rule m* = 7/8/8/8, half 7/8/8/9 (does not move with k >= 8), perm 7/8/9/17 and binom 7/17/22/28 (chance-based rules grow with k; binom is far too lenient, the top noise feature reaches 0.84 at k=10 under permuted labels). The stable set always equals the top-m |corr| set (Jaccard 1). Honest stable set (k=10) beats all 150 with L2 logreg by +0.063 (p = 0.038); naive-honest bias is -0.001 at k=10 but +0.034 at k=5 and +0.069 for the 17-feature perm set at k=20; stability selection never beats plain honest top-m. Follow-up R320 (20261022-074624): the label-free block beats the stable set for logreg, QDA and GMM.
