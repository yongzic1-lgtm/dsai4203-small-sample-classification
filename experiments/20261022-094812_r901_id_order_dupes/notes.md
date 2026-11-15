# 20261022-094812_r901_id_order_dupes

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

No leakage channel outside the features: (1) AUC(id, y) and AUC(row position, y) within 0.5 +/- 0.07 (2 SE for 113/137 rows), runs test p > 0.01; (2) S003 test scores unrelated to test id / order (|Spearman| < 0.03, decile chi-square p > 0.01); (3) no exact duplicate rows between train and test and no train row closer to a test row than the 1% quantile of test-to-test NN distances in any space; (4) adversarial validation AUC ~0.5 (train is a random subset), predicted test positive rate within 2 SE of 45.2%; (5) number format independent of y; (6) the block is found by the variance rule on train-only and on test-only rows alike, with a wide gap in std ratio around the 1.5 threshold.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(leak) No leakage channel outside the features. Train ids are 0..249 and test ids 250..19999 (both sorted, train = first 250 rows of the id sequence), yet AUC(id, y) = 0.552 (Mann-Whitney p 0.16), quintile positive rates 0.40-0.52 (chi2 p 0.70), runs test p 0.60, lag-1 p 0.52; S003 test scores vs test id: Spearman -0.0013, decile chi2 p 0.17, runs p 0.27; id vs features max |Spearman| 0.016 (< 4 SE 0.028). No exact duplicate rows in any space; train-to-test NN distances match the test-to-test reference (rows closer than its 1% quantile: 2/3/1 of 250 in all150/block13/informative12, expected 2.5; none closer than its minimum). Adversarial validation AUC 0.499 +/- 0.025 (all 150), so train is a random subset; number format independent of y (row-mean decimals AUC 0.456, p 0.23). The block is reproduced exactly from test-only features (13 columns; train-only 250 rows give 19, a superset), and its columns are not the top |corr| columns (15/30/64 rank 85/44/33 of 150); the 12 informative columns include 61 (|corr| 0.006, rank 139), so neither set was found with labels. The recomputed S003 test prediction equals the archived one to 6e-11.
