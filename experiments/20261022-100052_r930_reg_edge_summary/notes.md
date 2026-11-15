# 20261022-100052_r930_reg_edge_summary

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

GMM fold AUC is 1.0 in 100/100 folds; every supervised QDA variant is <= GMM fold by fold, so the paired mean is <= 0 with a one-sided sign; the Nadeau-Bengio corrected test is non-significant for all (|t| small because the corrected SE is inflated by n_val/n_train = 0.25); the 95% interval of the paired mean excludes differences larger than ~5e-4, i.e. CV can only bound the gap, not show equality.

## Result

(see summary.json)

## Conclusion (write after reading the results)

GMM: 1.0 in 100/100 folds. Every strictly supervised QDA variant is <= GMM in every fold (never better): 1 fold worse for reg in [1e-4, 3e-3] (diff -1.6e-5, corrected p 0.84), 3 folds for reg <= 3e-5, 4 folds for AUC tuning (-9.7e-5, sign test p 0.125). The 95% intervals with the Nadeau-Bengio corrected SE are [-1.8e-4, +1.5e-4] (log-loss tuned) and [-6.1e-4, +4.2e-4] (AUC tuned): CV can only bound the gap to a few 1e-4; it cannot show equality. OOF log-loss profile: 0.0137 (3e-3), 0.0087 (1e-3), 0.0071 (3e-4), 0.0074 (1e-4), 0.0080 (3e-5), 0.0085 (1e-5): the r210 grid edge (1e-3) was not the log-loss optimum (3e-4), but no reg changes the CV AUC conclusion. Hypothesis confirmed. (cand)
