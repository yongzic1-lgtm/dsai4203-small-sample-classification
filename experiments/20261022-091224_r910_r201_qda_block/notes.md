# 20261022-091224_r910_r201_qda_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Today's code (bundle hash in meta.json) reproduces the archived run bit for bit: all model seeds are fixed (GMM random_state 0, liblinear random_state 0, inner CV seed 0, QDA deterministic), the package pins are identical (numpy 2.4.6, scipy 1.17.1, sklearn 1.9.1) and the computational code is unchanged (only plot titles / an unused `scoring` argument differ from the archived bundles). Prediction: every fold AUC identical; test_pred max |diff| < 1e-12, Spearman 1.0. Possible source of a difference: OpenBLAS kernel / thread count of another VM type (summation order) -> diffs ~1e-15 in probabilities; for the nested QDA a flipped inner-CV tie could change a chosen reg_param in a fold (visible in fold_fit_info.csv). Refuted if any fold AUC differs by > 1e-9 or test Spearman < 0.999999.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) Hypothesis confirmed: 20261022-071234_r201_qda_block reproduces bit for bit on a 2-core VM (archived: 8-core, bundle f84e098deaf7a1bf; today lib_sup.py differs only by an unused `scoring` default and plotting code): all 100 fold AUCs (0.999903), oof, test_pred (sha16 0d7e9b64d238586a), and per fold the chosen reg_param, the in-fold block and the best inner AUC are identical.
