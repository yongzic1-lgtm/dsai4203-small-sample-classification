# 20261022-091211_r910_s003_gmm2_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Today's code (bundle hash in meta.json) reproduces the archived run bit for bit: all model seeds are fixed (GMM random_state 0, liblinear random_state 0, inner CV seed 0, QDA deterministic), the package pins are identical (numpy 2.4.6, scipy 1.17.1, sklearn 1.9.1) and the computational code is unchanged (only plot titles / an unused `scoring` argument differ from the archived bundles). Prediction: every fold AUC identical; test_pred max |diff| < 1e-12, Spearman 1.0. Possible source of a difference: OpenBLAS kernel / thread count of another VM type (summation order) -> diffs ~1e-15 in probabilities; for the nested QDA a flipped inner-CV tie could change a chosen reg_param in a fold (visible in fold_fit_info.csv). Refuted if any fold AUC differs by > 1e-9 or test Spearman < 0.999999.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) Hypothesis confirmed: on a fresh 2-core Colab VM (bundle df6dea90c123b9f4) today's TransductiveGMM reproduces 20261022-050708_r003_gmm2_hv (S003) bit for bit: all 100 fold AUCs, oof.npy (5,000 values) and test_pred.npy (sha16 71fb85e7168cde88) identical (research/r911_repro_compare.py, run 20261022-091852_r911_repro_compare). The same model on the Windows laptop (local smoke, scratch archive) differs by up to 4.4e-11 in every test value (other OpenBLAS kernel), which changes 1 of 19,750 values of the %.8f submission file.
