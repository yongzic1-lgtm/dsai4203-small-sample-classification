# 20261022-091223_r910_e3_final_k20

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Today's code (bundle hash in meta.json) reproduces the archived run bit for bit: all model seeds are fixed (GMM random_state 0, liblinear random_state 0, inner CV seed 0, QDA deterministic), the package pins are identical (numpy 2.4.6, scipy 1.17.1, sklearn 1.9.1) and the computational code is unchanged (only plot titles / an unused `scoring` argument differ from the archived bundles). Prediction: every fold AUC identical; test_pred max |diff| < 1e-12, Spearman 1.0. Possible source of a difference: OpenBLAS kernel / thread count of another VM type (summation order) -> diffs ~1e-15 in probabilities; for the nested QDA a flipped inner-CV tie could change a chosen reg_param in a fold (visible in fold_fit_info.csv). Refuted if any fold AUC differs by > 1e-9 or test Spearman < 0.999999.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) Hypothesis confirmed: the E3 final model (CorrelationTopK(20) + L2 logreg, refit on 250 rows) rebuilt from extensions/e3_leakage/run.py and the archived config fields reproduces 20261022-063908_e3_leakage bit for bit: 100 honest fold AUCs (mean 0.793441), the leaky k20/5-fold cell (0.866781, in run 20261022-091403_r910_repro_env) and test_pred.npy (sha16 eadb9ff18d9c6cec) identical, although the archived run recorded no code bundle (--no-sync) and ran on an 8-core VM.
