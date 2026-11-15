# 20261022-185320_r401_inductive_official_checks

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. (1) Default-grid EB of r255_bqda_eb_trainonly picks kappa0 = 1.0 (grid edge) in 100/100 folds as archived; the wide grid moves it to 3-10. (2) The re-implementation in r400_inductive_lib reproduces the archived test_pred of r201_qda_block and r255_bqda_eb_trainonly bit-identically, and the official ensemble run's test_pred equals their average to < 1e-15.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(1) Confirmed: default-grid EB picks kappa0 = 1.0 (largest grid value) in 100/100 real folds; the wide grid picks 10 (90 folds) or 3 (10). In-fold block size 16-23 columns. (2) Partly refuted: recomputed in this run's main process (multi-threaded BLAS), r201 and r255_trainonly are NOT bit-identical to the archived test_pred (sha16 fe0bb283 / 929c63d6 vs 0d7e9b64 / 412c1f9f). In single-thread loky workers (r402 real-data pass, 20261022-182709_r402_inductive_replicas_s136-246) r201 is bit-identical and r255_trainonly differs by at most 7.8e-16 with 0 CSV values changed (20261022-194238_r404_inductive_pooled, real_check.csv), so the difference is BLAS threading, not code. The official ensemble run equals the member average to 1.3e-15, 0 CSV values differ. (r2_inductive)
