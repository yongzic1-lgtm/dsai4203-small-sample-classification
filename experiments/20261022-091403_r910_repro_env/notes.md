# 20261022-091403_r910_repro_env

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

BLAS thread count does not change any result (limits=1 vs default: max |diff| < 1e-12, identical fold AUCs); every pinned package is installed at its pin.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) Confirmed: every package of colab/requirements-colab.txt is installed at its pin (25/25, python 3.11.17, OpenBLAS 0.3.30/0.3.31 Haswell kernels on 'Intel Xeon @ 2.20GHz'); limiting BLAS to 1 thread and joblib to n_jobs=1 changes nothing (all three models bit-identical), so thread count is not a source of nondeterminism on this VM type; the CPU kernel is (Windows laptop: GMM test_pred differs by up to 4.4e-11).
