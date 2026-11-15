# 20261022-091852_r911_repro_compare

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

A: every re-run fold AUC, oof and test_pred value equals the archived one (bit-identical, or max |diff| < 1e-12 for the BLAS-heavy GMM). B: S001-S003 files are valid, their sha256 prefixes match the registry and their values are exactly '%.8f' of test_pred.npy. C: make_submission would accept both runs (E3 with a non-official-CV warning, because its summary has no official_cv flag). D: every Colab run recorded numpy 2.4.6 / pandas 3.0.6 / scipy 1.17.1 / sklearn 1.9.1 / lightgbm 4.7.0 / xgboost 3.2.0 = the lock; refuted by any mismatch.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) A, B and D confirmed, C confirmed with one caveat. A: all three re-runs (r910, Colab) are bit-identical to the archived runs (fold AUCs, oof, test_pred, r201 per-fold choices). B: S001-S003 files are valid (columns id,target; 19,750 rows; ids = sample_submission in order; no NaN/inf; sha256 prefixes match; values = '%.8f' of test_pred.npy, max |diff| 5.0e-9), but the %.8f format merges tied rows: S003 has 7,503 distinct test_pred values and 2,897 in the file (8,918 rows tie at the minimum, 6,416 at the maximum; Spearman(file, test_pred) = 0.969), S002 760/718 rows at exactly 0/1. C: make_submission would accept both runs (validate_file: no problems; would be S004); E3 triggers the 'not validated on the official CV' banner because its summary.json has no official_cv flag although its split signature is the official 5708ce5c3828; the r201 file would keep 7,724 distinct values of 18,332 (largest tie 5,157). D: 0 of 168 runs recorded a package version different from requirements-lock.txt / requirements-colab.txt; python 3.11.17 (CPU VMs), 3.11.16 (25 L4 GPU runs), 3.11.5 (laptop); 47 Colab runs have no code bundle hash ('unsynced (--no-sync)', empty env_hash) and 10 pre-D005 runs have no compute block.
