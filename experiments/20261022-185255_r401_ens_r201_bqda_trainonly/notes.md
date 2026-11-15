# 20261022-185255_r401_ens_r201_bqda_trainonly

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Written before running. Averaging r201 (CV 0.99990) and r255_bqda_eb_trainonly (0.99903) gives a CV AUC between the two and within 2 split-SE of r201, about 0.9996-0.9999. Its test_pred equals 0.5 * (r201 + r255_trainonly) of the archived runs up to floating-point rounding (< 1e-15), because the members are fitted exactly as in those runs.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.99981 +/- 0.00024 (9 folds below 1.0) vs r201_qda_block 0.99990 (4 folds below 1.0): paired -0.0001, split SE ~0.0001, corrected p 0.77, fold win rate 0.48 -> within split noise; hypothesis (between the members, within 2 split-SE of r201) confirmed. test_pred equals 0.5 x (recomputed r201 + recomputed r255_trainonly) to 1.3e-15, no CSV value differs (r401_inductive_official_checks). Strictly inductive (no test feature used). Replica verdict in r404. (r2_inductive)
