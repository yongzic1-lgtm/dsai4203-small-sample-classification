# 20261101-123515_r201_qda_block

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

R003: QDA(reg 0.1) on the train+test block = 0.9958. Prediction: nested reg on the in-fold block gives 0.994-0.997 (within 0.002 of R003). A smoke check showed that R001's ratio-1.5 rule on 200 rows returns a superset (16-23 columns, always containing the 13); prediction: the extra noise columns cost 0.001-0.005, so qda_blockratio (ratio tuned in the inner CV) and qda_block_tt are slightly better than qda_block; PCA basis within split noise; extra in-fold |corr| features change AUC by < 0.002; QDA on all 150 features or on corr-top-k is clearly worse (< 0.97) because covariance estimation in high dimension with 200 rows is poor. Refuted if nested QDA on the block < 0.99.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(main session, 2026-11-01) Purpose of this run: test the README reproduction route for the second default final S005 (`research/r201_sup_quadratic.py --only qda_block` on Colab, job J20261101-123642-pro1-repro, code bundle 00a20dcb38357b0d, 2-core VM). The hypothesis above is the script's built-in text from the original run; no separate hypothesis was written for this re-run, and the expected outcome (bit-identical) followed from R910 (`20261022-091224_r910_r201_qda_block`). Result: reproduces `20261022-071234_r201_qda_block` bit for bit. All 100 fold AUCs (mean 0.999903, split signature 5708ce5c3828), oof.npy and test_pred.npy are identical (max |diff| 0); per fold the chosen reg_param, the in-fold block (support) and the best inner AUC are identical (only fit_seconds differ). `make_submission.py --run 20261101-123515_r201_qda_block --dry-run` gives sha256 prefix 3be09c0723286457 = registry row S005. No submission file was written and nothing was uploaded.
