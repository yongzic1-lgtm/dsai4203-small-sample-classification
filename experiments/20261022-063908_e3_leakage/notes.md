# 20261022-063908_e3_leakage

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

Selecting the top-k features by |corr| on all 250 rows before CV (leaky) gives a higher CV AUC than redoing the selection inside every fold (honest). The gap is 0 at k_feat = 150, small for tiny k_feat if the strongest features are real, and largest at intermediate k_feat. To first order it does not depend on k_fold. With permuted labels honest CV is ~0.5 while leaky CV stays clearly above 0.5. Real data (written before the full run, ext34 2026-10-22): R001 found 7 features with |corr| 0.34-0.40 (98, 17, 100, 112, 80, 77, 46) and the rest <= 0.24, so the gap should be near 0 for k_feat <= 7 (the strong features are found inside every fold too) and largest for k_feat ~ 20-50, where most kept features are noise picked for their chance correlation with the validation labels; the gap should be flat in k_fold.


## Result

(see summary.json)

## Conclusion (write after reading the results)

(ext34, 2026-10-22) Leakage confirmed and sized: at k_feat=20 / 5-fold the leaky CV is 0.867 +/- 0.008 vs honest 0.793 +/- 0.014 (gap 0.073 +/- 0.014, corrected p = 0.0003); the gap is ~0 at k_feat=7 (the 7 strong |corr| features are re-selected in every fold), grows to 0.090 at k_feat=30-50 and is exactly 0 at 150; under permuted labels honest stays at 0.49-0.51 while leaky reaches 0.71 (k_feat=30). The gap shrinks with k_fold on the real data (k_feat=20: 0.087 / 0.082 / 0.073 / 0.063 for 2/3/5/10 folds, because the honest in-fold selection improves) but not under the null (0.201 / 0.206 / 0.210 for 3/5/10). Leaky CV would also pick the wrong k_feat (50 instead of 7). Public LB of test_pred.npy (k_feat=20) pending one submission; code = bundle logs/code_bundles/b413e6aa3bae4443.tar.gz (job ran with --no-sync because of a transient remote.py bug).

Update 2026-10-22 (fix round 2, AI): LB comparison done - S004 public 0.79136 (uploaded 2026-10-22 16:08 JST, registry/submissions.csv; pre-registered B005: expected 0.82, 90% 0.785-0.854); leaky - public +0.075 (3.2 LB SE), honest - public +0.002 (reports/tables/e3_leaderboard.csv); the ± 0.014 above is the SD over 20 split seeds, the corrected SE is 0.020 (p 0.0003; cells.csv corrected_t 3.72).
