# 20261022-165947_r933_replicas_param_s136-246_tn0_r8_cv

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Test-label noise only (train labels clean, as on the real data). Predictions (written before running): the supervised nested QDAs lose ~2e-4 (r210) to ~3e-4 (r201) private AUC to S003 at every noise level (the clean-label gap of the s0-7 run), not the 4e-4 to 1.3e-3 seen with train-label noise; BQDA ~1.4e-4 and self-training ~7e-5 below; the GMM family within 1e-6 of S003 in expected AUC. With the CSV's 8-decimal rounding the GMM family ties S003 exactly in most replicas, so the r102 'lottery' hedge (r932: E[max(0, diff)] = 3.9e-5 on raw scores) shrinks to < 1e-5; P(beat S003) < 0.5 for every candidate.

## Result

(see summary.json)

## Conclusion (write after reading the results)

111 replicas, test-label noise only, CSV-formatted scores (pooled verdicts in r936). Hypothesis confirmed: R156-weighted private diff vs S003 (CSV scores) r105 -3.3e-6, r155 -2.4e-6, r102 -3.2e-6 (tie-breaking noise; clean-label gaps < 6e-9), BQDA -1.12e-4, self-training -6.7e-5, r210 -1.71e-4, r201 -2.64e-4, ext-grid -6.8e-5; the clean-label gaps (-1.19e-4, -6.4e-5, -1.59e-4, -2.49e-4) carry over unchanged under test-label noise. CSV rounding cuts S003's distinct test scores from 7,592 to 2,896 and r102's from 14,414 to 2,902. The real-data CV recomputed on the VM equals the archived fold_scores.csv of all 8 candidates (max |diff| <= 1.1e-16). Replica CV is much less often perfect than the real CV (median folds below 1.0: r255 3, r210 6, r201 11 vs real 0 / 1 / 4). (cand)
