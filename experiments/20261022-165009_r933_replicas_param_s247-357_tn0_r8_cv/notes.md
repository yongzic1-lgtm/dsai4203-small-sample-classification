# 20261022-165009_r933_replicas_param_s247-357_tn0_r8_cv

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Test-label noise only (train labels clean, as on the real data). Predictions (written before running): the supervised nested QDAs lose ~2e-4 (r210) to ~3e-4 (r201) private AUC to S003 at every noise level (the clean-label gap of the s0-7 run), not the 4e-4 to 1.3e-3 seen with train-label noise; BQDA ~1.4e-4 and self-training ~7e-5 below; the GMM family within 1e-6 of S003 in expected AUC. With the CSV's 8-decimal rounding the GMM family ties S003 exactly in most replicas, so the r102 'lottery' hedge (r932: E[max(0, diff)] = 3.9e-5 on raw scores) shrinks to < 1e-5; P(beat S003) < 0.5 for every candidate.

## Result

(see summary.json)

## Conclusion (write after reading the results)

111 replicas, test-label noise only, CSV-formatted scores (pooled verdicts in r936). Same picture as s136-246: R156-weighted private diff vs S003 r105 +1.2e-6, r155 -1.1e-6, r102 +1.3e-6, BQDA -1.20e-4, self-training -7.1e-5, r210 -1.66e-4, r201 -2.50e-4, ext-grid -5.8e-5. Real-data CV recomputed on this VM is again identical to the archived runs. The replica CV fold counts are somewhat higher than in s136-246 (median folds below 1.0: r255 7 vs 3, r210 9 vs 6, r201 14 vs 11; Mann-Whitney p 0.002-0.017, strongly correlated across procedures) although replica difficulty is the same (Bayes clean AUC p 0.91, test boundary rows p 0.95) -> cross-VM determinism re-run of seeds 136-143 on vcand2. (cand)
