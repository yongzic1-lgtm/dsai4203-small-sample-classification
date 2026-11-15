# 20261022-172258_r933_replicas_param_s136-143_tn0_r8_cv

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Test-label noise only (train labels clean, as on the real data). Predictions (written before running): the supervised nested QDAs lose ~2e-4 (r210) to ~3e-4 (r201) private AUC to S003 at every noise level (the clean-label gap of the s0-7 run), not the 4e-4 to 1.3e-3 seen with train-label noise; BQDA ~1.4e-4 and self-training ~7e-5 below; the GMM family within 1e-6 of S003 in expected AUC. With the CSV's 8-decimal rounding the GMM family ties S003 exactly in most replicas, so the r102 'lottery' hedge (r932: E[max(0, diff)] = 3.9e-5 on raw scores) shrinks to < 1e-5; P(beat S003) < 0.5 for every candidate.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Cross-VM determinism check, not new replicas (do not pool: its seeds duplicate s136-246). Seeds 136-143 re-run on pro2/vcand2 are bit-identical to the same seeds of 20261022-165947_r933_replicas_param_s136-246_tn0_r8_cv (run on vcand): priv_auc, priv_auc_r8, pub_auc_r8, cv_auc, extra and real_cv_auc all np.array_equal. So the difference in replica CV fold counts between seed ranges 136-246 and 247-357 is sampling variation, not a VM effect. (cand)
