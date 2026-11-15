# 20261022-092058_r932_real_repro_agreement

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

(A) r931_cand_lib reproduces every archived candidate's test_pred.npy (max |diff| < 1e-9 for the GMM family, < 1e-6 for the others) and the archived nested reg choices of r201/r210 in 100/100 folds. (B) The 4 GMM-family candidates (S003, r105, r155, r102) order the test rows almost identically (Spearman > 0.99 on the S003 boundary rows); BQDA and self-training differ more (0.7-0.95) and strictly supervised QDA most (0.4-0.6, R249 reported 0.45-0.49). Under a plug-in truth the private-AUC differences vs S003 are of order 1e-5 to 1e-4 for the supervised ones and < 1e-6 within the GMM family, i.e. far smaller than the label-noise spread of the private AUC itself.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(A) Reproduction confirmed: r931_cand_lib reproduces every archived candidate's test_pred.npy (S003, r209, r210, r201 bit-identical; r105 / r102 / r255 max |diff| <= 7.2e-15; r155 4.4e-14), and the nested reg choices of r201 and r210 in 100/100 folds; the fast metric copies give the same choices and test predictions as sklearn's roc_auc_score / log_loss (100/100, max diff 0). With the reg grid extended below r210's edge, inner log-loss picks 1e-4 in 53, 3e-4 in 36 and 0.001 in 11 folds (1e-4 is the floor allowed by sklearn QDA's rank check tol 1e-4 on the rank-12 block), so r210's 100/100 at 0.001 WAS a grid-edge artefact. (B) GMM family (S003, r105, r155, r102) agree on the 204 S003 boundary rows (Spearman >= 0.9999); BQDA 0.55, self-training 0.70, r210 0.49, r201 0.45, ext-grid QDA 0.58 (hypothesis range for BQDA/self-training 0.7-0.95 refuted downwards). Plug-in private AUC (labels ~ Bernoulli((1-f) pi + f/2), f = 0.0044, pi = converged GMM posterior): diff vs S003 r105 -4e-7, r155 -4e-7, r102 -5e-7, BQDA -1.1e-4, self-training -6.1e-5, r210 -2.0e-4, r201 -3.6e-4, ext -9.2e-5; but the plug-in is circular (with BQDA / self-training / QDA scores as the truth those win by +2e-4 to +7e-4), so it only bounds the size of possible differences (~1e-4 to 7e-4), it does not rank them. Not predicted: r102's private AUC differs from S003's with sd 1e-4 at f = 0.0044 although its ranking is the same up to ties (clean labels: sd 2e-8) -- the two scores tie confident rows differently, and mislabelled confident rows then score differently; this gives a free 'lottery' hedge E[max(0, diff)] = 3.9e-5. Replica verdicts: r934. (cand)

Correction (added after r936, 2026-10-22 17:5x JST): the r102 'free lottery hedge' E[max(0, diff)] = 3.9e-5 above was computed on full-precision scores. The submission CSV is written with float_format '%.8f', which merges the confident rows into ties (r102: 14,414 distinct test scores -> ~2,900; S003 7,503 -> 2,897). On CSV-formatted scores the replica hedge value of r102 is 6.8e-6 +- 0.7e-6 (raw scores 3.6e-5), no larger than r105 / r155 (7.9e-6 / 8.0e-6); see 20261022-172146_r936_final_selection. (cand)
