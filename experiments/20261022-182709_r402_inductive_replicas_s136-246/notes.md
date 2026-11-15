# 20261022-182709_r402_inductive_replicas_s136-246

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. Same replicas and random streams as the tn0 group of r936, so S003, r255_hv, r209, r201, r210 and the ext-grid QDA reproduce r933's CSV-score private AUCs exactly at flip 0-0.01. Strictly inductive candidates, private AUC difference vs r201 (CSV scores, R156 weights): ext-grid QDA +1.5e-4 to +2.2e-4 and r210 +6e-5 to +1.2e-4 (r936: +1.94e-4, +8.8e-5); r255_bqda_eb_trainonly (real CV 0.9990 vs r201 0.9999) -2e-4 to -1e-3; the wide-kappa EB variant within 1e-4 of it and choosing kappa0 > 1 in most folds; QDA-LW on the in-fold block (real CV 0.986) -3e-3 to -1.5e-2; the r201+BQDA-trainonly probability average within +-1.5e-4 of r201. Differences are flat across flip_y 0-0.03 (sd of private AUC grows ~linearly in flip_y). Boundary rows (Bayes posterior in (0.1, 0.9), clean labels): the Bayes posterior reaches ~0.83; S003 is within 0.005 of it; r209 and r255_hv lose 0.01-0.05, the nested supervised QDAs 0.03-0.10, and the BQDA-trainonly / QDA-LW more. The r254 gap 0.825 vs 0.702 (6 replicas, GMM-boundary rows) shrinks on the neutral Bayes-boundary definition but keeps its sign in > 90% of replicas.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Half of the 222-replica design (pooled analysis: 20261022-194238_r404_inductive_pooled). Pooled verdict on the predictions above: identical to r933 for the shared procedures/seeds (56/56 checks, max diff 0); vs r201 (CSV, R156) ext-grid QDA +1.94e-4 and r210 +8.8e-5 as predicted; r255_bqda_eb_trainonly only -9.6e-5 (predicted -2e-4 to -1e-3, refuted on the small side); wide-kappa EB picks kappa0 3 or 10 in >99.9% of folds and changes the AUC by < 1e-5; QDA-LW -9.8e-3; the r201+BQDA average +6e-6 (= r201). Boundary rows: Bayes 0.806 (not 0.83), S003 0.804; r209 / r255_hv / supervised QDAs lose 0.09-0.14, larger than predicted (0.01-0.10); S003 ranks better in 222/222 replicas for every procedure. (r2_inductive)
