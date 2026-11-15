# 20261022-194238_r404_inductive_pooled

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running (after the r402 job logs printed per-run quick summaries for flip 0/0.005/0.01 only). (a) Strictly inductive candidates vs r201 (CSV scores): ext-grid QDA +1.5e-4 to +2.2e-4, r210 +6e-5 to +1.2e-4, BQDA-trainonly and its wide-kappa variant below r201, QDA-LW far below (-3e-3 or worse), the r201+BQDA-trainonly average within +-1.5e-4 of r201; the ranking is the same under the R156 and the flat-R922 weights, and every difference is flat across flip_y. Best inductive pair: ext-grid QDA with r210 or r201; the pair adds < 2e-5 over the ext-grid QDA alone. (b) Bayes-boundary rows: S003 within 0.005 of the Bayes posterior's AUC; S003 ranks better than every supervised procedure in > 90% of replicas; the r254 gap (0.825 vs 0.702) is of the right sign but its size differs on the neutral definition. (c) r402 = r933 exactly for the shared procedures and seeds; the real-data pass reproduces the archived test predictions bit-identically (r255_bqda_eb_hv to < 1e-13).

## Result

(see summary.json)

## Conclusion (write after reading the results)

(a) 222 replicas, CSV scores, R156 weights (flat-R922 weights give the same order and differences within 1e-5): vs r201, ext-grid QDA (r930) +1.94e-4 (MC SE 8.8e-6, P(beat) 0.93), r210 +8.8e-5 (7.8e-6, 0.79), r201+BQDA-trainonly average +6e-6 (8e-6, 0.55), BQDA-trainonly -9.6e-5 (2.0e-5, 0.37), wide-kappa -8.7e-5, QDA-LW in-fold -9.8e-3; flat across flip_y 0-0.03. Best inductive pair ext + (r201+BQDA average) +2.03e-4 vs r201 alone, only +8.5e-6 (1.7e-6) over ext alone; ext + r201 +4.8e-6 (1.1e-6). Hypothesis confirmed except that the best partner is the average, not r210/r201 (differences < 5e-6). (b) Bayes-boundary rows (185 per replica): S003 0.804, Bayes 0.806, r209 0.718, ext 0.698, r255_hv 0.678, r210 0.670, r201 0.664; S003 better in 222/222 for each; r254's definition gives S003 0.814 vs r255_hv 0.686 (r254: 0.825 vs 0.702 on 6 replicas), so D009 evidence (2) holds in sign and size. (c) r402 = r933 exactly (56/56); real-data pass bit-identical for 6/8 references, the two BQDA within 7.8e-16 (0 CSV values). (r2_inductive)
