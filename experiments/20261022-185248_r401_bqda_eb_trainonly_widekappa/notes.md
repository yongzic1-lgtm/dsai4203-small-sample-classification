# 20261022-185248_r401_bqda_eb_trainonly_widekappa

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Written before running. R255's EB chose kappa0 = 1.0, the largest grid value, in 100/100 folds (for the hv version R256 moved it to 3 in 95/100 when widened). Prediction: with the wide grid EB picks kappa0 = 3-10 in most folds, and CV AUC stays within 2 split-SE of r255_bqda_eb_trainonly (0.99903), i.e. 0.9988-0.9993. Refuted if EB stays at 1.0 or the AUC moves by more than 2 split-SE.

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.99919 +/- 0.00082 (24 of 100 folds below 1.0) vs r255_bqda_eb_trainonly 0.99903 +/- 0.00114: paired +0.00016, split SE ~0.0001, corrected p 0.76, fold win rate 0.53 -> within split noise. With the wide grid EB picks kappa0 = 10 in 90/100 folds and 3 in 10/100 (default grid: 1.0 = grid edge in 100/100; 20261022-185320_r401_inductive_official_checks), so the R255 trainonly result was a grid-edge choice but the AUC does not depend on it; hypothesis confirmed. On the r402 replicas both BQDA-trainonly variants rank below r201 (see 20261022-*_r404_inductive_pooled). (r2_inductive)
