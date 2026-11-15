# 20261022-091506_r900_perm_null_a

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Null A (train and evaluate on permuted labels, official scheme): every honest pipeline (s003_gmm2_hv, r155_gmm2_inf_tight, r201_qda_block, r210_qda_block_nll, r209_selftrain_qda_lw, r255_bqda_eb_hv, e3_honest) has a mean AUC over permutations within 0.5 +/- 0.03, no permutation above 0.6, and an across-permutation SD comparable to e3_honest (<= ~0.05); the positive control ctrl_e3_leaky stays above 0.6 in every permutation (E3's own null: 0.705 at k=20, 5-fold); ctrl_disk_reader is ~0.5 here (Null A cannot see path b). The true-label rows reproduce the archived CV AUCs within 1e-4.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(leak) Hypothesis confirmed. Under 20 label permutations with the official 5x20 scheme every honest pipeline stays at chance: per-permutation mean AUC s003_gmm2_hv 0.496 +/- 0.041 (max 0.559), r155 0.495 +/- 0.041, r201 0.481 +/- 0.028, r210 0.477 +/- 0.041, r209 0.491 +/- 0.036, r255 0.479 +/- 0.035, e3_honest 0.485 +/- 0.040; no permutation above 0.6 for any of them (the slight shortfall below 0.5 is the usual negative bias of CV under the null, not leakage). The positive control ctrl_e3_leaky stays at 0.697 +/- 0.023 (min 0.641, 20/20 above 0.6), so the test has power against selection-outside-the-fold leakage; ctrl_disk_reader is 0.506 here, i.e. Null A alone cannot detect a pipeline that reads the true labels (see the Null B run). The true-label rows reproduce every archived CV AUC exactly (difference 0).
