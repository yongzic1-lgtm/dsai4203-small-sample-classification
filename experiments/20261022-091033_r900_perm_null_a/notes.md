# 20261022-091033_r900_perm_null_a

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Null A (train and evaluate on permuted labels, official scheme): every honest pipeline (s003_gmm2_hv, r155_gmm2_inf_tight, r201_qda_block, r210_qda_block_nll, r209_selftrain_qda_lw, r255_bqda_eb_hv, e3_honest) has a mean AUC over permutations within 0.5 +/- 0.03, no permutation above 0.6, and an across-permutation SD comparable to e3_honest (<= ~0.05); the positive control ctrl_e3_leaky stays above 0.6 in every permutation (E3's own null: 0.705 at k=20, 5-fold); ctrl_disk_reader is ~0.5 here (Null A cannot see path b). The true-label rows reproduce the archived CV AUCs within 1e-4.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Failed run (no result): the positive-control class DiskReader used an lru_cache-wrapped module-level helper that loky workers could not unpickle (BrokenProcessPool after the true-label reference rows). Fixed in research/r900_perm_null.py (train.csv read inside fit) and re-run as job J20261022-091532-pro2-vleak; use that run, not this one.
