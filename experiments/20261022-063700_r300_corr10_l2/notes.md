# 20261022-063700_r300_corr10_l2

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.8119 +/- 0.0146 (repeat sd). Best of the in-fold corr-filter + L2 family; beats top-20 by +0.018 (se_split 0.0035, corrected p 0.37). Supervised, far below QDA/GMM: not a submission candidate. (ext12)
