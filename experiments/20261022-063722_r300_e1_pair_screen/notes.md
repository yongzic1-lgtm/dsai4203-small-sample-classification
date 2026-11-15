# 20261022-063722_r300_e1_pair_screen

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Pre-registered rule applied: main E1 pair = corr top-30 vs top-50 + L2 (|gap| 0.0101), second pair = QDA reg 0.1 vs 0.3 (|gap| 0.0126). Tables: reports/tables/r300_candidates.csv, r300_pair_screen.csv. (ext12)
