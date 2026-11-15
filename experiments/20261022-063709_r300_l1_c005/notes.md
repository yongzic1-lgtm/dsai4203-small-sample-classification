# 20261022-063709_r300_l1_c005

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.8024 +/- 0.0132: C=0.05 over-regularises (-0.045 vs C=0.1, corrected p 0.003). (ext12)
