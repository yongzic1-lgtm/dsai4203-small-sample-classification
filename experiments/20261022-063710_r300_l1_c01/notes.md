# 20261022-063710_r300_l1_c01

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.84693 = R003 sup_l1_all (0.84693) exactly: cross-session reproducibility confirmed. C=0.2 is better by +0.026 (corrected p 0.07), so the L1 baseline C=0.1 is over-regularised. (ext12)
