# 20261022-063715_r300_qda_r01

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.99578 = R003 sup_qda_hv exactly (reproducibility check). Model A of the E1 second (ceiling-regime) pair: A - B vs reg 0.3 = +0.0126, se_split 0.0005, corrected p 0.02. (ext12)
