# 20261022-063717_r300_qda_r03

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.9832 +/- 0.0026. Model B of the E1 second pair. More shrinkage hurts monotonically (0.3 -> 0.5 -> 0.8: 0.983 / 0.961 / 0.902). (ext12)
