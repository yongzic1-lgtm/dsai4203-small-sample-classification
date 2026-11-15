# 20261022-063705_r300_corr30_l2

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.7910 +/- 0.0117. Chosen as model A of the E1 main pair by the pre-registered R300 rule: A - B (top-30 - top-50) = -0.0101, se_split 0.0023, corr of repeat means 0.76, corrected p 0.51. (ext12)
