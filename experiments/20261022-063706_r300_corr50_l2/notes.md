# 20261022-063706_r300_corr50_l2

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV 0.8011 +/- 0.0156. Model B of the E1 main pair (gap to top-30 = 0.0101, closest of all linear pairs to 0.01). Not a submission candidate. (ext12)
