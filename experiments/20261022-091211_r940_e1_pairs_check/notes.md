# 20261022-091211_r940_e1_pairs_check

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

E1's MDD for 5x20 (0.0061) was measured on a pair whose true CV gap is ~0.004 and whose models are highly correlated (corr of repeat means 0.76). Expectation: (i) E1's corr30/corr50 numbers reproduce exactly and the pooled 1000-repeat gap is about -0.0044, not -0.010 or -0.003; (ii) less similar linear pairs have a larger split-seed std of A - B, so for some pairs the 5x20 MDD (2 sd) approaches 0.01, and with the 80%-power definition (2.8 sd) 5x20 no longer resolves 0.01 with its whole CI.

## Result

(see summary.json)

## Conclusion (write after reading the results)

E1 reproduces exactly (corr30 vs corr50: 5x1 / 5x10 / 5x20 sd 0.014857 / 0.004165 / 0.003056, mean -0.004428, official seed -0.010063); the pooled 1000-repeat CV gap of E1's pair is -0.0044 +- 0.0004 (neither -0.010 nor -0.003). Over all 28 linear pairs the 5x20 sd of A - B is 0.0018-0.0046 (E1's pair 0.0031, about the 35th percentile): 25/28 pairs keep the whole 95% CI of MDD = 2 sd below 0.01 (the 3 failures all involve L1 C=0.05), but with the 80%-power MDD = 2.8 sd only 4/28 do; 5x10 resolves 10/28 pairs at 2 sd, while its p_correct(0.01) is >= 0.93 for every pair. Hypothesis confirmed: E1's "only 5x20 resolves 0.01" holds for most linear pairs under its own 2-sd (50%-power) definition, not under an 80%-power definition. (verifier ext)
