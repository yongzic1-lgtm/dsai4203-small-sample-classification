# 20261022-070542_r256_bqda_weak_plugin_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Near-ML QDA (weak prior, plug-in) is clearly worse than the same prior with the Student-t predictive (r255_bqda_weak_hv 0.9994): the predictive integration matters.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Weak prior + plug-in (almost ML-QDA): 0.99945 vs 0.99943 with the Student-t predictive (r255_bqda_weak_hv). Hypothesis refuted: integration over Sigma does not matter here.
