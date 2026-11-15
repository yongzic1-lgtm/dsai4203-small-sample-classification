# 20261022-094951_r921_bayes_regret_replicas

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

On parametric replicas the paired clean-label regret Bayes - S003 is a few 1e-7 (gen: 3.1e-7 +- 0.75e-7 SE, mix: 1.7e-7); with 400 replicas the MC SE is ~4e-8 for observed AUC and smaller for the expected AUC, so a positive regret of that size is real but ~1000x below the label-noise sd of the private AUC (~5e-4). On generic make_classification replicas the mean regret is larger and heavy-tailed (block misses an informative direction or contains noise columns). With flip_y = 0.01 the paired regret is swamped by label noise.

## Result

(see summary.json)

## Conclusion (write after reading the results)

On 1000 independent parametric replicas the clean-label private regret Bayes - S003 is 2.51e-7 +- 0.22e-7 (SE; t = 11; S003 beats Bayes on the sample in 36 % of replicas) and the label-free expected-AUC regret is 2.57e-7 +- 0.018e-7 (positive in 100 %, 90 % range 1.8-3.6e-7), i.e. ~24 discordant pairs out of 9.4e7; gen's 3.1e-7 +- 0.75e-7 and mix's 1.7e-7 (40 reps) are within ~1 of their own SEs of this. The gap is real (far outside MC noise) but 2000x below the label-noise sd of the private AUC (5.7e-4 at flip 0.01), where the paired regret is -0.3e-5 +- 0.5e-5 (indistinguishable from 0). The 8-decimal rounding adds 5e-11. On generic make_classification replicas (cs 0.71, hamming 4) the mean regret is 3.1e-5 +- 0.8e-5, driven by the 5.5 % where the high-variance block misses an informative direction (5.4e-4); with a spanning block it is 0.9-1.5e-6.
