# 20261022-092146_r922_label_noise_private

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

(a) Wrong-label counts drive the public score: with the tied S003 ranking a single confidently wrong label costs ~0.003 public AUC, while clean-label boundary rows cost ~1e-4, so 0.99923 (about 30 discordant pairs) is unlikely under flip_y = 0 but the likelihood is nearly flat for flip_y in ~0.002-0.012 and only falls slowly above; (b) the train labels add a factor ~exp(-125 f); (c) the posterior mean of flip_y depends strongly on the prior (gen's 4-point grid truncated at 0.01 understates the uncertainty); the private AUC is ~1 - f/2 with sd ~3-6e-4, so a 90 % predictive interval is roughly 0.994-0.9995 and B004 (private >= 0.999) has probability well below 70 % (expected 15-35 %).

## Result

(see summary.json)

## Conclusion (write after reading the results)

The S003 scores were reproduced bit-exactly (hash c70057e06406a2d6). With them and the real test features, P(public rounds to 0.99923 | k wrong public labels) = 1.24e-4 / 2.44e-3 / 4.1e-4 for k = 0/1/2, so the combined likelihood (with the 250 train labels, 0 confident contradictions) peaks at flip_y = 0.003 and is >= 1/8 of the max on [0, 0.0147]: LR(flip 0) = 0.20 (0.03-0.84 under posterior temperature 1.25-0.8), LR(0.01) = 0.37, LR(0.02) = 0.034. Under a flat prior on [0, 0.1] flip_y has posterior mean 0.0063 (90 % 0.001-0.015) and S003's private AUC has predictive mean 0.9968, 90 % [0.9923, 0.9996]; gen's 4-point grid reproduces their 0.9978. P(B004: private >= 0.999) is 0.14-0.17 under the uniform, 4-point and default-aware priors (0.51 only under a log-uniform prior), far below the registered 70 %; "flip_y = 0 nearly impossible" and "~0.2 % wrong labels" are overstated / too narrow.
