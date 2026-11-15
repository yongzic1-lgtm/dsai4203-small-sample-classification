# 20261022-081314_r156_flip_posterior

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The train labels alone weakly favour flip_y = 0 (no confident contradiction), the S003 public score favours flip_y > 0; combined, flip_y is most likely 0.0025-0.005 and S003's expected private AUC is ~0.997-0.998.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported. Train labels alone: posterior 0.39 / 0.29 / 0.21 / 0.11 for flip_y = 0 / 0.0025 / 0.005 / 0.01 (no confident contradiction among 250 rows); S003's public score alone: 0.04 / 0.30 / 0.36 / 0.29 (R153_param ABC). Combined: 0.08 / 0.41 / 0.36 / 0.16, posterior mean flip_y 0.0044, P(flip_y > 0) = 0.92. Expected private AUC of S003 (and of any Bayes-equivalent submission) 0.9977, i.e. about 0.0015 below its public score: label noise, not the model, sets the private ceiling. Caveats: coarse 4-point grid with a uniform prior; the ABC uses the parametric replica (fitted overlap) and a +-0.00015 window.
