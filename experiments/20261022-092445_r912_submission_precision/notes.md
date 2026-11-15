# 20261022-092445_r912_submission_precision

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The %.8f rounding and the float64 saturation only tie rows whose GMM posterior is < 1e-8 from 0 or 1, i.e. rows far from the decision boundary. A wrongly ordered pair inside such a tie group needs a label that disagrees with an almost certain posterior: with flip_y = 0 essentially never (Bayes posterior < 1e-8), with label noise the flipped labels are independent of x, so ordered or tied they earn 1/2 in expectation. Prediction: mean private AUC(csv8) - AUC(raw) and AUC(logodds) - AUC(raw) are within +-2e-5 (|mean| < 2 x its SE) at both noise levels; refuted if either difference is systematically negative by more than 1e-4 (then make_submission should write ranks or %.17g).

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) Hypothesis confirmed. Without label noise the %.8f file and the unsaturated log-odds give exactly the same private AUC as the float64 score in all 30 replicas (difference 0.0). With flip_y = 0.005 (mean 50 flipped labels) AUC(csv8) - AUC(raw) = +5.1e-6 +- 9.5e-6 (SE; range -1.3e-4 .. +1.5e-4, 13 of 30 negative) and AUC(logodds) - AUC(raw) = -2.1e-6 +- 3.1e-5: the ties (on average 8,450 + 8,560 rows in the two extreme groups, Spearman(file, raw) 0.974, as on the real S003 file) cost nothing in expectation and move a single private score by at most ~1.5e-4. Side observation (not this lens): on these generator-level replicas the S003 GMM is within 1e-6 of Bayes in 27/30 seeds but misses by 6e-4 to 7e-3 in 3/30 (seeds 14, 19, 28; highvar block 11-17 columns).
