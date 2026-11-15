# 20261022-092201_r941_e2_replica_truth

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

If the parametric replica is a faithful population model, its expected test AUC at t = 40..200 training rows matches E2's observed CV means (L1: 0.659, 0.735, 0.810, 0.830, 0.847; QDA: 0.973 ... 0.996) within their SE. Then the replica's expected AUC at t = 500 is the target of E2's extrapolation. Expectation: L1 truth at 500 lies inside E2's envelope [0.857, 0.974] but possibly below the power-law point 0.906 (a fixed-C linear model has a ceiling well below 1, while the fitted asymptote is a = 0.996); the spread of the AUC across fresh 200-row samples is larger than E2's within-sample interval, i.e. E2's intervals are too narrow as population statements.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Calibration: replica truth for L1 C=0.1 at t = 40/80/120/160/200 is 0.636/0.768/0.802/0.828/0.851 vs E2's observed CV means 0.659/0.735/0.810/0.830/0.847 (good for t >= 120); QDA-on-block truth is ~0.003 below E2 at every size (0.9925 vs 0.9960 at 200), so the replica is slightly too hard at the QDA ceiling. L1 truth at 500 training rows = 0.9230 +- 0.0006 (sd across training sets 0.0064): above E2's power-law 90% interval [0.881, 0.916] and above the quoted 0.88-0.92, inside the envelope [0.857, 0.974]; power law -0.017, log-linear +0.022, drop-50 -0.054. E1 pair: population gap corr30 - corr50 = +0.0087 +- 0.0011 at 250 rows (sd across 250-row samples 0.0105), opposite in sign to the real-data CV gap -0.0044. (verifier ext)
