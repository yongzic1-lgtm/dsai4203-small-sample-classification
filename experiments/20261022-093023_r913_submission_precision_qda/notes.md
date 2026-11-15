# 20261022-093023_r913_submission_precision_qda

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

As for the GMM (R912): saturated or rounded QDA scores only tie rows whose posterior is within 5e-9 of 0 or 1; prediction: mean private AUC(csv8) - AUC(raw) within +-2e-5 and |mean| < 2 SE; AUC(logodds) - AUC(raw) may be slightly positive if the QDA log-odds order the confident rows informatively (it is the sharper model), but < 1e-4. Refuted if csv8 - raw < -1e-4 on average.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(repro, 2026-10-22) Hypothesis confirmed for QDA as well. Without label noise AUC(csv8) - AUC(raw) is -1.8e-10 (reg 0.003) and +8.9e-10 (reg 0.1) on average, log-odds - raw exactly 0. With flip_y = 0.005: csv8 - raw = +5.1e-6 +- 2.4e-5 (reg 0.003, range -3.2e-4 .. +3.1e-4) and +3.6e-6 +- 4.2e-6 (reg 0.1); log-odds - raw = -1.9e-5 +- 1.7e-5 and -5e-7 +- 4e-7 (all |t| < 1.6), although 6,013 test rows are exactly 0 or 1 in float64 at reg 0.003 (largest csv tie 7,921). The %.8f format is harmless in expectation; it adds per-file noise of at most ~3e-4 on the private AUC.
