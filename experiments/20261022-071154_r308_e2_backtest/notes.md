# 20261022-071154_r308_e2_backtest

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Fitted on the small sizes only, the E2 power law under-covers the larger observed sizes: its 90 % bootstrap interval misses the held-out means for the L1 baseline (the curve shape at small n is dominated by collapsed L1 fits), while the envelope of curve forms covers them; prediction errors grow with the extrapolation ratio, so the 2.5x extrapolation to 500 rows is less reliable still.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Hypothesis refuted for the power law: fitted on sizes 50-150 (x <= 120) it predicts the held-out means at x = 160 / 200 within +0.003 / +0.006 (L1) and -0.002 / -0.003 (QDA), every 90% interval covering; the log-linear form overshoots the L1 curve by +0.022 / +0.039 and misses 3 of 6 targets. The recomputed subset means equal the E2 runs (0.8301, 0.8474; 0.9950, 0.9960). For the 2.5x extrapolation to 500 rows the power law is the better-supported form; the envelope's upper end comes from log-linear. Parent-sample noise is still not covered. (ext12)
