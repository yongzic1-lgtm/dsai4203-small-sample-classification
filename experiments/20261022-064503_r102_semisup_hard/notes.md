# 20261022-064503_r102_semisup_hard

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

As semisup_eps01; hard constraints force boundary training rows, still negligible.

## Result

(see summary.json)

## Conclusion (write after reading the results)

As semisup_eps01 (Kendall 0.9995, log-loss 0.0077). R103: ~0 with clean labels, -2.3e-7 with label noise (hard constraints trust wrong labels). Rejected in favour of soft constraints.
