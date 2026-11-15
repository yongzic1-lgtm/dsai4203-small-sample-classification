# 20261022-064436_r102_ref_s003

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Reproduces the S003 test predictions exactly (max |diff| < 1e-9); gives its CV log-loss/Brier.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed: reproduces S003 exactly (max |diff| 6e-11). CV AUC 1.0000, CV log-loss 0.0171, Brier 0.00268, 1 training row on the wrong side (row 9, posterior 0.79). Reference for all r102/r105 runs; replicas (R103) put it 1.7e-7 test AUC below the Bayes ceiling.
