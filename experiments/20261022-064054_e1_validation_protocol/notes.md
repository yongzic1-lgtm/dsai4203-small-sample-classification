# 20261022-064054_e1_validation_protocol

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

The spread of a single hold-out AUC (one split of 250 rows) is several times larger than 0.01, so a hold-out cannot rank two models that differ by 0.01. Plain k-fold shrinks the spread as k grows only modestly; only repeated k-fold (5-fold x 10, or the official 5-fold x 20) brings the MDD (2 x std of the paired difference of two close candidates) below 0.01.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Real data, main pair corr top-30 vs top-50 + L2 (50 split seeds per scheme). std of the paired difference: hold-out 20% 0.0287, 5-fold 0.0149, 10-fold 0.0149, 5x10 0.0042, 5x20 0.0031; MDD (2 sd) < 0.01 with its whole 95% CI only for 5x20 (0.0061 [0.0051, 0.0076]); 5x10 is borderline (0.0083 [0.0070, 0.0104]). The true CV gap of this pair is small: reference 5x50 A-B = -0.0029 (SE 0.0014), so the -0.0101 of the official seed (R300) was a 2-sd split-noise draw. Confirms D002 (official 5x20) for 0.01 decisions between similar linear models. (ext12)
