# 20261022-075059_r257_hs_identifiability

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

beta is not identified along the 8 exact linear dependencies, so chains disagree on beta (R-hat >> 1) but agree on the linear predictor (R-hat(eta) < 1.1); on a QR-reduced identifiable design the chains mix (R-hat(beta) < 1.1) and CV AUC is unchanged within 2 split-SE.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed. Label-free null space of the 20,000 x 150 matrix has dimension 8; it involves all 13 block columns and 7 outside features (6, 17, 52, 57, 61, 142, 144), i.e. those 7 are (near-)exact linear combinations of block features (the 250 train rows are numerically full rank only because of CSV rounding). In the full 150-column fit one chain used 57/98/144 where the others used 46/30/100/142: R-hat(beta) max 4.6 but R-hat of the linear predictor eta max 1.17 (median 1.016). After QR column pivoting (labels unused; 5 columns dropped: 26, 49, 67, 77, 98) all chains agree: R-hat(beta) max 1.003, ESS >= 712, 0 divergences. Clearly included (1 - kappa > 0.9, 90% CI excludes 0): 46, 6, 80, 30, 142, 100; next 138, 117, 8. Features 6 and 142 lie outside the variance block but are linear combinations of it.
