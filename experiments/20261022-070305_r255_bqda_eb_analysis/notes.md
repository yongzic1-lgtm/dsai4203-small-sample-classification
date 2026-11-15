# 20261022-070305_r255_bqda_eb_analysis

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

EB picks a weak-to-moderate prior (nu0 - d - 1 small relative to ~90 rows per class) because the class covariances differ strongly (QDA >> LDA).

## Result

(see summary.json)

## Conclusion (write after reading the results)

EB (type-II ML on the fold's training rows) picked kappa0=1, nu0_extra=4 (nu0=d+5), pooled-covariance prior in 100/100 folds and on all 250 rows: a weak prior (worth ~4 pseudo-observations) centred on the pooled covariance. kappa0=1 was the grid edge; R256 widens it (EB then picks kappa0=3; AUC unchanged). Hypothesis (weak-to-moderate prior) confirmed.
