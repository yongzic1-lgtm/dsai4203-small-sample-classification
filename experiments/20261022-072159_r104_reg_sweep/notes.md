# 20261022-072159_r104_reg_sweep

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

In the PCA/whitened bases the log-likelihood and the boundary ranking stop changing below reg 1e-6 (Kendall vs S003 > 0.99 for reg <= 1e-6), i.e. S003's reg is already effectively the MLE.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed with a nuance. The smallest within-component covariance eigenvalue in the PCA basis is 1.1e-4, so S003's reg_covar 1e-6 is ~1 % of it: the log-likelihood still improves slightly down to reg 1e-8 (BIC 343,911.05 -> 343,910.64) and the ranking converges only for reg <= 1e-8 (PCA) or <= 1e-6 (whitened basis; all such fits Kendall 1.000 with each other). The unregularised MLE re-orders the S003 boundary rows marginally (Kendall 0.9959 on S003's 204 boundary rows, i.e. ~0.2 % of pairs) and leaves the expected AUC under the reference unchanged (0.999948). In the 13-column standardised basis (S003's own, with an exact null direction) the BIC is meaningless (it diverges as reg -> 0 along the null direction) but the ranking converges the same way. Decision: the reg -> 0 MLE (PCA basis reg 1e-9) is the principled refinement of S003; test it on replicas (R103 block_polish_reg_small) and as an official-CV run (R105).
