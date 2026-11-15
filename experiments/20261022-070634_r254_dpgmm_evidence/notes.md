# 20261022-070634_r254_dpgmm_evidence

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

With a DP prior the variational mixture keeps about 2 effective components for small gamma (one per class) and splits a class into more components only for large gamma; its plug-in BIC is not better than the finite K=2 GMM, and its test ranking agrees with S003 on > 90% of the ~200 boundary rows (sides of 0.5). On replica data with known labels the transductive GMM K=2 beats the supervised Bayesian QDA (250 labels) on the GMM's boundary rows, and averaging the two does not beat the GMM.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Label-free evidence: the variational DP mixture keeps exactly 2 effective components (weights 0.499/0.499, all others < 0.001) for every gamma in 0.001..100, truncation 10 or 20, DP or finite-Dirichlet prior, raw block or PCA(12) basis. Plug-in BIC (PCA12) 343,919-343,925 vs finite GMM K=2 343,911 (K=1 465,604; K=3..6 344,762-347,261): K=2 is the supported structure. Component label rates (all 250 labels): 113/114 and 0/135. Full-train test scores agree with S003 on 98.5-99.5% of the 204 boundary rows (Spearman 0.999). Replica simulation (3 replicas per generator; G1 = fitted GMM K=2, G2 = fitted DP mixture): test AUC GMM K=2 0.99488 / DP 0.99479 / supervised BayesQDA 0.99476 / oracle 0.99514 (G1), similar under G2; on the GMM's ~200 boundary rows GMM 0.825, DP 0.824, BayesQDA 0.702, GMM+BQDA mean 0.795. Hypotheses confirmed: the DP prior adds nothing over K=2, and the transductive GMM ranks the boundary rows far better than the 250-label Bayesian QDA (averaging hurts). Caveat: replicas are drawn from Gaussian mixtures, which favours the mixture models.
