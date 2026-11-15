# 20261022-074857_r105_gmm2_block_mle

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Removes the ~1 % reg bias of S003 in the smallest-variance direction: boundary Kendall vs S003 ~0.996, same expected AUC to 1e-6; replicas (R103) decide whether it is better.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed: the reg -> 0 MLE re-orders S003's boundary rows slightly (Kendall 0.9959, Spearman 0.99992, 0 side flips), CV AUC 1.0, log-loss 0.0171, expected AUC under the reference unchanged (0.999948). R103 replicas (block_polish_reg_small, reg 1e-8 = same ranking per R104): +2.3e-9 +- 0.8e-9 expected AUC vs S003, better in 78 % of replicas - the only refinement with a consistent sign, but ~1e-5 of the private LB's SE. Principled 'final' form of S003; a submission would not be distinguishable from S003.
