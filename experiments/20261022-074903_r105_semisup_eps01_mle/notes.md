# 20261022-074903_r105_semisup_eps01_mle

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

MLE + 250 soft label constraints; indistinguishable from gmm2_block_mle (Kendall > 0.999).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Equivalent to gmm2_block_mle (same Kendall 0.9959 vs S003, expected AUC +1e-11); lower CV log-loss (0.0077) only because it reports the posterior instead of rate-compressed scores. No gain on replicas (R103 semisup_eps01 +1.6e-10).
