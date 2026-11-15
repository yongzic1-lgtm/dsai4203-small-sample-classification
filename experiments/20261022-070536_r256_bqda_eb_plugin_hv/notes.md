# 20261022-070536_r256_bqda_eb_plugin_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

If the R255 gain comes from integrating over Sigma, the plug-in version of the same EB prior loses most of it (AUC closer to QDA 0.996).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Same EB prior, Gaussian plug-in at the posterior means: CV AUC 1.0000, log-loss 0.0095 (Student-t version 0.0094), identical boundary-row behaviour (side agreement with S003 73.5%). Hypothesis refuted: the posterior-predictive integration over Sigma contributes nothing measurable; the gain over QDA(reg 0.1) is the EB-chosen shrinkage (towards the pooled covariance, ~4 pseudo-observations).
