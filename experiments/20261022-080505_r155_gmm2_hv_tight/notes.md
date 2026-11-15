# 20261022-080505_r155_gmm2_hv_tight

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Same CV AUC as S003 (saturated 1.0); converged EM gives the MLE posterior; on parametric replicas (R153) the converged/informative-column mixture is at least as good as gmm2_hv on the private test, so this is a safe alternative submission. It differs from S003 only in the order of a few boundary rows.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Official CV AUC 1.0000 on all 100 folds (saturated, as predicted). EM converged (tol 1e-8) in 8-10 iterations vs 6 for sklearn's default tol in S003; log-likelihood of the mixture is the converged MLE. Caveat: the summary fields "test_boundary_0.01_0.99", "spearman_boundary_vs_s003_equiv" and "pair_order_changes_boundary" use the Beta-smoothed SCORE (positive-component rate ~0.987), so the "boundary" set contains every positive-cluster row and these three numbers are not meaningful; the correct posterior comparison is R150 posterior_agreement (max posterior difference 0.013, 1 side change, Spearman 0.99998 on rows with 0.01<p<0.99). Decision: equivalent to S003, not worth a submission slot as an improvement: on 100 parametric replicas (R153_param) the converged / informative-column mixtures have the same expected private AUC as gmm2_hv within 1e-8.
