# 20261022-212507_final_s003_gmm2_hv

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Reproduction of S003 (run 20261022-050708_r003_gmm2_hv) through scripts/run_experiment.py: official CV AUC 1.0 in all 100 folds (split signature 5708ce5c3828) and a test_pred that is bit-identical on Colab, so make_submission writes a file with sha256 prefix 3aae140e5837753f; on the Windows laptop the GMM agrees only to about 1e-11 (R910/R911), so one CSV value and the sha256 may differ there.

## Result

(see summary.json)

## Conclusion (write after reading the results)

S003 reproduced through the new config route on Colab (job J20261022-212641-pro1-ssc): AUC 1.0 in 100/100 folds, split signature 5708ce5c3828, test_pred bit-identical to 20261022-050708_r003_gmm2_hv (max |diff| 0), and make_submission --dry-run gives sha256 prefix 3aae140e5837753f, the hash of S003. The README reproduction route works; meta.json now carries compute.research_data, threadpools (OpenBLAS 0.3.30, Haswell) and the estimator parameters. Hypothesis confirmed. (main session)
