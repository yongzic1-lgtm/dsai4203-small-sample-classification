# 20261022-092943_r900_perm_null_b

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Null B (train on permuted labels, evaluate against the TRUE labels, official splits): honest pipelines are symmetric around 0.5 (count of permutations above 0.5 consistent with Binomial(n, 1/2); the GMM pipelines are bimodal near 0 or 1 because labels only orient the components); ctrl_disk_reader stays > 0.95 in every permutation; ctrl_e3_leaky ~0.5 here.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(leak) Hypothesis confirmed. Trained on 20 permuted label vectors and scored against the TRUE labels on the official splits, no honest pipeline shows a preference for the true orientation: permutations with mean AUC > 0.5 are s003_gmm2_hv 11/20 (sign-test p 0.82; bimodal, 0.04-1.00, because labels only orient the label-free components), r155 11/20 (identical values), r201 10/20 (0.518 +/- 0.114), r210 11/20 (0.516 +/- 0.131), r209 8/20 (bimodal 0.00-0.995), r255 10/20 (0.538 +/- 0.128), e3_honest 10/20 (0.504 +/- 0.040). The positive control ctrl_disk_reader (reads train.csv's target) is 1.000 in 20/20 permutations, so a channel that brings the true labels in from outside the training fold would have been seen. ctrl_random is constant 0.508 here by construction (its seed depends only on the training rows, which are the same official splits in every permutation).
