# 20261022-070600_r304_label_requirement

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The transductive GMM reaches AUC ~1 on the unseen rows with only a handful of labels (one per class maps the two components correctly whenever the labelled negative is not in the positive component); supervised QDA on the block needs on the order of 100-200 labels to approach 0.99, and the L1 baseline stays below 0.85 even at 200 labels.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Out-of-subset AUC over 50 label sets per size: the transductive GMM scores 1.0000 in every set already with 2 labels (one per class); supervised QDA on the block is undefined below 14 rows per class (sklearn svd solver needs more rows per class than the 13 dimensions; n <= 20 fails), 0.950 at 30, 0.971 at 40, median >= 0.99 from 75 labels; L1 C=0.1 is a constant predictor (AUC 0.5) up to 20-30 labels and 0.848 at 200. Hypothesis confirmed. (ext12)
