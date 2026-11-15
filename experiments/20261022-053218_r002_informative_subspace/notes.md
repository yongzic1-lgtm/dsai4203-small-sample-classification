# 20261022-053218_r002_informative_subspace

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The high-variance block spans a ~5-dim subspace (informative + redundant linear combos); the remaining features are independent Gaussian noise; classes form Gaussian clusters there.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Partly supported. The block has rank 12 (not ~5), and features outside it correlate with the target and the block, so the variance rule is incomplete (later explained by R150/R100: the 7 extra structured columns are exact linear combinations of the block). Key result: an unsupervised 2-component full GMM on 20,000 rows splits train labels 0.000 / 0.991 -> motivated R003 and D007. (written 2026-10-22 by the main session after round 1)
