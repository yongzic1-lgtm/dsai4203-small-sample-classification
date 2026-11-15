# 20261022-082329_r249_sup_comparison

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Summary analysis (no new model). Expected: the best strictly supervised model is a QDA variant at ~0.995; every linear and tree model is clearly worse; sklearn semi-supervised baselines lie between QDA and the TransductiveGMM; the GMM is better than the best supervised model by ~0.004-0.005 AUC with p_corr < 0.05.

## Result

(see summary.json)

## Conclusion (write after reading the results)

63 official candidates of the track (587 configurations in total) + recomputed TransductiveGMM. Best strictly supervised: r210_qda_block_nll (1.0000, log-loss 0.0087); pre-registered AUC-tuned r201_qda_block 0.9999. GMM vs best supervised: not distinguishable in CV AUC (p_corr 0.84). Linear ceiling ~0.94, trees ~0.91, GPC-ARD 0.978, graph SSL <= 0.957; self-training QDA = 1.0000 with the best log-loss (0.0071) and the highest agreement with the GMM on the 204 test boundary rows (Spearman 0.70 vs 0.45-0.49 for supervised QDA). Value of unlabeled data shows in the label-budget curve (r249_label_budget.png), not in 250-label CV AUC. Hypothesis partly refuted: the GMM is NOT better than the best supervised model by 0.004 (that was the over-regularised R003 QDA). (sup)
