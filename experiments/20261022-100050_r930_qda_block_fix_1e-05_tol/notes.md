# 20261022-100050_r930_qda_block_fix_1e-05_tol

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Profile of fixed reg values around the r210 grid edge (post-hoc, not an honest estimate of a procedure): prediction: CV AUC 0.9999-1.0000 for every reg in [1e-5, 3e-3]; OOF log-loss minimal near 1e-4-3e-4 and rising again at 1e-5 (the 3-10 extra noise columns of the in-fold block make tiny reg overfit).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Post-hoc profile point below sklearn's rank-check floor (reg 1e-05 with QDA tol 1e-8): CV AUC 0.99995 (3 folds below 1.0) and OOF log-loss 0.0080 (3e-5) / 0.0085 (1e-5), i.e. worse than reg 1e-4 - 3e-4 on both. Prediction (log-loss rises again below 1e-4) confirmed; AUC also drops by 3e-5. (cand)
