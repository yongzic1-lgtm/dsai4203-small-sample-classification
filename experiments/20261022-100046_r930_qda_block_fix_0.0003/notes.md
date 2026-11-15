# 20261022-100046_r930_qda_block_fix_0.0003

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Profile of fixed reg values around the r210 grid edge (post-hoc, not an honest estimate of a procedure): prediction: CV AUC 0.9999-1.0000 for every reg in [1e-5, 3e-3]; OOF log-loss minimal near 1e-4-3e-4 and rising again at 1e-5 (the 3-10 extra noise columns of the in-fold block make tiny reg overfit).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Post-hoc profile point (fixed reg 0.0003, not an honest estimate): see the reg profile in the Conclusion of 20261022-100052_r930_reg_edge_summary. CV AUC 0.99998 (1 fold below 1.0) for every reg in [1e-4, 3e-3]; OOF log-loss 0.0137 / 0.0087 / 0.0071 / 0.0074 for 0.003 / 0.001 / 3e-4 / 1e-4. (cand)
