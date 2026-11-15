# 20261022-182022_r441_label_budget_rerun

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Re-running R206's design on new folds (official repeats 4-19) and new label subsamples reproduces its label-budget statements within sampling noise: the transductive GMM (S003) is at AUC ~1.0 from 2-4 labels (1-2 per class; R304 saw 1.0 at 2 labels in 50/50 subsets); self-training QDA-LW reaches mean AUC >= 0.99 at about 20 labels (threshold 0.75, R206) and the S006 configuration (threshold 0.99) at 20-40; QDA reg 0.003 on the train+test block at about 40; strictly supervised QDA (reg 0.003, in-fold block) at about 80; the actual nested S005 procedure (r201) is undefined below ~40 labels and reaches 0.99 at 80-120. At every common budget the mean AUC of the R206-equivalent curves is within 3 combined fold-cluster SEs of R206's. A difference beyond 3 SE at several budgets, or a first-hit budget that moves by more than one grid step, would mean R206's numbers depend on its seeds and must be quoted with an interval.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(r2_audit, 2026-10-22) Hypothesis confirmed for every R206 statement: on 80 new folds x 5 new label draws, the first budget with mean AUC >= 0.99 is 4 for the GMM (1.0 already at 2 labels, 400/400), 20 for self-training QDA-LW (0.9959; S006's threshold 0.99: 0.9991), 40 for QDA reg 0.003 on the train+test block (0.9925) and 80 for the strict in-fold QDA reg 0.003 (0.9939); no budget defined in both runs differs from R206 by more than 2 fold-cluster SE (r440 compare table). The actual nested S005 procedure (r201) needs more: mean 0.9842 at 80 labels (median 0.9984), 0.9993 at 120, undefined in 371/400 draws at 40; log-loss tuning (r210) is worse at small budgets (0.9376 at 80), and the strictly inductive Bayesian QDA needs 160. Quote "about 80-120 labels for strictly supervised QDA" rather than "about 80".

