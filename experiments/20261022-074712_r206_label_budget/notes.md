# 20261022-074712_r206_label_budget

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Value of unlabeled data. Prediction: TransductiveGMM stays at AUC ~1.0 down to 4-6 labels (labels only name the two components); self-training QDA and label spreading lie in between; supervised QDA on the train+test block needs >= 80 labels to pass 0.98 and falls below 0.9 at <= 10 labels; the strict in-fold QDA is worse still at small n_l because the variance rule on a few rows cannot find the block; L2 logistic stays <= 0.9. (A smoke check showed SelfTraining with a weak base model can fall BELOW its base: self-training is not guaranteed to help.)

## Result

(see summary.json)

## Conclusion (write after reading the results)

Value of unlabeled data (first 4 official repeats x 5 folds x 3 draws; mean validation AUC). TransductiveGMM = 1.000 from 4 labels on. Self-training QDA-LW: 0.80 (6), 0.88 (10), 0.995 (20), 0.9996 (40), 1.000 (>= 80). Supervised QDA-LW on the train+test block: 0.63 (6), 0.68 (10), 0.85 (20), 0.94 (40), 0.98 (80), 0.996 (200); QDA reg 0.003 (train+test block) 0.993 (40), 0.998 (80); strict in-fold QDA reg 0.003 0.994 (80), 0.999 (120), 1.000 (200); L2 logistic <= 0.90; label spreading <= 0.92. Labels needed to reach AUC 0.99: ~4 (GMM), ~20 (self-training), ~40 (supervised QDA, label-free block), ~80 (strictly supervised). Hypothesis confirmed in shape; the supervised side reaches 0.98 later than predicted only for LW shrinkage. The reg-0.003 curves are informed by R201's nested choice. Figure redrawn by R249 as r249_label_budget.png. (sup)
