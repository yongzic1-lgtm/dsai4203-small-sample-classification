# 20261022-072747_r205_select_supervised

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Winner's curse check. Prediction: the inner CV picks a QDA configuration in >= 90% of outer folds; the selection-aware outer AUC is within 0.003 of the best single nested supervised family (QDA, ~0.995), i.e. the optimism from choosing among ~70 supervised configurations is small because one family dominates. Refuted if the outer AUC is > 0.01 below the best single candidate.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Selection-aware (fully nested) estimate of 'choose the best of a 46-configuration strictly supervised menu': 0.9995 +/- 0.0006 (log-loss 0.028). The inner CV picked QDA in 100/100 outer folds (variance ratio 1.6: 51, 1.7: 29, 1.5: 20; reg 0.01 = the menu's grid bottom in 95). Caveat: the menu's QDA reg grid stops at 0.01 (written before R201 showed 0.001-0.003 is best), so the 0.0004 gap to the best single run (R201 qda_block 0.9999) mixes winner's-curse optimism with the coarser grid. Either way the optimism of reporting the best of ~400 supervised configurations is at most ~0.0005 AUC here, because one family (QDA) dominates. Prediction confirmed. (sup)
