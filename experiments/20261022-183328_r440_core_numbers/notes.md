# 20261022-183328_r440_core_numbers

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Audit hypothesis (written before running): every CV number in sup.md, fm.md and registry/submissions.csv re-computes from fold_scores.csv to the printed precision; the generator facts in gen.md match the R150/R154/R920-R922 summaries; the r249 table reproduces row by row. Expected exceptions are wording, not numbers: 'no method exceeded 0.88 on all 150 features' (fm's TabICL/TabPFN/horseshoe on all 150 features are higher), 'horseshoe = best linear model' (block L2 0.938 is higher), and R206's partial means at budgets where most fits are undefined. For the label budget, R441 (new folds and draws) gives the same first budget reaching AUC 0.99 as R206 for every matched model, within one grid step, and no budget with both runs fully defined differs by > 3 SE.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(r2_audit, 2026-10-22) Hypothesis confirmed: 233 of 240 claims match the archive at the printed precision, and all 64 rows of r249_comparison.csv recompute from fold_scores.csv (mean, sd, paired diff, split SE, corrected p). The 7 "no" rows are 2 wording errors (P31 "no method above 0.88 on all 150 features": TabICL 0.9929; F21 "horseshoe = best linear model": block L2 0.938), 2 last-digit roundings in fm.md (F13 0.9989 not 0.9990, F27 99.6% not 99.5%), 2 items not checkable from the archive (D5 public size, F42 BQDA ties), and L14 (the nested S005 procedure needs 120 labels for mean AUC >= 0.99, not R206's 80). Label budget: 37 cells defined in both R206 and R441, max |z| 1.98, none beyond 2 SE.
Superseded by 20261022-184020_r440_core_numbers (same numbers; adds claims D10-D11 and fixes the journey wording "pending" -> "not submitted" for candidates without a sub_id). Cite the later run.

