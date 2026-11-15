# 20261022-184020_r440_core_numbers

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Audit hypothesis (written before running): every CV number in sup.md, fm.md and registry/submissions.csv re-computes from fold_scores.csv to the printed precision; the generator facts in gen.md match the R150/R154/R920-R922 summaries; the r249 table reproduces row by row. Expected exceptions are wording, not numbers: 'no method exceeded 0.88 on all 150 features' (fm's TabICL/TabPFN/horseshoe on all 150 features are higher), 'horseshoe = best linear model' (block L2 0.938 is higher), and R206's partial means at budgets where most fits are undefined. For the label budget, R441 (new folds and draws) gives the same first budget reaching AUC 0.99 as R206 for every matched model, within one grid step, and no budget with both runs fully defined differs by > 3 SE.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(r2_audit, 2026-10-22) Final audit run; supersedes 20261022-183328_r440_core_numbers (same numbers, plus claims D10-D11 and "not submitted" wording in the journey table). Hypothesis confirmed: 235 of 242 claims match the archive at the printed precision and all 64 r249 rows recompute from fold_scores.csv. The 7 "no" rows: two wording errors (P31 "no method above 0.88 on all 150 features" - TabICL v2 0.9929; F21 "horseshoe = best linear model" - block L2 0.938), two last-digit roundings in fm.md (F13 0.9989, F27 99.6%), two items not checkable from the archive (D5 public size, F42 BQDA ties), and L14 (the nested S005 procedure needs 120 labels, not 80). Label budget vs R206: 37 comparable cells, max |z| 1.98. Report: research/reports/r2_audit.md.

