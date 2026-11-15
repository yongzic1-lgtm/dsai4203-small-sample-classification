# 20261022-051627_r001_data_structure

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Few features carry signal; informative ones may differ in variance/shape from noise and may be linearly redundant (make_classification-like generator).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Hypothesis supported. 13 high-variance features (8,15,26,30,46,49,64,67,77,80,98,100,112) hold 9 of the 10 most target-correlated features; 8 near-zero correlation eigenvalues reveal exact linear redundancy; no train/test shift (all KS p >= 0.01). Led to R002. Refined later by R150 (generator: 12 informative + 8 redundant columns). (written 2026-10-22 by the main session after round 1)
