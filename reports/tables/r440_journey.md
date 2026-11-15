Journey from the naive baseline to the final candidates (official 5x20 CV; public LB from registry/submissions.csv; HM SE = Hanley-McNeil SE of the expected public AUC on 395 rows (198/197); surprise = (public - expected) / HM SE; pending rows show the replica-ABC prediction given S003's public 0.99923 (r936_public_abc, CSV scores); last column = expected private-AUC difference to S003 on 222 parametric replicas (r936_private, tn0, csv8, R156 weights); S003's own private expectation is about 0.997 (R922, flat prior). Run 20261022-184020_r440_core_numbers.)

| Step | Sub | Model | Test features used | CV AUC (sd) | Expected public | Public LB | Public - CV | Surprise (HM SE) | Replica private vs S003 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S001 | L1 logistic, all 150 features | labels only | 0.8469 (0.0120) | 0.8500 | 0.83589 | -0.0110 | -0.72 |  |
| 2 | S002 | QDA reg 0.1 on the 13-column block | block from train+test features | 0.9958 (0.0011) | 0.9900 | 0.99277 | -0.0030 | 0.55 |  |
| 3 | S003 | Transductive GMM K=2 on the block | yes (transductive) | 1.0000 (0.0000) | 0.9990 | 0.99923 | -0.0008 | 0.14 | 0 (reference) |
| 4 | S004 | E3 final: in-fold corr top-20 + L2 logistic | labels only | 0.7934 (0.0142) | 0.8200 | 0.79136 | -0.0021 | -1.35 |  |
| 5 | S005 | Strict supervised QDA, nested reg (r201) | labels only (strictly inductive) | 0.9999 (0.0002) | 0.9988 | pending (pred. 0.99884 ± 0.00066) |  |  | -2.6e-04 |
| 6 | S006 | Self-training QDA-LW (r209) | yes (self-training on test rows) | 1.0000 (0.0000) | 0.9991 | pending (pred. 0.99913 ± 0.00033) |  |  | -6.9e-05 |
| - | - | Strict QDA, nested reg, log-loss inner (r210) | labels only | 1.0000 (0.0001) |  | not submitted (pred. 0.99901 ± 0.00051) |  |  | -1.7e-04 |
| - | - | Bayesian QDA EB, train+test block (r255) | block from train+test features | 1.0000 (0.0000) |  | not submitted (pred. 0.99906 ± 0.00033) |  |  | -1.2e-04 |
| - | - | Bayesian QDA EB, in-fold block (r255) | labels only (strictly inductive) | 0.9990 (0.0011) |  | - |  |  |  |
