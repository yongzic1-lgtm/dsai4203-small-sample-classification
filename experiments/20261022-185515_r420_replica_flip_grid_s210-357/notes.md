# 20261022-185515_r420_replica_flip_grid_s210-357

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. (1) At the 4 original levels and the first 16 public subsets the arrays equal the archived r933 tn0 arrays bit for bit (same seeds, nested label noise). (2) S003's private AUC (csv scores) has mean ~1 - f/2 at every level (R922: 0.99244 at 0.015, 0.98995 at 0.02) and sd ~6.6e-4 / 7.8e-4 there. (3) The paired private differences vs S003 stay flat in f up to 0.02 (r201 ~ -2.6e-4, r209 ~ -7e-5, GMM family ~ -1e-6). (4) Public AUC of S003 within +-1.5e-4 of 0.99923 occurs in ~1-3 % of (level, subset) pairs at f = 0.0025-0.01 and much less at f = 0 or f >= 0.02. (5) Conditional on S003's public, the public AUCs of r201 and r209 change by < 1e-4 when the grid is extended (their public difference to S003 comes from boundary misorderings that do not depend on f), so B006 (r201 ~0.9988) and B007 (r209 ~0.9991) stay about right.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported; second half of the extended-grid replicas (148 replicas, 1,077 s, 30,167 stored ABC rows), pooled with 20261022-182358_r420_replica_flip_grid_s136-209 in 20261022-191747_r425_flip_grid_reweight. Bit-exact against the archived r933 tn0 runs at the 4 shared levels (r425 check table, max |diff| 0). Pooled per-level results (r425 levels table): S003 private mean = 1 - f/2 within 1.2e-4 up to f = 0.03; r201 - S003 private -2.4e-4 to -2.6e-4 and r209 - S003 -5.5e-5 to -7.0e-5 at every level, i.e. flat in flip_y as predicted. (r2_noise)
