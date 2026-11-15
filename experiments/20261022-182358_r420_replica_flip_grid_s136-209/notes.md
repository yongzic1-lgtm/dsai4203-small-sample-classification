# 20261022-182358_r420_replica_flip_grid_s136-209

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. (1) At the 4 original levels and the first 16 public subsets the arrays equal the archived r933 tn0 arrays bit for bit (same seeds, nested label noise). (2) S003's private AUC (csv scores) has mean ~1 - f/2 at every level (R922: 0.99244 at 0.015, 0.98995 at 0.02) and sd ~6.6e-4 / 7.8e-4 there. (3) The paired private differences vs S003 stay flat in f up to 0.02 (r201 ~ -2.6e-4, r209 ~ -7e-5, GMM family ~ -1e-6). (4) Public AUC of S003 within +-1.5e-4 of 0.99923 occurs in ~1-3 % of (level, subset) pairs at f = 0.0025-0.01 and much less at f = 0 or f >= 0.02. (5) Conditional on S003's public, the public AUCs of r201 and r209 change by < 1e-4 when the grid is extended (their public difference to S003 comes from boundary misorderings that do not depend on f), so B006 (r201 ~0.9988) and B007 (r209 ~0.9991) stay about right.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported; pooled with 20261022-185515_r420_replica_flip_grid_s210-357 in 20261022-191747_r425_flip_grid_reweight. (1) At the 4 r933 levels and the first 16 public subsets all csv-score public/private AUCs equal the archived r933 tn0 arrays bit for bit (222 replicas, 142,080 + 142,080 values, max |diff| 0; r425 check table). (2) S003's private AUC (csv scores) is 1 - f/2 within 1.2e-4 at every level: 0.99239 (sd 7.0e-4) at f = 0.015 and 0.98991 (sd 7.9e-4) at f = 0.02. (3) The paired private differences to S003 are flat in f: r201 -2.4e-4 to -2.6e-4 (MC SE 1-2e-5), r209 -5.5e-5 to -7.0e-5 at every level 0-0.03. (4) Acceptance of S003's public within +-1.5e-4 of 0.99923: 0.16 % at f = 0, 1.0-1.3 % at 0.0025-0.009, 0.28 % at 0.02. (5) See r425/r426 for the conditional public predictions (r201 0.99868, r209 0.99910; B006 / B007 hold). This job: 74 replicas, 553 s, 15,719 stored ABC rows. (r2_noise)
