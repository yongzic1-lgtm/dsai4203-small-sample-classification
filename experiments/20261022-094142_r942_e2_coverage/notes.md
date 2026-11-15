# 20261022-094142_r942_e2_coverage

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

E2's intervals resample subsets of ONE 250-row sample, so as statements about the population AUC at 500 rows they should under-cover: the 90% bootstrap interval of the power law covers the replica truth in clearly fewer than 90% of replicas (expected 30-60%), the envelope in more. The power-law point is roughly unbiased if the back-test (R308) is representative.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Over 100 parametric replicas (fresh 250-row samples, E2 protocol unchanged except 500 bootstrap refits and no out-of-subset view), E2's 90% power-law bootstrap interval for AUC at 500 training rows covers the population truth (L1 0.9230, R941) in only 21% of replicas (QDA: 57%); the power-law point is biased low by -0.031 for L1 (sd 0.031, RMSE 0.044), log-linear is biased high by +0.017 (coverage 37%), and even the envelope covers only 81% (L1 and QDA). Hypothesis confirmed: E2's intervals are far too narrow as statements about the population, and the power law underestimates the L1 gain on average. (verifier ext)
