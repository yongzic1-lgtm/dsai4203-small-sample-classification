# 20261022-161150_r944_e3_replica_decomp

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

E3 attributes its whole gap to leaky optimism (winner's curse among real features for k_feat < 7, lucky noise features above). Expectation on parametric replicas (100 fresh 250-row samples, E3 protocol unchanged): (i) honest CV is ~unbiased for its own fold models at every k_feat (|honest_cv - test_honest_fold| < 0.005); (ii) leak optimism (leaky_cv - test_leaky_fold) is positive for every k_feat < 150 but small (< 0.015) at k_feat = 2-7 and large (> 0.05) at k_feat 20-50, so the small-k gap is mostly the fact that the honest filter sees 200 instead of 250 rows (honest pessimism relative to the final model), not optimism; (iii) against the final 250-row model (the LB target), leaky CV is within +-0.015 at k_feat <= 7 and optimistic by > 0.03 at k_feat >= 20; (iv) the near-zero gap at k_feat = 7 is specific to the real sample (mean replica gap at 7 clearly > 0.001), and the replica gap curve has E3's shape (peak at 20-50, 0 at 150).

## Result

(see summary.json)

## Conclusion (write after reading the results)


Over 100 replica samples (5-fold) honest CV is unbiased for its own fold models (|bias| <= 0.0044 at every k_feat), and the mean gap 0.015/0.010/0.008/0.024/0.041/0.064/0.078/0.087/0.056 (k_feat 2..100) is almost entirely leak optimism (0.014/0.009/0.002/0.014/0.030/0.054/0.070/0.079/0.052); hypothesis (i), (iii), (iv) confirmed, but (ii)'s "small-k gap is mostly honest pessimism" is refuted ON AVERAGE (honest - final is only -0.002/-0.003 at k_feat 2/5). The real k_feat = 2 gap 0.056 is a 96th-percentile draw (replica mean 0.015, sd 0.017); in the 8 replicas with gap >= 0.04 it comes from honest-CV under-estimation (honest bias -0.037, leak optimism +0.002), matching the real-data proxy pattern. Leak optimism is flat in k_fold (0.054/0.054/0.053 for 2/5/10 folds at k_feat 20) while honest - final shrinks (-0.046/-0.016/-0.010), confirming E3's k_fold explanation; the gap peaks at k_feat 50 (65% of draws) or 30 (25%). (verifier ext)
