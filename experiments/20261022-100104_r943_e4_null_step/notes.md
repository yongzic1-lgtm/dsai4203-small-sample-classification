# 20261022-100104_r943_e4_null_step

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The gap rule always returns a largest step; under permuted labels the top of the sorted curve is also steep (E4: the top noise feature reaches 0.73-0.93 at its 95% quantile), so the null largest step is expected to be 0.2-0.4 for k = 10-20, i.e. the observed 0.30-0.49 is not far above noise for k = 20. The perm rule should admit 0 features under the null in about 95% of leave-one-out checks. E3: more features than ~48 exceed |corr| = 1/sqrt(250) on the real labels because 20 features carry signal.

## Result

(see summary.json)

## Conclusion (write after reading the results)

E4's drop table is reproduced exactly with E4's own functions and seeds (m_gap, gap size, m_half, m_binom, m_perm, null top-1 q95 all equal reports/tables/e4_drop.csv; checked locally after the merge). Under 50 permuted-label curves the largest step is 0.13-0.15 on average (q95 0.25-0.36, max 0.39-0.51) and sits at position 1-2; the observed steps 0.49/0.37/0.37/0.30 are exceeded by 1 of 50 null curves for every k (p ~ 0.02), so the drop is sharper than noise but not by a wide margin at k = 20. The half rule admits >= 1 pure-noise feature in 28/72/90/100% of null curves (k = 5/10/13/20; mean 5.2 at k = 20), the binomial rule 27-40, while the pre-registered perm rule admits >= 1 in only 6-8% (leave-one-out), i.e. only the perm rule is calibrated, and it moves from 8 to 17 at k = 20. E3: 61 features exceed |corr| = 1/sqrt(250) on the real labels vs 50 (mean of 50 permutations; 47.6 expected). (verifier ext)
