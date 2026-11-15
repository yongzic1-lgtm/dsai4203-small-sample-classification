# Extension E3: Information leakage

**Author: TODO Full Name** (Kaggle: TODO)

*AI use: Claude Code (Anthropic; Claude Opus 5.5) designed this study, wrote and ran its code on Google Colab, uploaded S004 with our authorisation and drafted this text; a separate AI verification round re-checked the numbers against the archive (R944). [TODO owner: say what you re-ran, rewrote and checked.] Log: `registry/ai_usage.md`, rows 3, 4, 7, 8, 10 and 21.*

<!--
Sources (all numbers): E3 run 20261022-063908_e3_leakage (Colab job J20261022-064045-pro1-rext34) ->
reports/tables/e3_grid.csv (= experiments/<run>/cells.csv), e3_null.csv, e3_leaderboard.csv (refreshed with S004 by
run.py --refresh-lb); S004 = registry/submissions.csv, pre-registration B005.
Proxy test AUC: run 20261022-072942_r321_e3e4_proxy_test -> reports/tables/r321_e3_proxy.csv (proxy_auc_hard).
Replicas: run 20261022-161150_r944_e3_replica_decomp -> reports/tables/r944_e3_replica_decomp.csv, per_draw.csv.
Corrected SE = gap_mean / corrected_t (cells.csv, k_feat 20, 5-fold: 0.0733 / 3.72 = 0.0197).
LB SE 0.0233 = public_lb_se_approx in e3_leaderboard.csv (Hanley-McNeil at AUC 0.79, 395 rows).
Edited 2026-10-22 by fix_ext34 (Claude Code) after verification round 1 (verify.md items 15, 19 = synthesis corrections 13, 17).
Length: the figure reports/figures/e3_gap_vs_kfeat.png (gap vs k_feat, one line per k_fold) was taken out to keep
the page within 2 A4 pages; its 2- and 10-fold values are now a table column. Its fonts are legible only at
>= 80 % of the text width, which does not fit. Checked with XeLaTeX, Calibri 11 pt, A4, 2.54 cm margins.
-->

## Question

The filter "keep the k features with the largest |corr(feature, target)|" is applied either once on all 250 rows
before k-fold CV (*leaky*) or inside every fold on its training part only (*honest*). How much too high is the
leaky CV AUC, compared with honest CV and with the leaderboard? How does the gap change with k (both k_feat, the
features kept, and k_fold, the folds), and why?

## Protocol

Model: standard scaler, Pearson |corr| top-k_feat filter, L2 logistic regression (C = 0.1, fixed seed;
scikit-learn, Pedregosa et al. 2011). The procedures differ only in which rows the filter sees (selection bias:
Ambroise & McLachlan 2002; Hastie et al. 2009, Sec. 7.10.2) and share their splits: 10 values of k_feat × k_fold
in {2, 3, 5, 10}, 20 stratified split seeds each (5-fold = the team's official CV). Statistics: mean and SD over
seeds, Nadeau–Bengio corrected paired t-test (Nadeau & Bengio 2003). Checks: the gap is exactly 0 at
k_feat = 150, and honest CV is about 0.5 with permuted labels (30 permutations × 5 seeds). The final model
(k_feat = 20, all rows; the same for both procedures) was submitted once (S004; expected score pre-registered in
B005). As the public LB has only ~395 rows (SE 0.023; Hanley & McNeil 1982), every k_feat is also scored on the
19,750 test rows against proxy labels from the team's transductive GMM (public 0.99923). The proxy is not a
leaderboard score.

## Results

| k_feat | leaky CV | honest CV | gap ± SD | corrected p | fold overlap | gap, 2 / 10 folds | null gap | proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.787 | 0.731 | 0.056 ± 0.020 | 0.052 | 0.66 | 0.062 / 0.056 | 0.129 | 0.776 |
| 5 | 0.816 | 0.794 | 0.023 ± 0.010 | 0.14 | 0.81 | 0.020 / 0.025 | 0.162 | 0.821 |
| **7** | 0.835 | **0.834** | **0.001 ± 0.003** | 0.84 | **1.00** | 0.014 / 0 | 0.178 | 0.852 |
| 10 | 0.843 | 0.818 | 0.025 ± 0.014 | 0.098 | 0.87 | 0.040 / 0.025 | 0.187 | **0.864** |
| 13 | 0.851 | 0.806 | 0.044 ± 0.010 | 0.011 | 0.82 | 0.064 / 0.044 | 0.196 | 0.841 |
| **20** | 0.867 | 0.793 | **0.073 ± 0.014** | 0.0003 | 0.78 | 0.087 / 0.063 | 0.206 | 0.820 |
| 30 | 0.881 | 0.791 | 0.090 ± 0.014 | 0.0001 | 0.75 | 0.107 / 0.082 | 0.206 | 0.783 |
| **50** | **0.894** | 0.803 | **0.090 ± 0.014** | < 0.0001 | 0.77 | 0.115 / 0.085 | 0.186 | 0.825 |
| 100 | 0.840 | 0.796 | 0.044 ± 0.012 | 0.004 | 0.84 | 0.063 / 0.037 | 0.103 | 0.805 |
| 150 | 0.769 | 0.769 | 0 (exact) | 1 | 1 | 0 / 0 | 0 | 0.802 |

*Table 1. 5-fold CV (official splits), mean and SD over 20 split seeds; gap = leaky − honest. Fold overlap: share
of the all-rows feature set that the in-fold filter also picks. Null gap: permuted labels, 5-fold (leaky
0.61–0.71, honest 0.49–0.51). Proxy: all-rows model vs GMM proxy labels on 19,750 test rows (run
20261022-072942_r321_e3e4_proxy_test). Run 20261022-063908_e3_leakage; e3_grid.csv, e3_null.csv,
r321_e3_proxy.csv.*

**k_fold (k_feat = 20).** The gap is 0.087 / 0.082 / 0.073 / 0.063 for 2 / 3 / 5 / 10 folds: leaky CV stays at
0.854–0.867 while honest CV rises from 0.767 to 0.805 (fold overlap 0.629 to 0.855). Under the null the gap is
flat (0.201 / 0.206 / 0.210 for 3 / 5 / 10 folds).

**Leaderboard (S004, k_feat = 20).** Public AUC 0.79136. Leaky CV (0.867) is 0.075 above it (3.2 LB standard
errors), honest CV (0.793) 0.002 above. The proxy (0.820) is 0.028 above it (1.2 SE). The public score lies inside the
pre-registered 90% range 0.785–0.854 (B005; reports/tables/e3_leaderboard.csv).

## Conclusion

**How much too high.** At k_feat = 20 (the submitted configuration) leaky CV exceeds honest CV by 0.073 (SD over
20 split seeds 0.014; Nadeau–Bengio corrected SE 0.020, p = 0.0003) and the public score by 0.075. With 5 folds the
gap reaches 0.090 (k_feat = 30–50), with 2 folds 0.115. Honest CV matched the public score to 0.002, partly by luck
(LB SE 0.023). Against the proxy, leaky CV is 0.047 too high and honest CV 0.026 too low (10-fold: 0.015), as
honest models train on 200 of the 250 rows.

**k_feat: not monotone, and why.** For a noise feature with all-rows correlation s, the training-part and
validation-part correlations have expected product s² − 1/n under leaky selection, so a noise feature kept for its
large |s| adds fake signal on the validation rows. Here 7 features have |corr| 0.34–0.40 (5–6 null SEs), all
others ≤ 0.24. At k_feat = 7 the in-fold filter re-selects these 7 in every fold, and the gap vanishes; each
further feature is mostly lucky noise, and the gap grows. The summed optimism Σ(s² − 1/n) peaks when all features
with |s| > 1/√n are kept (61 here, 48 under pure noise); the AUC gap levels off earlier (k_feat = 20–30 under the
null, 30–50 on the real data) and is 0 at 150.

**What is specific to this sample.** For k_feat ≤ 10 leaky CV is within 0.021 of the proxy and honest CV
0.017–0.046 below it, so our small-k gap is mostly honest-CV pessimism; for k_feat = 20–50 leaky CV is
0.047–0.098 above the proxy. This is not the average case. On 100 replica datasets from a parametric model
fitted to our data (R944), honest CV is unbiased for its own fold models (|bias| ≤ 0.004) and the mean small-k gap
(0.015 at k_feat = 2) is leak optimism. Our 0.056 at k_feat = 2 is a 96th-percentile sample; replicas with such gaps have a
low honest estimate (bias −0.037). The exact zero at k_feat = 7 is partly luck (replica median 0.006).

**k_fold.** More folds put less of each validation part into the selection but make its correlation noisier; to
first order the effects cancel (flat null gap). On the real data the gap shrinks because larger training parts help
the honest filter find the 7 strong features (R944: leak optimism 0.054 / 0.054 / 0.053 for 2 / 5 / 10 folds).

**Consequence.** Leaky CV also picks the wrong model, k_feat = 50 (proxy 0.825); honest CV picks k_feat = 7
(proxy 0.852), near the proxy optimum k_feat = 10 (0.864). Every label-dependent step, feature selection included,
belongs inside the pipeline that is refitted in each fold.

**References.** Ambroise, C. & McLachlan, G. J. (2002), *PNAS* 99, 6562–6566. · Hanley, J. A. & McNeil, B. J.
(1982), *Radiology* 143, 29–36. · Hastie, T., Tibshirani, R. & Friedman, J. (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer. · Nadeau, C. & Bengio, Y. (2003), *Machine Learning* 52, 239–281. · Pedregosa, F.
et al. (2011), scikit-learn, *JMLR* 12, 2825–2830.
