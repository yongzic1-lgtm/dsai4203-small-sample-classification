# Extension E4: Stability of the feature ranking

**Author: TODO Full Name** (Kaggle: TODO)

<!--
Sources (all numbers): E4 run 20261022-071414_e4_feature_stability (Colab job J20261022-071554-pro1-rext34) ->
reports/tables/e4_frequency.csv, e4_drop.csv, e4_stable_sets.csv, e4_usefulness.csv, e4_paired.csv,
e4_infold_selection.csv, reports/figures/e4_frequency_sorted.png.
Null comparison of the largest step and null calibration of the rules: run 20261022-100104_r943_e4_null_step ->
reports/tables/r943_e4_drop_vs_null.csv (frac_null_step_ge_obs = 0.02 = 1 of 50; +1-corrected p = 2/51 = 0.039).
Stable set vs label-free block: run 20261022-074624_r320_e4_stable_vs_block -> reports/tables/r320_usefulness.csv,
r320_paired.csv, r320_feature_table.csv. (The proxy test AUC of R321, r321_e4_proxy.csv, is no longer quoted here.)
Best linear model: run 20261022-063713_r300_block_l2_c1 (reports/tables/r300_candidates.csv, 0.898).
Config: extensions/e4_feature_stability/config.yaml (drop.primary_rule: perm).
Edited 2026-10-22 by fix_ext34 (Claude Code) after verification round 1 (verify.md item 18 = synthesis
corrections 16-17; verify_ext.md section 3). The former Table 1 (drop position by rule, e4_drop.csv) was merged into the figure (its panel labels
give m* for every rule) and one sentence of Results. Checked with XeLaTeX, Calibri 11 pt, A4, 2.54 cm margins.
Figure 1 redrawn 2026-11-01 (ext_figures, Claude Code) by
`extensions/e4_feature_stability/run.py --replot 20261022-071414_e4_feature_stability`, from the run's archived
frequency.csv, null_sorted_curves.csv and drop.csv (nothing recomputed): one row of four panels at the full text
width, printed text 6.3-7 pt (before: 2-3 pt), log rank axis, legend inside the figure. The panel titles now give
m* under the perm rule only; all four rules are in the Results sentence below the figure. Place the figure at
100% of the text width (its natural size): E4 measures 1.86 pages (plain spacing, content-fitted tables;
1.94 with equal-width tables), as before the redraw.
History 2026-11-24: rewritten as a formal article by Claude Code (formal-rewrite workflow, agent for part e4; baseline =
git 259279b): telegraphic results turned into paragraphs, terms and abbreviations defined, archive identifiers used only
as bracketed citations, captions and the reference list formalised (titles omitted for the page limit, venues abbreviated
as in the core report); no number changed. Results order: text first, then Figure 1 and Table 1 (page 2 top). Added
from the archive: "most others below it" in the Conclusion (e4_frequency.csv: at k = 5 / 10 / 13 / 20, 142 / 132 / 127 /
114 of the 143 features outside the top seven are below k/150). Details moved out of the visible text in this rewrite
(all still in the archive) are listed in the comments inside Results. Typeset 2026-11-24 with draft_build.py (Times New
Roman 11 pt): 2 pages, 1.94 used.
History 2026-11-24 (review round, Claude Code fixer for E4; findings XF1, XF5, XF6, XF11, XF18-XF22, XL28-XL36).
Baseline claims corrected from the archive: the "never stable" block features are now limited to the top-10 stable set,
with 67, 8 and 26 named as above the perm drop at k = 20 (67 also at k = 13), and "missed by the ranking" is gone
(features 67 and 8 are ranks 9 and 10 by |corr|) (r320_feature_table.csv, e4_stable_sets.csv;
research/reports/verify_ext.md section 3, E4 item 5); "(thinnest margin at k = 20)" -> "(narrowly for k >= 10)" (r943_e4_drop_vs_null.csv:
margin over the null 95% quantile 0.134 / 0.034 / 0.065 / 0.047 for k = 5 / 10 / 13 / 20, over the null mean 0.349 /
0.219 / 0.223 / 0.168, so "thinnest at k = 20" held only against the mean); "all others <= 28% at k = 10" -> "all others
stayed below 0.29 at k = 10" (e4_frequency.csv: feature 67 freq_k10 0.2825); "Alone, they gave" -> "Alone (6.4 features
per fold), they gave" (r320_usefulness.csv block_minus_stable n_features_mean 6.4); Table 1 n/a "fewer training rows per
class (n) than features (p)" -> "not run (per class, n < p)" (no QDA row for all 150 features in r320_usefulness.csv;
QDA-LW on all 150 was run in R249, 0.875); "which inflates CV" -> "which can inflate CV" (e4_paired.csv: leaky - honest
-0.001 at k = 10 and +0.0036 at k = 13, p = 0.65). Corrected wording, logged per HOUSE_STYLE section 5: "at k = 5 the
seven compete for five places and the curve is cut inside the group" -> "it lies below all of them at k = 5" (e4_drop.csv:
m* = 7 under every rule at k = 5; the largest step, 0.4935, lies between rank 7 and rank 8). Restored to the text: the
R249 benchmark (0.932-0.938), the perm exceptions 67, 8 and 26, and "+0.028, p = 0.30" in the Conclusion. Terms: "naive"
-> "leaky" for selection on all 250 rows, as in E3 and the core (archive variant names stay naive_*); "corrected paired
t-tests" -> "corrected t-tests"; block defined as in the glossary; "most others below it" given a verb ("and"); selection
frequencies as proportions throughout (98.5% -> 0.985; 4%, 65%, 79%, 92% -> 0.04, 0.65, 0.79, 0.92). AI statement:
"drafted and revised this text" (registry/ai_usage.md rows 10 and 21 and this rewrite), row 24 added (2026-11-23 redraw
of ext_e4.png and its caption), "The use is logged in" -> "The log is in"; archive-tag sentence in house wording.
Captions: Figure 1 caption cut to what the panel titles do not say (the removed encodings are in the comment under the
figure); Table 1 caption names runs instead of files; descriptive figure alt text; references in house style. Wording
shortened for the page limit: "on all 13 block columns" -> "on the block", "added nothing to it" -> "added nothing",
"the 17 features above" -> "the 17 above" (baseline wording), "more steeply than in 49 of 50" -> "than 49 of 50". Not
applied for the page limit: "inverse regularisation strength" for C (XL33; typeset 1.98) and the LIBLINEAR citation (Fan
et al., 2008; XF25, XL27), which the core report's software line cites; the archive-tag sentence stays in Question
(HOUSE_STYLE section 2 allows it; moving it into the AI statement as in E1 and E3 costs a line). Typeset with
draft_build.py: 2 pages, 1.94 used.
Main session 2026-11-25: 'admits 67 at k >= 13, and 8 and 26 at k = 20' (comma added against a misreading); 'binom (17–28)' -> 'binom (up to 28)' (binom m* is 7 at k = 5); AI-use statement lists row 25. Measured: 2 pages, 1.94 used.
2026-11-25 (main session, at the member's request): the AI-use statement is removed from this part for now; the member will write the AI disclosure together with the contribution table (course rule 6 requires it in the submitted report; facts in registry/ai_usage.md; previous wording in git 63b56f9).
2026-11-25 (main session, at the member's request): the reference list of this part moved to one complete, alphabetical list for the whole document, reports/references.md (written by scripts/make_references.py from registry/references.md; built as part 'refs' after the extensions).
-->

## Question

We ranked the 150 features by |corr|, the absolute Pearson correlation between a feature and the label, on bootstrap resamples of the 250 labelled rows. How often does each feature enter the top *k*, relative to the chance rate *k*/150? Does the sorted frequency drop sharply, and does the drop move with *k*? Does a model on the features above the drop match or beat one on all 150? How does being stably selected relate to being worth keeping? Bracketed identifiers cite the project archive: runs (R) and decisions (D). Cited works are listed in the References at the end of the document.

## Protocol

We drew *B* = 2,000 bootstrap resamples of the 250 rows (seed 2026; Efron, 1979) and ranked the features by |corr| in each, without fitting a model. A feature's top-*k* frequency is the share of resamples that rank it among the *k* largest, for *k* = 5, 10, 13 and 20; 13 is the size of the label-free high-variance block: the features with more than 1.5 times the median standard deviation over all training and test rows. The pre-registered primary rule *perm* sets the drop position *m*\*(*k*) to the number of leading features above the rank-wise 95% envelope of 50 permuted-label curves; secondary rules cut at the largest step between sorted frequencies (*gap*), at frequency 0.5 (*half*) or at the Bonferroni bound of Bin(*B*, *k*/150)/*B* (*binom*).

We measured usefulness with L2-penalised logistic regression (inverse penalty *C* = 0.1) under the official cross-validation (CV), stratified 5-fold CV repeated 20 times, with corrected *t*-tests (Nadeau & Bengio, 2003). *Leaky* sets are chosen on all 250 rows, which can inflate CV (Ambroise & McLachlan, 2002; Extension E3); *honest* sets are re-chosen in each training fold by stability selection (frequency ≥ 0.5; Meinshausen & Bühlmann, 2010) or as the top *m* by |corr|. A follow-up run compared L2 logistic regression, quadratic discriminant analysis (QDA; regularisation 0.1) and our transductive Gaussian mixture model (GMM; two components fitted without labels to training and test features [D007]) on the block, the in-fold stable set and their difference [R320], all in scikit-learn (Pedregosa et al., 2011).

## Results

Seven features (98, 17, 100, 112, 80, 77, 46; |corr| 0.34–0.40) had top-*k* frequencies ≥ 0.985 for *k* ≥ 10 and shared the five places at *k* = 5 (frequencies 0.53–0.95; Figure&nbsp;1a). Feature 49 (|corr| 0.24) reached 0.04, 0.65, 0.79 and 0.92 at *k* = 5, 10, 13 and 20; all others stayed below 0.29 at *k* = 10. For these *k*, *m*\* is 7, 8, 9 and 17 (perm), 7, 8, 8 and 8 (gap), 7, 8, 8 and 9 (half) and 7, 17, 22 and 28 (binom). The largest step was exceeded by 1 of 50 permuted-label curves at every *k* (*p* ≈ 0.02–0.04); on those curves the half rule admitted noise features in 72–100% of cases for *k* ≥ 10, the perm rule in 6–8% [R943].
<!-- Moved out of the text (2026-11-24; e4_drop.csv gap_size, r943_e4_drop_vs_null.csv obs_max_step): the observed largest steps are 0.49 / 0.37 / 0.37 / 0.30 for k = 5 / 10 / 13 / 20 (exactly 0.4935, 0.3695, 0.3690, 0.2975); under permuted labels the null largest step averages 0.13-0.15 (q95 0.25-0.36). The perm rule was checked on the null curves leave-one-out (each null curve against the envelope of the other 49; R943 null_frac_m_perm_loo_ge1). With no signal (permuted labels) the top noise feature reaches a top-10 frequency of 0.84 (95% quantile; e4_drop.csv null_q_top1, r943_e4_drop_vs_null.csv null_top_freq_q95), the evidence that the binom rule is too lenient. The same 2,000 resamples serve every k (config.yaml bootstrap.seed 2026). Margin of the observed largest step over the null 95% quantile: 0.134 / 0.034 / 0.065 / 0.047 for k = 5 / 10 / 13 / 20 (over the null mean 0.349 / 0.219 / 0.223 / 0.168; null maximum 0.506 / 0.470 / 0.456 / 0.385; r943_e4_drop_vs_null.csv), hence "narrowly for k >= 10" in the Conclusion (review 2026-11-24). Selection frequencies are given as proportions since the review: 98.5% = 0.985 (minimum freq_k10 of the seven, 0.9855), feature 49 at 0.041 / 0.652 / 0.792 / 0.9155, all others at k = 10 at most 0.2825 (feature 67; e4_frequency.csv). -->

Table 1 gives the cross-validated area under the receiver operating characteristic curve (AUC). With L2 logistic regression the honest stable set at *k* = 10 beat all 150 features by 0.063 (*p* = 0.038), whereas the 17 features above the perm drop at *k* = 20 gained only 0.028 (*p* = 0.30). Leaky selection overstated the AUC for the unstable set at *k* = 5 (+0.034, *p* = 0.016) and the weak 17-feature set (+0.069, *p* = 0.0006), but not at *k* = 10 (−0.001). Stability selection never beat the top-*m* filter significantly (−0.033 to +0.002), and the stable set lost to the block in all three models: by 0.030 with L2, 0.035 with QDA and 0.017 with the GMM (*p* = 0.018, 0.0006 and 0.009) [R320].

Block features outside the stable set carry signal in their variance. Six of the 13 are outside it (top-10 frequencies 0.002–0.28), although the perm rule admits 67 at *k* ≥ 13, and 8 and 26 at *k* = 20 (Figure&nbsp;1b). Between the two label-free GMM components, features 15, 30 and 64 have standardised mean differences of about 0, while features 8, 64 and 67 reach log variance ratios of −1.0, +0.63 and +0.83. Alone (6.4 features per fold), they gave QDA 0.917 and the GMM 0.932, close to the best linear models on the block (0.932–0.938) [R249]. Conversely, the second most stable feature, 17, is an exact linear combination of the block (label-free *R*² = 1 − 3 × 10⁻¹⁴) and added nothing (QDA +0.0004, *p* = 0.44) [R320].
<!-- Sources of this paragraph (r320_feature_table.csv, e4_stable_sets.csv, r320_usefulness.csv; updated in the review 2026-11-24): the six block features outside the top-10 stable set are 67, 8, 26, 64, 30 and 15 (top-10 frequencies 0.2825, 0.1735, 0.096, 0.023, 0.013, 0.0015; ranks by |corr| 9, 10, 17, 33, 44, 85). Still moved out of the text: of the six, features 15, 30 and 64 stay below chance at k = 10. Back in the text since the review: under the perm rule feature 67 is above the drop at k = 13 and 20, features 8 and 26 at k = 20 (e4_stable_sets.csv; visible in Figure 1b), so "never stable" holds only for the top-10 / frequency >= 0.5 set (research/reports/verify_ext.md section 3, E4 item 5). "Alone" = the block minus the in-fold stable set, 6.4 features per fold on average (r320_usefulness.csv block_minus_stable; 5-7 per fold, reports/tables/r320_infold_sets.csv). Best linear models on all 13 block columns (back in the text since the review): 0.932 with the block chosen in each fold (L1 logistic, r200_l1_block), 0.938 with the train+test block (L2 logistic, r207_l2_block_tt_wide) (R249, reports/tables/r249_comparison.csv); the six lie well above L2 logistic regression with C = 0.1 on the block, 0.862 (Table 1). -->

![Top-k selection frequency and the high-variance block](../figures/ext_e4.png)

*Figure 1. Top-k selection frequency over 2,000 bootstrap resamples. Encodings are in the panel titles; numbers in (b) name features. Run 20261022-071414_e4_feature_stability; block [R320].*
<!-- Figure sources moved out of the caption (2026-11-24): e4_frequency.csv, e4_drop.csv and null_sorted_curves.csv of the E4 run; block positions from reports/tables/r320_feature_table.csv (R320). Caption shortened in the review 2026-11-24 to what the panel titles do not say; the encodings it no longer repeats: (a) sorted top-k frequency for k = 5, 10, 13, 20 on a log rank axis and square-root frequency axis, dots m* under the perm rule, grey 95% envelope of the 50 permuted-label curves for k = 20, right-hand lines the chance rate k/150; (b) positions of the 13 label-free high-variance block features (ties stacked), shaded ranks 1 to m*, ticks the first rank below chance (k/150). -->

| Feature set | Size | L2 logistic | QDA | Transductive GMM |
|---|---:|---:|---:|---:|
| All 150 features | 150 | 0.769 ± 0.018 | n/a (*n* < *p*) | 1.000 |
| Stable set (top-10, frequency ≥ 0.5) | 7.6 | 0.832 ± 0.005 | 0.961 ± 0.006 | 0.983 ± 0.003 |
| Above the perm drop, *k* = 20 | 17 | 0.797 ± 0.015 | – | – |
| High-variance block | 13 | **0.862 ± 0.007** | **0.996 ± 0.001** | **1.000 ± 0.000** |
| Block minus stable set | 6.4 | 0.765 ± 0.011 | 0.917 ± 0.008 | 0.932 ± 0.031 |

*Table 1. Cross-validated AUC, mean ± standard deviation over the 20 CV repeats. Size: mean number of features per training fold; n/a: not run (per class, n < p); –: not evaluated. Runs as in Figure&nbsp;1 and the follow-up run [R320]; GMM on all 150 features [R003].*
<!-- Moved out of the caption and the text (2026-11-24): e4_usefulness.csv rows all150, honest_stable (k = 10) and honest_drop_top_m (k = 20); the stable and drop sets are re-chosen in each training fold, hence the fractional sizes; block plus stable set (union, r320_usefulness.csv block_or_stable): 0.860 / 0.996 / 1.000 for L2 / QDA / GMM, no gain over the block; GMM on all 150 features = run 20261022-050732_r003_gmm2_all; paired differences: e4_paired.csv (L2) and r320_paired.csv (stable - block); split SE of the +0.063 difference 0.004 (e4_paired.csv, k = 10 honest_stable vs all150); naive - honest = -0.001 at k = 10 because the in-fold sets match the full-sample set (mean Jaccard index 0.94, e4_infold_selection.csv); honest top-m uses m = size of the full-sample set (e4_usefulness.csv). Review 2026-11-24: the text calls selection on all 250 rows "leaky", as E3 and the core do; the archive's variant names stay naive_* (e4_usefulness.csv, e4_paired.csv). The caption names runs instead of files (sources above). n/a: QDA on all 150 features was not run in R320 (r320_usefulness.csv has no such row; per class about 90 and 110 training rows per fold against 150 features); QDA with Ledoit-Wolf covariances on all 150 features was run elsewhere (R249, r249_comparison.csv r208_qda_all_lw, 0.875). -->

## Conclusion

**Stability and the sharp drop.** The |corr| ranking is stable only at its top: seven features are selected far above the chance rate *k*/150 for every *k* and most others below it, and the sorted frequency drops sharply, more steeply than 49 of 50 permuted-label curves (narrowly for *k* ≥ 10). The drop never splits the seven: it lies below all of them at *k* = 5, where they share five places, and below them and feature 49 for *k* ≥ 10. Which weaker features also clear the envelope depends on *k*: the perm rule adds nine features with |corr| 0.13–0.18 at *k* = 20 (*m*\* = 17). The gap and half rules stay at 7–9, but half admits noise, and binom (up to 28) fails because every resample reproduces a noise feature's large sample correlation.

**Stably selected versus worth keeping.** For a linear model the stable features are worth keeping (+0.063 over all 150), whereas the 17 above the perm drop at *k* = 20 add little (+0.028, *p* = 0.30). Every stable set, however, equals the top-*m* set by |corr|: stability measures how reproducible the marginal correlation is, not whether a feature carries information. Block features outside the top-10 stable set carry signal in their variance, and the stable feature 17 is redundant. Stability and usefulness therefore coincide only when the signal lies in the class means; usefulness depends on the model and must be judged by honest CV. For our covariance-based final model, the transductive GMM, the |corr| ranking is the wrong guide.
