# AI review round 1 of the 2026-11-01 drafts (archived reviewer output)

来源：工作流 `wf_f9aee159-d9b`（`ssc-deliverables-round1`，2026-11-01）的三个只读审稿代理 review_numbers、review_claims、review_rubric（Claude Code，Claude Opus 5.5）。它们审的是 `reports/report_core.md` 10:13:50 JST 的版本（review_claims 另外审了四份扩展草稿、贡献表、`reports/reflections/FACTS.md`、`video/PLAN.md`、`docs/COMPETITION_RULES.md`）。下面的行号都指那个版本，之后 core_fixer 改过报告，行号已经变了。审稿意见原文（英文）按原样存档，没有改动；每条后面的 **Resolution** 是处理结果：core_fixer 代理的说明，或主会话（2026-11-01，工作流结束后）的说明。

编号：N = numbers，C = claims，U = rubric，按审稿人给出的顺序。工作流的最终检查（`research/reports/deliverables_round1.md` §6）写的时候这份原文还没有存档。

| Review | Issues | Applied | Not applied |
|---|---|---|---|
| Numbers review (review_numbers) | 22 | 22 | – |
| Claims review (review_claims) | 36 | 36 | – |
| Rubric, format and readability review (review_rubric) | 19 | 18 | U12 |

C26 was applied in part (one clause rejected, reason given). Items in files outside `reports/report_core.md` were applied by the main session.

## Numbers review (review_numbers)

**Reviewer summary.**

I checked reports/report_core.md number by number against the archive, along with Table 1, Figures 1 and 2, core_journey/core_boundary/core_label_budget.{md,csv} and core_manifest.json. Sources used: r440_core_numbers.csv (242 rows: 235 verified yes, 7 no), registry/submissions.csv, beliefs.md, decisions.md, experiments.csv, the summary.json / notes.md / fold_scores.csv of every cited run, and the cited reports/tables files. Some things I recomputed myself: Hanley-McNeil SEs and z-scores under both SE conventions, the '30 of 139' count from fold_scores.csv, the pair counts and the 15x ratio.

What checks out:
- All 31 run folders named in the report exist, along with every file, script, config and README section it references.
- Table 1 is byte-identical to core_journey.md.
- The manifest input hashes match the current files, so the core figures and tables are not stale.
- About 170 numbers match their sources at the printed precision. This covers S001-S006 CV, public scores and sha256; every z value; the E1 MDDs and 64%; E3; R900; R901 0.499; R002 0.000/0.991; R101 +851 and >1e5; R154; R920; R921; R922; R936 diffs and gains; R404 boundary AUCs and 0.99910; R441 budgets and 400/400; the R249 ablations; FM scores; R320; the 214 runs; 242/235/7; and the package versions.

No critical or high problems. Five medium problems are sentences whose wording the cited numbers contradict:
- 'no candidate beats it on the private split': candidates beat S003 in 6-21% of replica draws.
- B004 'at most about 0.16': R922 gives 0.17 under one prior and 0.51 under another.
- 'QDA on the label-free block about 40': that is the fixed-reg row that appears only in the table; Figure 2a shows 80 for the label-free-block QDA.
- 'not in means': R154 gives a means-only AUC of 0.947.
- The R205 drop from 0.99998 is attributed entirely to selection; R205's own notes say part of it comes from a coarser grid.

The rest are low-severity fixes: rounding, replica counts, ambiguous 'private sd', prior-dependent gains, missing run ids or sources, two notes inside HTML comments, and a git-commit wording.

### N1 (medium)

- **Location:** Section 4 'Why this model', item (iii) (line 99): 'On 222 replicas no candidate beats it on the private split; paired differences ...'
- **Problem:** R936 reports MEAN paired differences. Per replica draw, candidates do beat S003: P(beat) is 0.21 for ext-grid QDA, 0.21 for S006, 0.18 for Bayesian QDA, 0.08 for r210 and 0.06 for S005; the GMM variants beat it in 0.45-0.50 (coin flips on ties). 'No candidate beats it' is only true of the mean.
- **Fix:** Replace with: '(iii) On 222 replicas no candidate has a higher mean private AUC; mean paired differences −6.3·10⁻⁵ (extended-grid QDA), −6.9·10⁻⁵ (S006), −1.16·10⁻⁴ (Bayesian QDA), −1.68·10⁻⁴ (log-loss-tuned QDA), −2.57·10⁻⁴ (S005) (R936).'
- **Evidence:** reports/tables/r936_private_weighted.md, rows tn0 / csv8 / weighted_r156, column p_beats_s003: x_qda_block_nll_ext 0.2074, r209 0.2105, r255_bqda_eb_hv 0.1771, r210 0.07947, r201 0.05726, r105/r155/r102 0.5045/0.4571/0.4509. experiments/20261022-172146_r936_final_selection/notes.md: 'BQDA / self-training beat S003 in 18 / 21%'.
- **Resolution:** core_fixer: applied. Now: the other candidates all have a lower expected private AUC, mean paired differences (R936), but each of the five beats S003 in 6-21% of replica private splits. Archive: r936_private_weighted tn0/csv8 p_beats_s003 0.057-0.2105.

### N2 (medium)

- **Location:** Section 3 'What we learned' (line 86): 'with label noise modelled its probability is at most about 0.16 (R922)'
- **Problem:** 0.16 is the flat-prior value only. R922 gives 0.17 under the default-aware prior and 0.51 under a log-uniform prior, so 'at most about 0.16' is contradicted by the archive. B004 is miscalibrated either way, because every value is far below the registered 70%.
- **Fix:** Replace with: 'with label noise modelled its probability is 0.14–0.17 under three priors on the flip rate (0.51 under a log-uniform prior; R922)'
- **Evidence:** experiments/20261022-092146_r922_label_noise_private/summary.json, posterior_by_prior.*.P_B004_private_ge_0.999: uniform_0_0.1 0.1626, uniform_0_0.03 0.1630, gen_grid4 0.1366, default_aware 0.1726, loguniform_1e-4_0.1 0.5128. notes.md Conclusion: '0.14-0.17 ... (0.51 only under a log-uniform prior)'.
- **Resolution:** core_fixer: applied, resolved with C6. Now: 0.16 under a flat prior (0.14-0.17 under three other priors, 0.51 under a log-uniform one; R922). Conflict: N2 said 'three priors'; R922 posterior_by_prior shows four priors in 0.137-0.173, so the archive's flat prior plus three others is used.

### N3 (medium)

- **Location:** Section 5 'What the unlabelled rows add' (a) (line 113): 'self-training needs about 20, QDA on the label-free block about 40, and strictly supervised QDA 80 (fixed regularisation) to 120'
- **Problem:** The '40' is the R206 model with reg fixed at 0.003, a value taken from R201's nested choice (R440 L3 note). That row appears only in core_label_budget.md, marked *, and is not in Figure 2(a). The label-free-block QDA that Figure 2(a) does show (tuning-free Bayesian QDA r255) needs 80, and S002's QDA (reg 0.1) needs 120 by mean (80 by median) in R441. So the text and the figure next to it disagree, and the fixed-reg caveat the sentence applies to strict QDA is missing here.
- **Fix:** Replace with: 'self-training needs about 20, QDA on the label-free block 40 with reg fixed at 0.003 (table only) to 80 (tuning-free Bayesian QDA), and strictly supervised QDA 80 (fixed regularisation) to 120 (nested, the S005 procedure).'
- **Evidence:** reports/tables/core_label_budget.md: 'QDA reg 0.003, train+test block (R206 model) *' first ≥0.99 = 40; 'Bayesian QDA, train+test block (r255)' = 80. reports/tables/r440_label_budget_first_hit.md: eq206_qda_tt_r01 r441_first_mean_ge_099 = 120, first_median = 80. reports/tables/r440_core_numbers.csv L3 note.
- **Resolution:** core_fixer: applied, resolved with C16 and U4. The text describes the plotted rows (2 / 20 / 80 / 120-160, S005 120). The table-only 40 / 80 are quoted only as 'reg fixed at 0.003, a value chosen with all 250 labels (R201) ... (table only)'. Archive: core_label_budget.md, r2_audit 2.3.

### N4 (medium)

- **Location:** Section 3, S002 bullet (line 81): 'The classes differ in covariance, not in means.'
- **Problem:** R154 shows the class means differ in the 4 active informative columns, and a means-only Gaussian rule reaches population AUC 0.947 (covariance-only 0.9998). The same sentence quotes LDA, a means-only model, at 0.878. The signal is mainly in the covariance, not only there.
- **Fix:** Replace with: 'The classes differ mainly in covariance (R154: a covariance-only rule reaches AUC 0.9998, a means-only rule 0.947).'
- **Evidence:** experiments/20261022-065605_r154_class_structure/summary.json: auc_cov_only_same_mean 0.9998031, auc_mean_only_pooled_cov 0.9470353, diff_unscaled_active [-1.3758, 1.338, 1.6233, 1.3378]. notes.md: 'almost all separability is in the covariance difference'.
- **Resolution:** core_fixer: applied, merged with C1 and U8. Now: the signal is mainly in the class covariances; covariance-only and means-only rules reach 0.9998 and 0.947 (R154).

### N5 (medium)

- **Location:** Section 4 'Why this model', item (iv) (line 99): 'tuned supervised models fall from 0.99998 to 0.9995 ± 0.0006 under nested selection; R205'
- **Problem:** 0.99998 is r210's CV (R440 P21). R205 nests a 46-configuration menu whose QDA reg grid stops at 0.01, and its own Conclusion says the gap to the best single run (r201, 0.9999) 'mixes winner's-curse optimism with the coarser grid'. The sentence attributes the whole 4.5·10⁻⁴ drop to selection. The numbers themselves are right.
- **Fix:** Replace the parenthesis with: '(selecting among a 46-configuration supervised menu inside the CV scores 0.9995 ± 0.0006, against 0.9999–0.99998 for the best single runs, partly because that menu's reg grid is coarser; R205; Varma & Simon, 2006)'
- **Evidence:** experiments/20261022-072747_r205_select_supervised/summary.json: n_configs 46, auc_mean 0.999548, auc_std_repeat 0.000623, chosen reg_param 0.01 in 95 folds. notes.md Conclusion. research/reports/verify.md C3: 'Quote 0.9995–0.9999 after model selection'. R440 P17 (r201 0.999903), P21 (r210 0.999984).
- **Resolution:** core_fixer: applied. Now: choosing among 46 supervised configurations inside the CV gives 0.9995 ± 0.0006, against 0.9999-0.99998 for the best single runs, partly because that menu's grid is coarser (R205 notes).

### N6 (low)

- **Location:** Section 4 'Final selections' (line 101): '... about 1/20 of the private sd'
- **Problem:** The 1/20 uses R936's replica-to-replica sd of S003's private AUC at a fixed noise level (3.1·10⁻⁴). Two paragraphs later the report gives the private uncertainty from R922 (90% 0.9923–0.9996, predictive sd 0.0023); against that sd the gain is about 1/160. The archive itself quotes 1/20 (R936), 1/30 and 1/40 (verify.md) and '< 1/100' (R425). Without naming the sd, the phrase is ambiguous.
- **Fix:** Replace 'about 1/20 of the private sd' with 'about 1/20 of the per-noise-level sd of a private score (3.1·10⁻⁴, R936)'
- **Evidence:** experiments/20261022-172146_r936_final_selection/notes.md ('S003 private 0.9977 (sd 3.1e-4) ... 1/20 of the private sd'). reports/tables/r936_private_weighted.md tn0 csv8 s003 private_auc_sd 0.0003102. R922 summary pred_private_sd 0.002345. R425 notes: '< 1/100 of the private sd'. research/reports/verify.md C9 ('≈ 1/30') and line 302 ('1/40').
- **Resolution:** core_fixer: applied, combined with N7. Now 'at most 1/15 of the sd of a private score at a given noise rate (3.1·10⁻⁴, R936)'. The archive gives 2.0e-5/3.1e-4 = 1/15 under the flat prior, so 1/20 holds only under R936's weights.

### N7 (low)

- **Location:** Section 4 'Final selections' (line 101): 'A second selection adds at most 1.45·10⁻⁵ in expected private AUC (S006; S005 4.1·10⁻⁶; R936)'
- **Problem:** These gains use R936's R156 level weights. The expected private score in the next paragraph uses R922's flat prior, and under that prior R425 gives +2.0·10⁻⁵ (S006) and +7.0·10⁻⁶ (S005). The 'at most' bound depends on the weighting.
- **Fix:** Replace with: 'A second selection adds at most 1.45·10⁻⁵ in expected private AUC (S006; S005 4.1·10⁻⁶; R936), or 2.0·10⁻⁵ and 7·10⁻⁶ under R922's flat prior (R425)'
- **Evidence:** experiments/20261022-191747_r425_flip_grid_reweight/summary.json: private.R922_flat_combined gain_second_r209 2.02e-05, gain_second_r201 7e-06 (r936_R156: 1.45e-05, 4.1e-06). reports/tables/r425_private.md.
- **Resolution:** core_fixer: applied. 1.45·10⁻⁵ / 4.1·10⁻⁶ under R936's weights, 2.0·10⁻⁵ / 7·10⁻⁶ under the flat prior (R425 summary.json).

### N8 (low)

- **Location:** Section 2 'Final-selection rule' (line 56): 'because the sd of their differences (3–6·10⁻⁴) exceeds any true gap'
- **Problem:** Rounding. The archived predictive sds of the public difference to S003 are 3.3·10⁻⁴ (S006) and 6.6·10⁻⁴ (S005); D010 item 5 writes 3.2–6.5e-4. 6.6 rounds to 7, not 6.
- **Fix:** Replace '(3–6·10⁻⁴)' with '(3.3–6.6·10⁻⁴, R936)'
- **Evidence:** reports/tables/r440_core_numbers.csv X13 0.000330508 and X15 0.000664785 (r936_public_abc.csv). registry/decisions.md D010 item 5 ('3.2–6.5e-4').
- **Resolution:** core_fixer: applied as '(3.3-6.6·10⁻⁴, R936)'. Following U2-C4/C5, the sentence moved from §2 into the S005/S006 bullet.

### N9 (low)

- **Location:** Section 1 'Why it is hard' (line 38): 'and 0.0015 near 0.999 (Hanley & McNeil, 1982)'
- **Problem:** Rounding. The Hanley-McNeil SE for 395 rows (198/197) at AUC 0.999 is 0.0016 (0.001594). 0.0015 holds only from about 0.9991 up (0.00149 at 0.99913).
- **Fix:** Replace 'and 0.0015 near 0.999' with 'and 0.0016 at 0.999'
- **Evidence:** reports/tables/core_journey.csv, S003 hm_se_at_expected = 0.0015935 (AUC 0.999). R440 D8: 0.00148629 at 0.99913. Recomputed with ssc.validation.auc_standard_error: 0.999 → 0.001594, 0.9991 → 0.001512.
- **Resolution:** core_fixer: applied as '0.0016 at 0.999'; the §2 range now reads 'SE 0.0016-0.02'. Recomputed: 0.001594.

### N10 (low)

- **Location:** Section 4 'Run and provenance' (line 97): 'on a Windows laptop the mixture agrees to about 10⁻¹¹.'
- **Problem:** The archived maximum difference is 4.4·10⁻¹¹, and it changes 1 of the 19,750 8-decimal values, so the file hash differs on the laptop. 'About 10⁻¹¹' understates this and hides the hash consequence.
- **Fix:** Replace with: 'on a Windows laptop the scores differ by up to 4.4·10⁻¹¹, which changes 1 of the 19,750 8-decimal values (R910).'
- **Evidence:** experiments/20261022-091211_r910_s003_gmm2_hv/notes.md ('differs by up to 4.4e-11 ... changes 1 of 19,750 values of the %.8f submission file'). research/reports/verify_repro.md lines 108-109. configs/final_s003_gmm2_hv.yaml hypothesis.
- **Resolution:** core_fixer: applied. 'differ by up to 4.4·10⁻¹¹, which changes 1 of the 19,750 8-decimal values (R910)'.

### N11 (low)

- **Location:** Section 5 'Refining the mixture' (line 117): '... on 40 mixture replicas (R103); ... a mixture on all 150 features 3.4·10⁻⁵ ...'
- **Problem:** R103 ran the all-150 (142-dim) mixture on 8 replicas only, not 40.
- **Fix:** Replace 'a mixture on all 150 features 3.4·10⁻⁵' with 'a mixture on all 150 features 3.4·10⁻⁵ (8 replicas)'
- **Evidence:** reports/tables/r103_replica_summary.csv: G0, all150, expected_auc_under_bayes, n_rep 8, mean_diff_vs_s003 −3.39e-05. research/r103_replica_bootstrap.py line 43 ('--all150-reps', default 8).
- **Resolution:** core_fixer: applied, '(8 replicas)'.

### N12 (low)

- **Location:** Section 5 'Feature stability' (line 119): '*Feature stability* (E4): ... QDA on them scores 0.961 against 0.996 on the block (p = 0.0006).'
- **Problem:** The E4 run evaluates only L2 logistic regression. The QDA numbers come from the follow-up run R320, which the body never cites (it appears only in the HTML comment).
- **Fix:** Replace '*Feature stability* (E4):' with '*Feature stability* (E4, R320):'
- **Evidence:** experiments/20261022-074624_r320_e4_stable_vs_block/usefulness.csv: qda stable 0.96111, qda block 0.99606. paired.csv: qda stable vs block corrected_p 0.000643. experiments/20261022-071414_e4_feature_stability/usefulness.csv has logreg variants only.
- **Resolution:** core_fixer: applied, '(E4, R320)'.

### N13 (low)

- **Location:** Section 5 'What the unlabelled rows add' (b) (line 113): 'Bayesian QDA with a normal-inverse-Wishart prior (Murphy, 2007) 0.678, supervised QDAs 0.66–0.70 (R404)'
- **Problem:** Incomplete. 0.678 is the train+test-block Bayesian QDA. The labels-only Bayesian QDA, plotted in Figure 2(b), scores 0.652, below the quoted range. R404 also has the in-fold QDA-LW at 0.593.
- **Fix:** Replace with: 'Bayesian QDA with a normal-inverse-Wishart prior (Murphy, 2007) 0.652–0.678, supervised QDAs 0.66–0.70 (R404)'
- **Evidence:** reports/tables/core_boundary.md: 'Bayesian QDA, in-fold block (r255, labels only)' 0.652. reports/tables/r404_boundary.md: r255_bqda_eb_trainonly 0.6515, r208_qda_block_lw 0.5932.
- **Resolution:** core_fixer: applied as 'Bayesian QDA 0.652-0.678, label-only QDAs 0.66-0.70 (R404)'.

### N14 (low)

- **Location:** Section 5 'What the labels alone can do' (line 107): 'no other model family trained on the 250 labels came near QDA on the block (S005: 0.9999)'
- **Problem:** This is true only inside the R249 supervised track. TabICL v2, which also uses only the 250 labels, scores 0.9989 on the block, as the report itself states three paragraphs later. R440 refuted the same scope error in sup.md (P31).
- **Fix:** Replace with: 'no other supervised model family in this search came near QDA on the block (S005: 0.9999)'
- **Evidence:** reports/tables/r440_core_numbers.csv F4a: r251_tabicl_hv 0.998917 (20261022-074643_r251_tabicl_hv); P31 marked refuted for the analogous project-wide claim.
- **Resolution:** core_fixer: applied. 'In the supervised track (R249 ...), no other model family came near QDA on the block'.

### N15 (low)

- **Location:** Use of generative AI (line 130): 'an audit of 242 report numbers (R440: 235 confirmed, 7 flagged)'
- **Problem:** R440 marks 235 claims 'verified: yes', but only 216 have status 'confirmed'; the others are superseded, registry-only, replicated, or 'pending' for the S005/S006 public scores. The 242 claims were also taken from the research reports before this draft, and many numbers in this report are not among them (R101, R103, R404, R425, R320, the E1 hold-out/5-fold/5x10 MDDs, the S005/S006 public scores).
- **Fix:** Replace with: 'an audit of 242 numbers from our research notes (R440: 235 verified, 7 flagged)'
- **Evidence:** reports/tables/r440_core_numbers.csv: 242 rows, verified yes 235 / no 7, status 'confirmed' 216; S005d and S006d status 'not uploaded yet'. research/reports/r2_audit.md §3.1 ('242 claims: 235 verified, 7 not').
- **Resolution:** core_fixer: applied, merged with C10 and U17. '... an audit, before drafting, of 242 numbers planned for this report (R440: 235 verified; the 7 that failed are corrected or not used)'. I checked that the 7 'no' rows are either corrected or not used.

### N16 (low)

- **Location:** Section 3, S003 bullet (line 82): 'Its public error (7.7·10⁻⁴) is 15 times the model's own Bayes error (R101)'
- **Problem:** The ratio is right (14.8), but 'Bayes error' here means 1 − Bayes AUC under the fitted mixture (5.2·10⁻⁵), not the Bayes error rate (0.33%, R154). Against 0.33% the ratio would be 0.23.
- **Fix:** Replace with: 'Its public error (7.7·10⁻⁴) is 15 times the fitted mixture's own Bayes shortfall (1 − AUC = 5.2·10⁻⁵, R101)'
- **Evidence:** experiments/20261022-064201_r101_bic_model_choice/summary.json: expected_auc_ref_self 0.999948; (1−0.99923)/(1−0.999948) = 14.8. R154 bayes_error_rate 0.003265 (R440 G23).
- **Resolution:** core_fixer: applied, merged with C5 and U9. 'shortfall 1 − AUC = 7.7·10⁻⁴ is 15 times what the fitted mixture predicts for itself with clean labels (5.2·10⁻⁵, R101)'.

### N17 (low)

- **Location:** Section 4 'Run and provenance' (line 97): 'All runs predate the first git commit (D004)'; also Limitations (3) (line 123): 'All runs predate the first git commit'
- **Problem:** The repository has no commit at all ('main' has no commits; git_commit = no-commit in all 214 rows of registry/experiments.csv). 'Predate the first commit' implies a commit exists.
- **Fix:** Replace with: 'No run is tied to a git commit (the repository has no commit yet; D004)'. In Limitations (3): 'No run is tied to a git commit'.
- **Evidence:** git rev-parse HEAD: 'your current branch main does not have any commits yet'. registry/experiments.csv: git_commit = no-commit in 214/214 rows. registry/decisions.md D004.
- **Resolution:** core_fixer: applied. §4 says no run is tied to a git commit (the repository has none yet; D004). Limitation (3) drops the git clause, as U2-C3 asked.

### N18 (low)

- **Location:** Traceability: Section 2 (line 54) '(30 of the 139 runs on the official splits)'; Section 5 foundation-model line (line 118); Section 4 'by the README's config route' (line 97); HTML run-id list (lines 21-31)
- **Problem:** Several numbers name no run or source. The 30/139 count is correct but unsourced; the TabPFN/TabICL scores have no run id; r210, r255 and the README re-run have no run ids in the list. All of these are correct and were recomputed or found in the archive.
- **Fix:** Line 54: '(30 of the 139 runs on the official splits; registry/experiments.csv)'. Line 118: '... TabICL v2 (Qu et al., 2026) 0.993 / 0.999 (all features / block; R251), below QDA'. Add to the run-id comment: 'R251 20261022-073743_r251_tabpfn2_all, 20261022-073935_r251_tabpfn2_hv, 20261022-074251_r251_tabicl_all, 20261022-074643_r251_tabicl_hv; r210 20261022-082047_r210_qda_block_nll; r255 20261022-070200_r255_bqda_eb_hv + 20261022-070202_r255_bqda_eb_trainonly; README route 20261022-212507_final_s003_gmm2_hv'.
- **Evidence:** Recount: 139 rows of registry/experiments.csv with split_sig 5708ce5c3828, 30 with auc = 1.0 in all 100 rows of fold_scores.csv. R440 F1a-F4a. Every listed run folder exists under experiments/. experiments/20261022-212507_final_s003_gmm2_hv/notes.md (100/100 folds, test_pred bit-identical, sha256 prefix 3aae140e5837753f).
- **Resolution:** core_fixer: applied. Added the 30/139 source (registry/experiments.csv) and R251 on the foundation-model line. The HTML run-id list gains R251 (4 runs), r201, r209, r210, r255 (2 runs), R940 and the README route run.

### N19 (low)

- **Location:** Section 1 (line 36): 'The public leaderboard (LB) uses about 2% of the test rows (≈ 395)'
- **Problem:** R440 flagged the 395 rows (D5) as 'external (not archived)'. The 2% has since been archived (D014), but the report does not cite that source.
- **Fix:** Replace with: 'The public leaderboard (LB) uses about 2% of the test rows (≈ 395; competition rule 3)'
- **Evidence:** reports/tables/r440_core_numbers.csv D5, status 'external (not archived)'. docs/COMPETITION_RULES.md line 17: 'The public leaderboard is computed on a small sample (2 %) of the test rows'. registry/decisions.md D014.
- **Resolution:** core_fixer: applied, '(≈ 395; competition rule 3)'.

### N20 (low)

- **Location:** Section 2 'Comparing models' (line 50): 'An improvement needs |difference| > 2 split SE **and** corrected p < 0.05 (`scripts/compare_runs.py`)'
- **Problem:** The written protocol has a third numeric condition, a fold win rate above 0.5, which the report leaves out. compare_runs.py prints only the first two conditions.
- **Fix:** Replace with: 'An improvement needs |difference| > 2 split SE, corrected p < 0.05 and a fold win rate above 0.5 (docs/VALIDATION_PROTOCOL.md; `scripts/compare_runs.py`)'
- **Evidence:** docs/VALIDATION_PROTOCOL.md line 35 ('... and A wins in most folds (win_rate_fold > 0.5)'). scripts/compare_runs.py line 41.
- **Resolution:** core_fixer: applied, adding 'and a fold win rate above 0.5 (docs/VALIDATION_PROTOCOL.md, scripts/compare_runs.py)'.

### N21 (low)

- **Location:** HTML comment, run-id list (line 22): 'R003 20261022-0506xx_r003_* (8 runs)'
- **Problem:** The pattern does not match the archive. The 8 R003 run ids run from 20261022-050647 to 20261022-050732; five of them are 0507xx.
- **Fix:** Replace with: 'R003 20261022-050647_r003_sup_l1_all to 20261022-050732_r003_gmm2_all (8 runs)'
- **Evidence:** experiments/: 20261022-050647_r003_sup_l1_all, -050653_r003_sup_lda_hv, -050659_r003_sup_qda_hv, -050704_r003_sup_logreg_hv, -050708_r003_gmm2_hv, -050716_r003_gmm2_hv_pca, -050723_r003_gmm4_hv, -050732_r003_gmm2_all.
- **Resolution:** core_fixer: applied, '20261022-050647_r003_sup_l1_all to 20261022-050732_r003_gmm2_all (8 runs)'.

### N22 (low)

- **Location:** HTML comment above Figure 1 (line 62): '... hence +0.6 / +0.4 SE there for S005 / S006.'
- **Problem:** Incomplete. beliefs.md also differs from Table 1 for S002 (+0.6 vs +0.55) and S004 (−1.2 vs −1.35). Text line 83 quotes −1.35 for S004 while B005 says −1.2.
- **Fix:** Replace with: '... hence +0.6 / −1.2 / +0.6 / +0.4 SE there for S002 / S004 / S005 / S006.'
- **Evidence:** registry/beliefs.md B002 (+0.6 SE), B005 (−1.2 SE), B006 (+0.6 SE), B007 (+0.4 SE). Recomputed with record_lb's convention (SE at the observed score, 179/216): +0.61, −1.23, +0.64, +0.41. core_journey.csv z_expected: 0.549, −1.348, 0.418, 0.304.
- **Resolution:** core_fixer: applied, '+0.6 / -1.2 / +0.6 / +0.4 SE ... for S002 / S004 / S005 / S006'.

## Claims review (review_claims)

**Reviewer summary.**

Read-only claims review of reports/report_core.md, the four extension drafts, reports/contribution_table.md, reports/reflections/FACTS.md, video/PLAN.md and docs/COMPETITION_RULES.md. I checked them against verify.md, fix_round1.md, r2_audit/r2_inductive/r2_noise, D007-D014 and the archive (r440_core_numbers.csv, r936/r404/r422 tables, run summaries, experiments.csv).

What the core draft gets right: it already follows most verification corrections. S003's private expectation is given as about 0.997 (90% 0.992-0.9996, flat prior). flip_y is given as MLE 0.003 with support 0-0.015, and zero noise as disfavoured about 5:1. Strictly supervised QDA needs 80-120 labels. It states the CV saturation using the D011 wording, says the replica gaps are model-based, says the design was chosen after seeing the labels, and gives the D007 rule basis.

Problems found:
- One false claim: S002's lesson says the classes differ 'not in means'. The archive says a mean-only rule reaches AUC 0.947.
- One causal claim fails a simple bound: the boundary rows 'cause' the 1e-4 private gaps. Pairs inside the boundary set explain only about 5-11% of those gaps.
- Several overclaims go against the verification round:
  - The public shortfall is called 'label noise, not ranking errors'.
  - B004 is called 'wrong', with 'at most 0.16'.
  - 'No candidate beats it', although candidates win 6-21% of the replica private splits.
  - The hold-out 'orders correctly 64%' is stated as observed, but it is extrapolated.
  - The model is said to have 'no scaler, selector or tuned hyperparameter'.
- Design history is left out:
  - The setup selection rule was replaced by D010.
  - The extended-grid QDA is a post-hoc variant.
  - The label budget uses a reg value chosen with all 250 labels.
  - S006's pseudo-labelling goes beyond D007, and the planned instructor question does not cover it.
- The AI-versus-member split is not stated honestly enough:
  - The AI lines of the extensions are vague and the core report describes the checks in the passive voice, though AI ran all of them.
  - 'sole author' is pre-filled in the contribution table.
  - registry/ai_usage.md has no row for the 2026-11-01 drafting.
- The documents disagree with each other: best linear model 0.898 vs 0.938; QDA label needs 75 vs 40; beliefs 'wrong' vs FACTS 'none refuted'.

No files were edited.

### C1 (high)

- **Location:** reports/report_core.md line 81 (Section 3, S002 bullet), sentence 'The classes differ in covariance, not in means.'
- **Problem:** False scientific claim. The classes differ mainly, not only, in covariance: a rule using only the class means reaches population AUC 0.947, and 4 of the 12 informative columns have large class-mean shifts. The report itself says later that the population limit for linear rules is 0.947, and E4 Table 1 gives L2 logistic on the block 0.862. One upload (S002) and the QDA-vs-LDA CV gap cannot establish 'not in means'.
- **Fix:** Replace with: 'The classes differ mainly in covariance: under the fitted generator a rule that uses only the class means reaches AUC 0.947, one that uses only the covariances 0.9998 (R154, R422).'
- **Evidence:** reports/tables/r422_population.csv: auc_mean_only_pooled_cov_mc 0.947035 (R154 archived), auc_cov_only_same_mean_mc 0.999803; R440 G15/G16 (8 zero-difference informative columns max |z| 2.08, 4 active columns min |z| 47); R440 G22 best linear 0.947; registry/journal.md J2 says '主要在协方差' (mainly).
- **Resolution:** core_fixer: applied, same text as N4.

### C2 (high)

- **Location:** reports/extensions/e1_validation_protocol.md:5, e2_learning_curve.md:5, e3_leakage.md:5, e4_feature_stability.md:5 (visible AI-use line)
- **Problem:** The line 'analysis code and this draft were prepared with an AI assistant (Claude Code) and checked against the archive' understates what the AI did. The AI designed each protocol (e.g. the R300 pair rule, the E4 perm rule), ran every run on Colab and wrote the drafts. The passive 'checked against the archive' hides that the checking was also done by AI agents (R940-R944, fix_round1). fix_round1 §7 asked to unify the four lines toward naming the AI checker; the 10-22 21:37 snapshot shows they were unified to the weaker wording. Extensions are marked individually, and rule 6 requires 'which tool, for what'.
- **Fix:** Replace line 5 in each draft with: '*AI use: Claude Code (Anthropic, Claude Opus 5.5) designed this study, wrote and ran its code on Google Colab, and drafted this text. A second AI verification round re-checked the numbers against the archive (E1: R940-R941; E2: R941-R942; E3: R944; E4: R943). [TODO owner: what you re-ran, rewrote and checked.] Log: `registry/ai_usage.md`, rows 3, 8 and 10.*' (keep only the run ids that belong to the draft).
- **Evidence:** registry/ai_usage.md rows 3, 8, 10 ('Verified by': AI re-check / final checker agent / TODO); research/reports/fix_round1.md §7 and summary ('统一四份草稿的 AI 说明…写明检查者是另一轮 AI 核查'); backups/snapshot_20261022-213717.zip shows the same weak line in all four; FACTS.md §1.4 ('Each draft was written by an AI agent').
- **Resolution:** main session (2026-11-01, after the workflow): applied: line 5 of all four extension drafts now names what Claude Code did (design, code, Colab runs, figures, upload for E3, draft), the AI verification runs (E1 R940-R941, E2 R941-R942, E3 R944, E4 R943), an owner TODO and the ai_usage rows.

### C3 (high)

- **Location:** reports/contribution_table.md lines 68-71 (Contributors column of the E1-E4 rows = 'sole author') and HTML comment item 5 (lines 26-27: 'Contributors "-" (or "sole author" in the extension row)')
- **Problem:** 'sole author' is pre-filled for all four extensions, but every extension draft was written by the AI. A member who keeps the cell makes a false authorship claim in an individually marked part. This contradicts the table's own AI-use legend.
- **Fix:** Replace 'sole author' with '–' in the four extension rows (AI use stays '3, 7, 8, 10'). In comment item 5 replace '(or "sole author" in the extension row)' with '(never "sole author": every extension draft was written by the AI, ai_usage rows 3, 8, 10)'.
- **Evidence:** registry/ai_usage.md rows 3, 8, 10; reports/reflections/FACTS.md §1.4; contribution_table.md legend line 78 ('Claude Code did the operating work (... extension drafts ...)').
- **Resolution:** main session (2026-11-01, after the workflow): applied: 'sole author' replaced by '–' in the four extension rows; comment item 5 now says never 'sole author'.

### C4 (medium)

- **Location:** reports/report_core.md line 113 (Section 5 'What the unlabelled rows add'), last sentence 'These differences on 1% of the rows cause the 10⁻⁴-level private gaps of Section 4.'
- **Problem:** Unsupported causal claim, contradicted by simple arithmetic. The boundary AUC is computed only on pairs inside the boundary set: about (185/2)² ≈ 8.5·10³ of ≈ 9.4·10⁷ private pairs (≈ 9·10⁻⁵). So r201's 0.140 boundary-AUC gap moves the private AUC by only ≈ 1.3·10⁻⁵ of its 2.57·10⁻⁴ gap (≈ 5%); for r209 it is ≈ 8·10⁻⁶ of 6.9·10⁻⁵ (≈ 11%). Most of each gap must come from ordering boundary rows against more confident rows. No pair decomposition exists in the archive; the sentence comes from r2_inductive's 'Text for the report'.
- **Fix:** Replace with: 'These boundary rows are where the procedures differ most; the 10⁻⁴-level private gaps of Section 4 also include how each procedure orders them against the more confident rows nearby.' Quote no causal share unless a pair decomposition is computed from R404's per-replica scores.
- **Evidence:** reports/tables/r404_boundary.csv (n_bnd_rows_mean 184.7; S003 minus r201 0.140, minus r209 0.086); verify.md B1 / R921 (≈ 24 of 9.4e7 private pairs); R936 private differences −2.57e-4 (r201) and −6.9e-5 (r209); research/reports/r2_inductive.md §3, §5.
- **Resolution:** core_fixer: applied by deleting the causal sentence. No pair decomposition is archived.

### C5 (medium)

- **Location:** reports/report_core.md line 82 (Section 3, S003 bullet): 'as predicted ("about 0.999, not 1.0"). Its public error (7.7·10⁻⁴) is 15 times the model's own Bayes error (R101): label noise, not ranking errors.'
- **Problem:** (a) 'Bayes error' here means 1 − the mixture's expected AUC under its own fit (5.2e-5). Elsewhere the archive uses 'Bayes error' for the 0.33% error rate, so the term is ambiguous. (b) 'label noise, not ranking errors' is stated as fact, but the verification round says zero noise is disfavoured only about 5:1 and not excluded, and the report itself says so a few lines later. (c) 'as predicted' hides that B003's stated reason (the ~1% boundary rows) was wrong.
- **Fix:** Replace the two sentences after 'public 0.99923 (first on the public LB at upload)' with: 'B003 had predicted about 0.999, not 1.0, because of the ~1% boundary rows; the score matched, the reason did not. The public shortfall (7.7·10⁻⁴) is 15 times the shortfall expected with clean labels under the fitted mixture (1 − 0.99995 ≈ 5·10⁻⁵; R101), which points to label noise: zero noise is disfavoured about 5:1, not excluded (R922).'
- **Evidence:** experiments/20261022-064201_r101_bic_model_choice/summary.json ('expected_auc_ref_self' 0.999948); R440 G23 (Bayes error rate 0.33%); verify.md B5 (LR(f=0) 0.20, 'disfavoured about 5:1, not excluded'); registry/beliefs.md B003; FACTS.md §4.3(a).
- **Resolution:** core_fixer: applied. Uses the clean-label shortfall wording and 'points to label noise rather than ranking errors'. That B003's stated reason was wrong is now in 'What we learned', and the 'disfavoured 5:1, not excluded' caveat stays there (R922).

### C6 (medium)

- **Location:** reports/report_core.md line 86 (Section 3 'What we learned'): 'Two beliefs were wrong. B004 (...) was miscalibrated: with label noise modelled its probability is at most about 0.16 (R922); we kept B004 and registered B008 (...)'
- **Problem:** (a) Only one belief is named. (b) B004 has not been tested by any leaderboard score (private pending), and FACTS §4.4 says 'No B row was refuted by a leaderboard score'; the two documents contradict each other. (c) 'at most about 0.16' contradicts the verification round: 0.14-0.17 under the reasonable priors, and 0.51 under a log-uniform prior.
- **Fix:** Replace with: 'Two expectations did not hold up. B003 put the gap to 1.0 down to boundary rows; the evidence points to label noise instead (above). B004 (S003's private AUC ≥ 0.999 and within 0.001 of public, 70%) now looks miscalibrated: given the public score its probability is about 0.16 under a flat prior on the noise rate (0.14-0.17 under the other priors we consider reasonable, 0.51 only under a log-uniform prior; R922). The private LB has not tested it yet. We kept B004 unchanged and registered B008 (median 0.997, 90% 0.992-0.9996).'
- **Evidence:** verify.md B8 and §3 C9; R440 G37 (0.162557, flat prior); reports/tables/r922_posterior_by_prior.md; registry/beliefs.md B004 Outcome 'pending'; reports/reflections/FACTS.md line 255.
- **Resolution:** core_fixer: applied. Names both expectations: B003's boundary-row reason (the evidence points to label noise) and B004, which looks miscalibrated and has not been tested by the private LB. Conflict with U10 ('one belief'): the archive (beliefs B003 'confirmed' with a contradicted reason, B004 'pending') supports naming both with that wording.

### C7 (medium)

- **Location:** reports/report_core.md line 99 (Section 4 'Why this model', item (iii)): 'On 222 replicas no candidate beats it on the private split; paired differences ... (R936).'
- **Problem:** Overclaim of replica evidence. On the replica private splits each candidate beats S003 fairly often: r209 21%, extended-grid QDA 21%, Bayesian QDA 18%, r210 8%, r201 6%. The verification round only confirmed that no candidate beats S003 significantly, i.e. that each has a lower expected private AUC. The gaps are also model-based, but the sentence does not say so.
- **Fix:** Replace item (iii) with: '(iii) On 222 replicas every candidate has a lower expected private AUC than S003; paired differences −6.3·10⁻⁵ (extended-grid QDA), −6.9·10⁻⁵ (S006), −1.16·10⁻⁴ (Bayesian QDA), −1.68·10⁻⁴ (log-loss-tuned QDA), −2.57·10⁻⁴ (S005) (R936). Each still beats S003 on 6-21% of the replica private splits, and the gaps are model-based (Limitation 2).'
- **Evidence:** reports/tables/r936_private.csv, group tn0, scores csv8, flip_y weighted_r156, column p_beats_s003: r209 0.210, x_qda_block_nll_ext 0.207, r255_bqda_eb_hv 0.177, r210 0.079, r201 0.057; verify.md C8 ('No candidate beats S003 significantly').
- **Resolution:** core_fixer: applied with N1, adding '(R936; model-based, Limitation 2)'.

### C8 (medium)

- **Location:** reports/report_core.md line 90 ('**Model** (...), with no scaler, selector or tuned hyperparameter:') and line 99 item (iv) ('It has no tuned parameter ... so its CV is no winner's-curse maximum')
- **Problem:** Internally contradictory. Step 1 selects columns (sd > 1.5 × median) and standardises them with train+test statistics, so the model does have a scaler and a selector. The threshold 1.5, K = 2, α = 1 and reg_covar 1e-6 are hyperparameters set by hand, and Limitation (1) says the block and K = 2 were chosen after looking at the labels. R101 also shows reg_covar is consequential. Saying 'no tuned parameter' as the reason for no selection optimism understates the label-informed design.
- **Fix:** Line 90: '**Model** (`configs/final_s003_gmm2_hv.yaml`, class `ssc.transductive.TransductiveGMM`). No step is tuned on labels; the settings below (threshold 1.5, K = 2, α = 1, `reg_covar` 1e-6) were set by hand (see Limitation 1):'. Item (iv): '(iv) None of its settings was chosen by a CV search, and its CV sits at the ceiling of 1, so it carries none of the tuning optimism that takes tuned supervised models from 0.99998 to 0.9995 ± 0.0006 under nested selection (R205; Varma & Simon, 2006). Its design was, however, chosen after looking at the labels (Limitation 1).'
- **Evidence:** src/ssc/transductive.py:43-45 (ratio 1.5 × median std), :54 (B.mean/B.std standardisation), :82-83 (n_init 5, alpha 1.0, reg_covar 1e-6); verify.md D2; experiments/20261022-064201_r101_bic_model_choice/notes.md ('the regularisation is NOT negligible: reg 1e-4 ... re-orders the S003 boundary rows (Kendall 0.77)').
- **Resolution:** core_fixer: applied. Lead now reads 'No setting below is tuned on labels; all were set by hand (Limitation 1)'. (iv) now reads 'None of its settings came from a CV search ... but its design was chosen after seeing the labels (Limitation 1)'. Conflict with U14: C8's wording chosen, since src/ssc/transductive.py selects and standardises columns inside the model.

### C9 (medium)

- **Location:** reports/report_core.md line 40, paragraph heading '**Structure found without labels** (R001, R150).'
- **Problem:** The heading contradicts Limitation (1) and the verification round. R001 plotted std against |corr| with the target ('13 high-variance features hold 9 of the 10 most target-correlated features'), and R002 checked label purity before the block + K=2 design was adopted. The procedure is label-free, but the discovery was not. verify.md D2 asked for the disclosure in Problem formulation, where this heading now says the opposite.
- **Fix:** Replace the heading with: '**Structure that needs no labels** (R001, R150; how the labels influenced its discovery: Section 5, Limitation 1).'
- **Evidence:** verify.md D2 ('confirmed for the code; the design saw labels ... One disclosure sentence in Problem formulation'); research/reports/verify_leak.md:55; experiments/20261022-051627_r001_data_structure/summary.json ('top10_corr_features') and notes.md.
- **Resolution:** core_fixer: applied, merged with U7, as '**Structure in the features** (R001, R150; on the role of the labels see Limitation 1)'.

### C10 (medium)

- **Location:** reports/report_core.md line 130 ('Use of generative AI'), from 'Its output was checked by' to the end; registry/ai_usage.md (no 2026-11-01 row)
- **Problem:** (a) The checks are written in the passive voice, but AI agents designed and ran every one of them (R910, R900, R103/R921/R936, tests/, the verify round, R440); no member check is recorded ('Verified by' is TODO or AI in all 13 rows). (b) 'an audit of 242 report numbers' overstates coverage: R440 audited numbers planned on 10-22, before this 11-01 draft. Several printed numbers are not among them (e.g. '30 of the 139 runs', 'K = 3: +851', '15 times', '2.4·10⁻⁸', 'tied 0.23'). (c) 'full log' points to a log with no row for the 2026-11-01 drafting round. (d) The paragraph says nothing about what members did. (e) It does not say that the AI also wrote every pre-registered belief, proposed every idea and wrote the rule paraphrase.
- **Fix:** Replace with: 'It also proposed every recorded idea (I001-I014), wrote every pre-registered belief (B001-B008), and wrote the rule paraphrase and licence check (D014). Its output was checked by bit-exact re-runs (R910), permutation tests with positive controls (R900), replicas with known labels (R103, R921, R936), unit tests (`tests/`), a separate AI verification round, and an audit of 242 numbers planned for this report, done before it was drafted (R440: 235 confirmed; the 7 that failed are corrected or not used). AI agents designed and ran all of these checks. [TODO members: what each of you re-ran, rewrote and checked, and which decisions you confirmed.] Under rule 6 every member must be able to explain every part (log: `registry/ai_usage.md`).' Then append an ai_usage.md row dated 2026-11-01 for this drafting round.
- **Evidence:** registry/ai_usage.md (13 rows, all dated 2026-10-22, Verified by TODO/AI); reports/contribution_table.md line 78 ('drafts of 2026-11-01 ...: row TODO'); R440 run 20261022-184020_r440_core_numbers vs draft date 2026-11-01 in report_core.md comment; reports/tables/r440_core_numbers.csv (no rows for the numbers listed); registry/ideas.md (Proposer 'AI (Claude Code)'); registry/beliefs.md (Author 'TODO (AI-assisted)').
- **Resolution:** core_fixer: applied in the report. AI proposed I001-I014, wrote B001-B008, D007, D014 and D010; AI agents designed and ran every check; audit scope corrected; added 'TODO (members): what each of you re-ran, rewrote and checked, and which decisions you confirmed'. The ai_usage.md row is in not_done. | main session (2026-11-01, after the workflow): ai_usage rows 14-21 appended (append only).

### C11 (medium)

- **Location:** reports/report_core.md line 56 (Section 2 'Final-selection rule (D010, ...)')
- **Problem:** Design-history omission. The setup protocol (docs/VALIDATION_PROTOCOL.md §5, shipped in the zip) set an advance rule: candidates from a clean git commit, best CV, tolerance band, simplest model. That rule was not applied: 30 runs tie at CV 1.0 and no run comes from a commit. D010, written later after the CV and replica analyses, replaced it. The report presents D010 as the rule decided in advance without saying the original rule was dropped. FACTS §2.2 records the switch, the report does not, and VALIDATION_PROTOCOL.md still says 'draft' with TODO(E1).
- **Fix:** Prepend: 'Our setup protocol (`docs/VALIDATION_PROTOCOL.md` §5) planned to take the simplest model within a CV tolerance band, from a clean git commit. With 30 official runs tied at CV 1.0 and no commit yet (D004), it could not separate the candidates, so D010 replaced it.' Also record the D002 freeze decision and update VALIDATION_PROTOCOL.md §5's status.
- **Evidence:** docs/VALIDATION_PROTOCOL.md §5 and line 3 (draft, TODO(E1)); reports/reflections/FACTS.md §2.2 'Draft selection rule ... not used'; registry/experiments.csv (git_commit 'no-commit' in 214/214); 30 official runs with AUC 1.0 in all 100 folds (fold_scores.csv count); verify.md E7.
- **Resolution:** core_fixer: applied in the report. D010 'replaced our setup protocol's rule (docs/VALIDATION_PROTOCOL.md §5 ...), which failed here: 30 runs tie at CV 1.0 and none comes from a commit (D004)'. The VALIDATION_PROTOCOL / D002 freeze parts are in not_done. | main session (2026-11-01, after the workflow): D015 appended (D002 frozen without change; §5 rule superseded by D010); D002 Status line points to D015; docs/VALIDATION_PROTOCOL.md status line, §2 E1 result, §5 'Superseded' note, §6 test_pred mode.

### C12 (medium)

- **Location:** reports/report_core.md line 46 ('steps that use only features (the block rule, the mixture) may use all 20,000 rows but never see a label (D007)') and line 56 (case 'If the instructor explicitly allows unlabelled test features: S003 + S006'); docs/ASSIGNMENT_CHECKLIST.md line 37 (planned instructor question)
- **Problem:** Rule-interpretation gap. D007 allows only unsupervised steps on test features (scaling, variance rule, PCA, GMM). S006 goes further: it retrains a supervised model on algorithmic pseudo-labels of the test rows, which only D014 extends the rule basis to. The planned question to the instructor asks only about unsupervised steps. A 'yes' would therefore not cover S006, yet D010 case 2 would select S006.
- **Fix:** After '(D007)' add: 'S006 goes further: it retrains on algorithmic pseudo-labels of the test rows, which D014 also counts as competition data.' Change the case 2 sentence to: 'If the instructor explicitly allows both uses of the unlabelled test features (unsupervised fitting and self-training on pseudo-labels): S003 + S006; if only unsupervised use is allowed: the default pair.' Rephrase the checklist question to name both uses.
- **Evidence:** registry/decisions.md D007 Decision (无监督步骤：标准化、按方差选列、PCA、GMM), D010 item 2, D014 item 2; docs/ASSIGNMENT_CHECKLIST.md line 37 ('能否在无监督步骤里用测试集的无标签特征'); docs/COMPETITION_RULES.md §4 item 1; registry/submissions.csv S006 description.
- **Resolution:** core_fixer: applied in the report. Case 2 now reads 'If the instructor explicitly allows the unlabelled test features, including S006's retraining on algorithmic pseudo-labels of the test rows (competition data under D014)'. I did not add a new 'only unsupervised allowed' case, because D010 has none. The checklist question is in not_done. | main session (2026-11-01, after the workflow): checklist A question now names both uses (unsupervised fitting; self-training on pseudo-labels) plus foundation models; D015 item 4 maps each answer to a D010 case.

### C13 (medium)

- **Location:** reports/report_core.md line 56 ('if the instructor forbids them: the extended-grid QDA (r930, not yet uploaded; predicted public 0.99910, R404) + S005') and line 99 item (iii)
- **Problem:** The verification round required disclosure, and it is missing. r930 is a post-hoc variant: its regularisation grid was extended after r210 was seen to pick the grid floor, so its replica lead is a second-level choice. It is also fragile under wrong training labels, which is why S005 is its partner. The report gives neither fact.
- **Fix:** Append after '+ S005': 'r930 is a post-hoc variant: its regularisation grid was extended after the log-loss-tuned QDA r210 was seen to choose the grid floor, so its replica lead over S005 (1.9·10⁻⁴) is a second-level choice. On replicas with flipped training labels it falls 4.7·10⁻⁴ below S003, which the pairing with S005 covers (R404; R936 tn1).'
- **Evidence:** research/reports/r2_inductive.md §7 ('ext is a post-hoc variant ... its selection history must be stated in the report'), Table A1 (+1.94e-4 vs r201), §2 'Choice of the second member' (tn1: ext −4.72e-4 vs S003); verify.md C4; D013 item 1.
- **Resolution:** core_fixer: applied. 'r930 is a post-hoc variant (its grid was extended after r210 picked the grid floor); on the replicas it leads S005 by 1.9·10⁻⁴ but falls 4.7·10⁻⁴ below S003 when training labels are flipped, which S005 covers (R404, R936)'. Archive: r2_inductive §2, r936 tn1 −4.718e-4.

### C14 (medium)

- **Location:** reports/report_core.md line 48 (Section 2 'Why this scheme (E1)'): 'A hold-out orders a 0.01 pair correctly only 64% of the time'
- **Problem:** p_correct = Φ(0.01/sd) is a normal-shift extrapolation, not an observed rate. The pair actually studied is about 0.004 apart (E1 says so itself). The 80%-power caveat from verification F1 is also missing: at 80% power even 5 × 20 is inconclusive.
- **Fix:** Replace with: 'Under a normal approximation, a hold-out would order a 0.01 pair correctly only 64% of the time (the pair actually studied is about 0.004 apart, R940); more folds do not help, and leave-one-out gives no per-fold AUC. At 80% instead of 50% power even 5 × 20 is inconclusive (MDD 0.0086 [0.0071, 0.0107]).'
- **Evidence:** verify.md F1 (80% power 0.0086 [0.0071, 0.0107]) and F2 ('p_correct at 0.01 is an extrapolation'); reports/extensions/e1_validation_protocol.md:48 and :56.
- **Resolution:** core_fixer: applied. 'Under a normal approximation ... (the pair studied is about 0.004 apart, R940)', plus the 80%-power MDD 0.0086 [0.0071, 0.0107].

### C15 (medium)

- **Location:** reports/report_core.md line 107 (Section 5 'What the labels alone can do')
- **Problem:** (a) Under a heading about labels alone, 'best penalised linear model 0.938' is r207_l2_block_tt_wide, whose block is chosen on train+test features; the verification round counts such runs as using unlabelled test features (group B). The best in-fold linear model is 0.932, and the LDA upper end 0.878 also uses the train+test block. (b) 'Across 587 supervised configurations (R249, re-checked in R440)': R440 re-computed only the 64 rows of the comparison table, and the track also contains semi-supervised runs.
- **Fix:** Replace the first sentence's list with: 'Across the 587 configurations of the supervised track (R249; its 64-row comparison table re-computed in R440), no other model family trained on the 250 labels came near QDA on the block (S005: 0.9999): L1 logistic 0.847 (S001), LDA 0.86, Gaussian naive Bayes 0.87, best penalised linear model 0.932 (0.938 with the block chosen on train+test features), RBF-SVM 0.951, kNN 0.902 (Hastie et al., 2009), Gaussian-process classification 0.976 (Rasmussen & Williams, 2006), tree ensembles ≤ 0.914 (...).'
- **Evidence:** reports/tables/r249_comparison.csv (r207_l2_block_tt_wide 0.9383 'block(train+test highvar...)'; r200_l1_block 0.9321 'block(in-fold highvar)'; r200_lda_block 0.8592); registry/journal.md J5 correction (group B); verify.md G4; research/reports/r2_audit.md §3.1.
- **Resolution:** core_fixer: applied. Uses 'R249: 587 configurations; its 64-row comparison table re-computed in R440', LDA 0.86, best penalised linear 0.932 (0.938 with the block chosen on train+test features). Archive: r249_comparison.csv.

### C16 (medium)

- **Location:** reports/report_core.md line 113 (Section 5 (a) label budget): 'QDA on the label-free block about 40, and strictly supervised QDA 80 (fixed regularisation) to 120 (nested, the S005 procedure)'
- **Problem:** Hidden optimism. The 40 and 80 both fix reg at 0.003, a value taken from R201's nested choices with all 250 labels, so they borrow information from the full label set. Neither row is drawn in Figure 2(a), which the paragraph cites; both are marked table-only.
- **Fix:** Replace with: '... self-training needs about 20. Supervised QDA needs about 40 on the label-free block and 80 with an in-fold block, both with the regularisation fixed at 0.003, a value chosen with all 250 labels (R201); with the regularisation tuned by nested CV, as in S005, it needs 120 (Figure 2 shows only the nested procedures; the fixed-regularisation rows are in `reports/tables/core_label_budget.md`).'
- **Evidence:** research/reports/r2_audit.md §2.3 ('80 ... holds only with reg fixed at 0.003, a value taken from R201's nested choices on all outer folds'); reports/tables/core_label_budget.md (rows marked '*' are table-only); D013 item 2.
- **Resolution:** core_fixer: applied, same text as N3.

### C17 (medium)

- **Location:** reports/extensions/e2_learning_curve.md line 35 (Question, item (b)): 'the best supervised model, QDA with `reg_param` 0.1 ... on the 13-column high-variance block'
- **Problem:** Outdated and wrong twice. QDA reg 0.1 (S002, CV 0.9958) is not the best supervised model: D009/D011 found reg 0.1 over-regularised, and nested-tuned strict QDA reaches 0.9999-1.0000. Its block is also chosen on the train+test features, so this model is not label-only.
- **Fix:** Replace with: '(b) supervised QDA with `reg_param` 0.1 (Friedman, 1989), the model of S002 (CV 0.9958; nested tuning reaches 0.9999 in S005), on the 13-column high-variance block, which is chosen without labels from the train+test features;'
- **Evidence:** registry/decisions.md D009/D011; R440 P17 (r201 0.9999), P21 (r210 1.0000); registry/submissions.csv S002; registry/journal.md J5 (blocks from train+test count as using test features).
- **Resolution:** main session (2026-11-01, after the workflow): applied as proposed (E2 Question item (b)).

### C18 (medium)

- **Location:** reports/extensions/e4_feature_stability.md lines 97-98 (Conclusion): 'Alone these six give QDA 0.917 and the GMM 0.932, above the best linear model in R300 (L2 on the whole block, 0.898).'
- **Problem:** Picks a weaker comparator and contradicts the archive and the core report. The best linear model on the block is 0.938 (0.932 in-fold), which R440 F21 established and the core report quotes. The six-feature GMM (0.932) is therefore not above the best linear model. E4's own Table 1 gives L2 on the block 0.862 with C = 0.1.
- **Fix:** Replace with: 'Alone these six give QDA 0.917 and the GMM 0.932: close to the best linear model on all 13 block columns (0.932 with the block chosen in each fold, 0.938 with the train+test block; R249) and well above L2 logistic regression with C = 0.1 on the block (0.862, Table 1).'
- **Evidence:** R440 P3 and F21 (best linear 0.938298, 20261022-074147_r207_l2_block_tt_wide); reports/tables/r249_comparison.csv (r200_l1_block 0.9321); reports/report_core.md line 107; e4 Table 1 (0.862 ± 0.007).
- **Resolution:** main session (2026-11-01, after the workflow): applied (E4 Conclusion): close to the best linear model on all 13 block columns (0.932 in-fold block, 0.938 train+test block; R249), well above L2 with C = 0.1 (0.862, Table 1). Checked against r249_comparison.csv and E4 Table 1.

### C19 (low)

- **Location:** reports/report_core.md line 101 (Section 4 'Final selections', first sentence): 'A second selection adds at most 1.45·10⁻⁵ ... (S006; S005 4.1·10⁻⁶; R936)'
- **Problem:** Mixed priors. The 1.45e-5 uses R156's 4-point noise weights, which are truncated at 0.01. The paragraph below uses the flat prior, under which the gains are +2.0e-5 and +7.0e-6 (R425), so 'at most 1.45e-5' is false under the report's own primary prior. The conclusion does not change.
- **Fix:** Replace with: 'A second selection adds at most 1.45·10⁻⁵ in expected private AUC under R936's noise weights (S006; S005 4.1·10⁻⁶), or 2.0·10⁻⁵ and 7.0·10⁻⁶ under the flat prior used below (R425): about 1/20 of the private sd or less. So D010 chooses it for robustness ...'
- **Evidence:** research/reports/r2_noise.md §1.4 (R922 flat: r209 +2.0e-5, r201 +7.0e-6; r936/R156: +1.45e-5, +4.1e-6); R440 X6, X9.
- **Resolution:** core_fixer: applied, same text as N7.

### C20 (low)

- **Location:** reports/report_core.md line 78 ('Every upload asked one question and was pre-registered') and the bullets that follow
- **Problem:** The 'journey' framing implies sequential learning from LB answers. In fact S001-S003 came from one research batch (R003) and were uploaded within 86 s, and S005-S006 within 66 s. No public score could inform the next upload, and by design none changed a decision.
- **Fix:** Add after Table 1: 'S001-S003 came from one research batch (R003) and were uploaded within two minutes, as were S005 and S006. The public scores therefore checked the pipeline and our calibration; by design (D008, D010) none of them steered the model choice.'
- **Evidence:** registry/submissions.csv submitted_at_utc: S001 2026-10-21T21:14:41Z, S002 21:15:16Z, S003 21:16:07Z; S005 2026-10-22T13:21:31Z, S006 13:22:37Z; registry/journal.md J2.
- **Resolution:** core_fixer: applied. 'S001-S003 (one batch, R003) went up within two minutes, as did S005-S006, so the public scores checked the pipeline and our calibration and, by design (D008, D010), steered no model choice'. Upload gaps 86 s and 66 s, from submissions.csv.

### C21 (low)

- **Location:** reports/report_core.md line 52 (Section 2 'Leakage control'): 'show that our pipelines do not leak: the seven honest pipelines ... score 0.477-0.496 on permuted labels, a leaky-filter control 0.697 and a control that reads the true labels from disk 1.000.'
- **Problem:** Two different nulls are mixed. 0.477-0.496 is Null A only; under Null B, where the disk-reader control belongs, honest pipelines score 0.480-0.573 (the GMM's scores are bimodal). S001 and S002 were not tested. A permutation test shows no detectable leak; it does not show 'do not leak' for all pipelines.
- **Fix:** Replace with: 'Label-permutation nulls with positive controls (R900) found no leak in the pipelines of all final candidates. Under Null A the seven honest pipelines (S003, five other candidates, E3 honest) score 0.477-0.496 and a leaky-filter control 0.697; under Null B they score 0.480-0.573 (the mixture's scores are bimodal) and a control that reads the true labels from disk 1.000.'
- **Evidence:** registry/journal.md J5 (Null B honest 0.480-0.573); R440 V4-V6; verify.md D1.
- **Resolution:** core_fixer: applied. Permuted-label training and scoring: 0.477-0.496 vs leaky control 0.697. Permuted-label training scored on true labels: 0.480-0.573 (mixture bimodal) vs disk-reader 1.000. Wording scoped to 'the seven honest pipelines tested'.

### C22 (low)

- **Location:** reports/report_core.md line 42 (Section 1 'The generator': first sentence 'These facts identify scikit-learn's make_classification ...' and last sentence 'The open problem is thus the ranking of the ~1% boundary rows ...'); line 54 ('from the identified generator')
- **Problem:** 'identify' and 'identified' overstate: generators outside make_classification were not modelled, and exact dependencies alone do not identify the column roles. 'The open problem is the ranking of the boundary rows' clashes with the report's own finding that S003 is 2.6e-7 from Bayes on them. The remaining loss is label noise, and the gaps between candidates also involve rows beyond the boundary set (see the separate boundary-rows issue).
- **Fix:** First sentence: 'These facts match scikit-learn's `make_classification` (Guyon, 2003) exactly, with 12 informative, 8 redundant and 130 noise columns, one Gaussian cluster per class and `class_sep` ≈ 0.70, followed by a per-column affine map (we did not model generators outside `make_classification`); ...'. Last sentence: 'Good models therefore differ mainly in how they rank the ~1% of rows near the class boundary and the rows around them; for a near-Bayes model the private score is then limited by label noise (Section 4).' Line 54: 'from the identified generator' → 'from the fitted generator'.
- **Evidence:** verify.md A2 caveat and §7.2 ('generators outside make_classification — not modelled'); R921 (regret 2.6e-7); FACTS.md §4.3(a).
- **Resolution:** core_fixer: applied. 'match ... exactly ... (we did not model other generators)', 'fitted generator' in §2, and the last sentence now reads 'Good models therefore differ mainly near the class boundary, and a near-Bayes model's private score is limited by label noise'.

### C23 (low)

- **Location:** reports/report_core.md line 7 (title block): 'Final Kaggle selections (rule D010, default case): S003 ... and S005 .... Private LB: TODO (published after the deadline).'
- **Problem:** The selections have not been made on Kaggle (planned 23-25 Nov). D010 is still 'proposed', and the case depends on the instructor's answer. 'Private LB: TODO' can never be filled, because the report is submitted before the private LB appears, and the TODO check will flag it.
- **Fix:** Replace with: 'Final Kaggle selections (D010, default case; [TODO: tick on Kaggle 23-25 Nov and update if the instructor's answer selects another D010 case]): **S003**, transductive Gaussian mixture (public 0.99923), and **S005**, strictly inductive QDA (public 0.99953). Private LB: not available when this report was submitted (published after the deadline).'
- **Evidence:** docs/ASSIGNMENT_CHECKLIST.md A (final selection unchecked, 11-23 to 11-25); registry/decisions.md D010 Status 'proposed'; docs/ARCHIVE_RULES.md §5 (report submitted before the private LB).
- **Resolution:** core_fixer: applied, merged with U16. '(rule D010, default case; TODO: tick them on Kaggle, 23-25 Nov)' and 'Private LB: after the deadline.'

### C24 (low)

- **Location:** reports/report_core.md line 101 ('D007 adopts this reading, but the instructor has not confirmed it yet'); registry/decisions.md D007 and D014 Status ('讲师确认仍在询问'); docs/COMPETITION_RULES.md §4 item 1 ('still being asked')
- **Problem:** The documents disagree on whether the instructor has been asked. D007/D014 and COMPETITION_RULES say confirmation is 'still being asked'. The checklist and FACTS say the question has not been asked yet. The report's wording implies a pending answer.
- **Fix:** Until the question is sent: 'D007 adopts this reading. We have not yet asked the instructor [TODO: replace with the date asked and the answer]; hence ...'. Update the D007/D014 Status notes and COMPETITION_RULES §4 item 1 to 'not yet asked (TODO member)'.
- **Evidence:** docs/ASSIGNMENT_CHECKLIST.md line 37 (unchecked, —T); reports/reflections/FACTS.md line 126 (I009 'The instructor has not been asked yet') and line 206; registry/decisions.md D007/D014 Status.
- **Resolution:** core_fixer: applied in the report with neutral wording, 'no instructor ruling is recorded yet (TODO members: ask; record date and answer)'. The archive disagrees with itself: D007/D014 say 'still asking', while checklist line 37 is unchecked. Registry and docs fixes are in not_done. | main session (2026-11-01, after the workflow): D007 and D014 Status lines and docs/COMPETITION_RULES.md §4 item 1 now say the instructor has not been asked yet (D015 item 4).

### C25 (low)

- **Location:** reports/report_core.md line 97 ('provenance rests on the archived code bundle and these re-runs (D012, D013)')
- **Problem:** Hidden caveat. S003's bundle b8534fcd0d68fc89 was saved after the run. It is consistent with the run but cannot prove what ran on the VM; only the bit-identical re-runs establish provenance.
- **Fix:** Replace with: '... so provenance rests on a code bundle saved after the run (consistent with it, but no proof of what ran; D012) and on these bit-identical re-runs (D013).'
- **Evidence:** registry/decisions.md D012 item 1 ('事后保存的副本 ... 不能证明 VM 上跑的就是它'); verify.md E6.
- **Resolution:** core_fixer: applied. 'a code bundle saved after the run (consistent with it, but no proof of what ran; D012) and ... these re-runs (D013)'.

### C26 (low)

- **Location:** reports/report_core.md line 123 (Limitations (1) and (3))
- **Problem:** (3) 'an AI assistant did most of the work' understates: the archive shows the AI did all of the recorded operating work. (1) The defence 'the block is reproduced from the test rows alone' shows the rule is label-free, not that the 1.5 threshold was. The stronger label-free argument is missing.
- **Fix:** (3): 'an AI assistant did almost all of the recorded work: design, code, runs, uploads and drafting (see Use of generative AI).' Append to (1): 'The 1.5 threshold falls in the gap between the largest noise-column ratio (1.47) and the smallest block ratio (1.68), and the 20 inter-correlated columns, found without labels, span the same 12 informative dimensions (R150, R901, R920).'
- **Evidence:** reports/reflections/FACTS.md §1.1-1.3; experiments/20261022-094812_r901_id_order_dupes/summary.json block.ratio_profile (13th ratio 1.675, 14th 1.473); research/reports/verify_leak.md:55; research/reports/r2_audit.md §3.4.
- **Resolution:** core_fixer: applied in part. (3) reads 'An AI assistant did almost all of the recorded work: design, code, runs, uploads and drafting'. (1) adds that the 1.5 threshold lies between the 13th and 14th largest sd ratios (1.68 / 1.47, R901). The second clause is rejected (see not_done). | second clause rejected by core_fixer: the 13 block columns are not the 12 informative columns (R901 block13 vs informative12), and the claim could not be stated exactly from the cited runs; the archived 1.5-threshold gap fact was added instead.

### C27 (low)

- **Location:** reports/report_core.md line 86 ('All six scores landed within 1.4 SE of expectation, and each was 0.1-0.6 SE below its CV mean: within noise, and expected where CV sits at the ceiling of 1.')
- **Problem:** Six of six publics fall below CV, a pattern the sentence does not explain. For S001 and S004, CV is not at the ceiling. The common cause is that all six share the same 395 public rows and their label noise, which the training labels do not show.
- **Fix:** Replace with: 'All six landed within 1.4 SE of expectation and 0.1-0.6 SE below their CV means. They share the same 395 public rows and their label noise, which the training labels do not show (R922), so a common small shortfall is expected; it is not evidence of CV optimism.'
- **Evidence:** reports/tables/core_journey.md z(CV) column (all negative); verify.md B4 (no train label contradicts a confident posterior); research/reports/r2_audit.md §4 notes.
- **Resolution:** core_fixer: applied. 'They share the same 395 public rows and their label noise, of which the training labels show no sign (R922), so a common small shortfall is expected, not CV optimism'. R922 has 0 training rows with P(observed label) < 0.2.

### C28 (low)

- **Location:** reports/report_core.md line 54 ('many candidates scored AUC 1.0 in all 100 folds (30 of the 139 runs on the official splits)')
- **Problem:** Runs are counted as candidates. Three of the 30 are re-runs of S003. None of the 30 is strictly inductive (r201 0.99990, r210 0.99998), which is itself the relevant point.
- **Fix:** Replace with: '(30 of the 139 official runs, three of them re-runs of S003; none of the 30 is strictly inductive)'.
- **Evidence:** registry/experiments.csv (139 runs with split_sig 5708ce5c3828) + each run's fold_scores.csv: 30 runs with all 100 folds = 1.0, incl. 20261022-064436_r102_ref_s003, 20261022-091211_r910_s003_gmm2_hv, 20261022-212507_final_s003_gmm2_hv.
- **Resolution:** core_fixer: applied. '30 of the 139 official runs in registry/experiments.csv, three of them re-runs of S003; none strictly inductive'. Recounted from fold_scores.csv.

### C29 (low)

- **Location:** reports/report_core.md line 117 ('changed the expected AUC by at most 2.4·10⁻⁸ on 40 mixture replicas (R103)')
- **Problem:** Hidden circularity. R103's replicas are drawn from the fitted K = 2 mixture itself, where the mixture is the true model by construction, so mixture refinements cannot help there. The text does not say so.
- **Fix:** Replace with: '... changed the expected AUC by at most 2.4·10⁻⁸ on 40 replicas drawn from the fitted mixture itself (R103), where the mixture is the true model by construction; ...'
- **Evidence:** experiments/20261022-073702_r103_replica_bootstrap/notes.md ('40 + 40 replicas of 20,000 rows from the fitted K=2 generator').
- **Resolution:** core_fixer: applied. '40 replicas drawn from the fitted mixture itself (the true model there by construction; R103)'.

### C30 (low)

- **Location:** reports/report_core.md line 135 (Software line: 'Methods used only in exploratory runs are listed in `registry/references.md`.')
- **Problem:** Rule 4 risk: 'Everything used must be cited in the report.' About 25 'status: used' entries are not in the document, e.g. NumPyro and JAX (horseshoe runs), random forests, extra trees, label spreading/propagation, DP-GMM, horseshoe/NUTS, ABC. The checklist says exploratory software and methods must also be listed. Citations in an unowned extension drop out of the document too.
- **Fix:** Add a compact line, e.g.: 'Exploratory runs only: NumPyro 0.22 (Phan et al., 2019) on JAX 0.10 (Bradbury et al., 2018); random forests (Breiman, 2001); extra trees (Geurts et al., 2006); label spreading and propagation (Zhou et al., 2004; Zhu & Ghahramani, 2002); DP-GMM (Blei & Jordan, 2006); horseshoe priors (Carvalho et al., 2010; Piironen & Vehtari, 2017) with NUTS (Hoffman & Gelman, 2014); ABC (Pritchard et al., 1999; Beaumont et al., 2002).' Add the full entries, or ask the instructor whether the references file in the zip is enough.
- **Evidence:** registry/references.md lines 69-231 ('% status: used' entries absent from the report: numpyro, jax, bingham2019, qu2025, breiman2001, geurts2006, zhou2004, zhu2002, blei2006, carvalho2010, piironen2017, hoffman2014, pritchard1999, beaumont2002, mardia1970, wilks1938, ...); docs/COMPETITION_RULES.md rule 4; docs/ASSIGNMENT_CHECKLIST.md line 55.
- **Resolution:** core_fixer: applied, together with U1.

### C31 (low)

- **Location:** reports/extensions/e2_learning_curve.md line 59 ('(c) the GMM has no learning curve: every subset scores 1.0 at every size') and line 69 ('two labels give a perfect ranking, against about 75 for QDA (median AUC 0.99)')
- **Problem:** (a) The flat curve reflects the CV ceiling, not the absence of learning, since 50-row folds rarely contain boundary rows (E2's own conclusion says this). (b) In the same document the core report says QDA on the label-free block needs 'about 40' labels (R441, reg 0.003, mean), while E2 says 'about 75' (R304, reg 0.1, median). The settings are not named, so readers see a contradiction.
- **Fix:** Line 59: '(c) the GMM's CV is at its ceiling at every size: every subset scores 1.0.' Line 69: '... against about 75 for QDA with `reg_param` 0.1 (median AUC 0.99; R304).'
- **Evidence:** reports/report_core.md line 113; research/reports/r2_audit.md §2.2 (R441 reg 0.003: 40; R304 reg 0.1: 75).
- **Resolution:** main session (2026-11-01, after the workflow): applied: (c) 'the GMM's CV is at its ceiling at every size'; '75 for QDA with reg_param 0.1 (median AUC 0.99; R304)'.

### C32 (low)

- **Location:** reports/extensions/e1_validation_protocol.md line 45 ('it orders a 0.01 pair correctly only 64 % of the time') and line 56 ('It is the cheapest scheme (200 fits) whose MDD CI lies wholly below 0.01')
- **Problem:** (a) p_correct is an extrapolation but is stated as observed; the draft admits this only a few lines later. (b) 'cheapest' implies other qualifying schemes exist, but 5 × 20 is the only tested scheme that qualifies, and the most expensive one.
- **Fix:** Line 45: '...; under a normal approximation it would order a 0.01 pair correctly only 64 % of the time (p_correct).' Line 56: 'It is the only scheme we tested whose MDD CI lies wholly below 0.01 (0.0061 [0.0051, 0.0076]; 200 fits per decision), which supports D002.'
- **Evidence:** verify.md F2; e1_validation_protocol.md:48 ('p_correct at 0.01 is a normal-shift extrapolation, not an observed rate'); Table 1 (5 × 10 upper CI 0.0104).
- **Resolution:** main session (2026-11-01, after the workflow): applied: 'under a normal approximation it would order ... (p_correct)'; 'the only scheme we tested whose MDD CI lies wholly below 0.01 (...; 200 fits per decision)'.

### C33 (low)

- **Location:** reports/extensions/e3_leakage.md lines 66-68 ('The proxy (0.820) is 0.028 above (1.2 SE), inside the pre-registered 90% range 0.785-0.854')
- **Problem:** Ambiguous: the sentence reads as if the proxy lies inside the pre-registered range. It is the public score that does.
- **Fix:** Replace with: 'The proxy (0.820) is 0.028 above it (1.2 SE). The public score lies inside the pre-registered 90% range 0.785-0.854 (B005; reports/tables/e3_leaderboard.csv).'
- **Evidence:** registry/beliefs.md B005; registry/submissions.csv S004 notes ('90% 0.785-0.854').
- **Resolution:** main session (2026-11-01, after the workflow): applied as proposed (E3 Results, Leaderboard paragraph).

### C34 (low)

- **Location:** reports/contribution_table.md lines 62-67, 75 and legend line 78 (AI-use cells); registry/ai_usage.md
- **Problem:** The Core rows cite no ai_usage row for the 2026-11-01 drafting of the core report (legend: 'row TODO'). The Video row says AI use '–' although video/PLAN.md is AI-written (row 1). The E2/E4 figure redraws of 11-01 also have no row.
- **Fix:** Append a 2026-11-01 row to registry/ai_usage.md (core report draft; scripts/make_core_figures.py with the core figures and tables; scripts/build_report.py; this table; reports/reflections/FACTS.md; D014 and docs/COMPETITION_RULES.md; E2/E4 figure redraws via run.py --replot). Put its number in every Core row, the 'Use of generative AI; References' row and the E2 and E4 rows, and set the Video row's AI use to '1'.
- **Evidence:** reports/contribution_table.md line 78 ('drafts of 2026-11-01 (this table, D014): row TODO'); registry/ai_usage.md (no row dated 2026-11-01); report_core.md comment 'Draft written 2026-11-01 by Claude Code'; e2_learning_curve.md / e4_feature_stability.md comments ('redrawn 2026-11-01 (ext_figures, Claude Code)').
- **Resolution:** main session (2026-11-01, after the workflow): applied: ai_usage rows 14-21 appended for 2026-11-01; the contribution table cites them (Core rows 14, E2/E4 16, all extensions 21, Video '1, 19', Report assembly '10, 15, 20'); legend lists rows 14-21, 'row TODO' removed.

### C35 (low)

- **Location:** reports/reflections/FACTS.md line 242 (§4.3 a: '83 % (R106)'), line 129 (§2.1 I012 reason: 'every refinement changes less than 1e-8'), line 140 (§2.2 D005: 'with each run's code bundle saved')
- **Problem:** Three errors in the fact sheet that would propagate into reflections. (1) The 83% figure is from R103, not R106. (2) R103 shows losses up to 2.4e-8 (bagging) and 3.7e-6 (2 components per class), so 'less than 1e-8' is wrong, and 'the rest is label noise' is a 5:1 inference. (3) D012: 47 --no-sync runs and the pre-D005 runs, S003 included, have no recorded bundle.
- **Fix:** (1) '(R106)' → '(R103)'. (2) I012 reason → 'no refinement gained more than 1e-8; losses reached 2.4e-8 (bagging) and 3.7e-6 (two components per class) (R103); the remaining public error most likely comes from label noise (about 5:1, R922)'. (3) Append to the D005 row: '(not achieved for the 47 --no-sync runs or the runs before D005, S003 included; D012)'.
- **Evidence:** experiments/20261022-073702_r103_replica_bootstrap/summary.json (G1 public_p_le_obs 0.8276) and notes.md; experiments/20261022-080900_r106_label_noise_rate/notes.md (no 83%); registry/decisions.md D012.
- **Resolution:** main session (2026-11-01, after the workflow): applied: (1) R106 -> R103 (checked: R103 summary public_p_le_obs 0.8276 under G1); (2) I012 reason corrected with R103's losses (2.4e-8, 3.7e-6) and 'about 5:1, R922'; (3) D005 row notes the 47 --no-sync runs and pre-D005 runs (D012).

### C36 (low)

- **Location:** video/PLAN.md line 18 (1:50-2:40 row: '一个被证伪的预测')
- **Problem:** The plan asks for a prediction falsified by the LB, but no belief was refuted by a leaderboard score. B004 is only model-judged miscalibrated and stays untested until the private LB. A script written from this plan would likely overclaim.
- **Fix:** Replace '一个被证伪的预测' with '一个与预期不符的地方（B003 的解释或 B005 的点预测；B004 只是被模型判为过于自信，private 还没公布，不能说被 LB 证伪）'.
- **Evidence:** reports/reflections/FACTS.md §4.4 ('No B row was refuted by a leaderboard score'); registry/beliefs.md B004 Outcome 'pending'.
- **Resolution:** main session (2026-11-01, after the workflow): applied as proposed (video/PLAN.md storyboard row 1:50-2:40).

## Rubric, format and readability review (review_rubric)

**Reviewer summary.**

Rubric, format and readability review of reports/report_core.md (version saved 2026-11-01 10:13:50). I did not edit any project file; all builds went to the scratchpad.

COVERAGE. Every required part is present:
- §1 problem formulation.
- §2 validation protocol, with leakage control and a selection rule written in advance.
- §3 submission history: the question each upload asked, the pre-registered expectation, the result and the lessons (Figure 1, Table 1, "What we learned").
- §4 final solution, with provenance and how to reproduce it.
- §5 analysis, with negative results and limitations.
- An AI disclosure that names the tool and what it was used for.
- References.

LENGTH. `.venv\Scripts\python.exe scripts\build_report.py --parts core` (output written to the scratchpad) gives 6 pages, 5.52 used, and warns "TODO left: core 6". `reports\report.pdf` (10:50, 15 pages) has the core on pp. 1–6, with the same page breaks. It also contains all four extensions, because `project.yaml` lists no owners.

FIGURES. Both are 6.3 in wide at 220 dpi and are placed at 99.5% of the text width, so their text is 7.5–9 pt and legible at 100%. Table 1 is readable. Figure 2 is referenced in the text; Figure 1 and Table 1 are not.

MAIN PROBLEMS
1. Citations are incomplete under rule 4. About 30 entries marked "used" in `registry/references.md` are missing, including LIBLINEAR, which fitted S001 and S004. Nine used software items are also missing (Python, PyYAML, joblib, threadpoolctl, tabulate, JupyterLab, huggingface_hub, NumPyro, JAX), plus the two Hugging Face model cards. The report sends the reader to a repository file instead.
2. The 6-page fit is fragile. The build keeps Table 1 and Figure 2 whole with their captions. When the needed additions are made without cuts, Table 1 and then Figure 2 move down and the core becomes 7 pages (tested: 6.37 and 6.51).
3. Readability for a non-specialist marker:
   - the internal codes are never explained;
   - "transductive", Bayes AUC, posterior and flip_y are never defined;
   - the Figure 2 text describes rows that the figure does not show;
   - "Bayes error" is used for an AUC loss;
   - two statements contradict other parts of the report ("Structure found without labels" vs Limitation 1; "differ in covariance, not in means" vs R154);
   - "Two beliefs were wrong" names only one.

TOP EDITS BY MARKING VALUE PER PAGE. All 26 exact replacements were tested together. Each old string occurs exactly once in the current file. With all of them applied the core is 6 pages, 5.91 used, with no lines sticking out into the margin.
1. Complete the references (R1–R3). Cost about +0.35 page, paid for by the cuts C1–C5.
2. Add the line explaining the codes (G1, 2 lines).
3. Make the Figure 2 text match the figure (F2, no extra lines).
4. Refer to Figure 1 and Table 1 in the text (F1, no extra lines).
5. Describe the final selections in plain words (O1), offset by O2 (no net change).
6. Fix the two contradictions (T3, T4).
7. Clarify "What we learned" (T1, T2, T11).
8. Say explicitly that a public-to-private gap is not a failure (T5).
9. Make Figure 1 shorter: about 2.05 in instead of 2.55 in, in `make_core_figures.py`. After the package there is less than 1 line of room before Table 1; one more line gives 7 pages. A simulated shorter Figure 1 gives about 3 lines of room.
10. Make the final-selection argument easier to follow (T9).

ONLY MEMBERS CAN DO: the team and member TODOs, the "Private LB" line, and the AI-disclosure sentence about members' own review.

FILES (scratchpad only):
- Edited copy: D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\review_rubric\vP3\report_core.md
- Its PDF: D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\review_rubric\vP3.pdf
- Diff: D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\review_rubric\report_core_package.diff
- Exact old/new pairs: D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\review_rubric\package_edits.json
- Scripts: apply.py, build_variant.py and headroom.py in the same folder.
- Baseline build: D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\review_rubric\core.pdf

Other agents were writing to video/ during this review. If report_core.md has changed since 10:13:50, re-check the old strings before applying.

### U1 (high)

- **Location:** References (report_core.md line 135, software line; lines 137-162, list)
- **Problem:** Rule 4 says everything used must be cited in the report. docs/COMPETITION_RULES.md §4.5 and checklist B1 say this includes exploratory runs. The report cites 27 works and says 'Methods used only in exploratory runs are listed in registry/references.md'. About 30 entries marked 'status: used' are missing:
- LIBLINEAR (Fan et al., 2008), which fitted the submitted models S001 and S004;
- the methods behind the Section 5 baselines: random forests, extra trees, SVM with Platt scaling, kNN, label spreading and label propagation;
- NumPyro, JAX and Pyro; Qu et al. (2025); Efron (1979); Cawley & Talbot (2010);
- the ABC, MCMC, normality and likelihood-ratio methods.
The software note also omits Python, PyYAML, joblib, threadpoolctl, tabulate, JupyterLab, huggingface_hub, NumPyro and JAX, and the two Hugging Face model cards that the Tutorials table marks as 'used'.
- **Fix:** (a) REPLACE «*Software:* scikit-learn 1.9.1 (Pedregosa et al., 2011), NumPy 2.4.6 (Harris et al., 2020), pandas 3.0.6 (McKinney, 2010), SciPy 1.17.1 (Virtanen et al., 2020), matplotlib 3.11 (Hunter, 2007), LightGBM 4.7 and XGBoost 3.2; for analysis only, TabPFN 9.2.0 and TabICL 2.2.0 on PyTorch 2.14.1 (Paszke et al., 2019). Methods used only in exploratory runs are listed in `registry/references.md`.» WITH «*Software* (Python 3.11): scikit-learn 1.9.1 (Pedregosa et al., 2011) with its LIBLINEAR solver (Fan et al., 2008), NumPy 2.4.6 (Harris et al., 2020), pandas 3.0.6 (McKinney, 2010), SciPy 1.17.1 (Virtanen et al., 2020), matplotlib 3.11.2 (Hunter, 2007), joblib 1.6.0, threadpoolctl 3.7.0, PyYAML 6.0.3, tabulate 0.10.0, JupyterLab 4.6.4, LightGBM 4.7.0 and XGBoost 3.2.0; for analysis only, TabPFN 9.2.0 (Hollmann et al., 2025) and TabICL 2.2.0 (Qu et al., 2025, 2026) with weights from the Hugging Face model cards Prior-Labs/TabPFN-v2-clf (Prior Labs License) and jingang/TabICL (BSD-3-Clause), accessed 2026-10-22 via huggingface_hub 2.2.0, on PyTorch 2.14.1 (Paszke et al., 2019), and NumPyro 0.22.0 (Phan et al., 2019; Bingham et al., 2019) on JAX 0.10.2 (Bradbury et al., 2018).»
(b) Between the 'Efron, B. & Tibshirani, R. J. (1993)…' line and the 'Friedman, J. H. (1989)…' line, insert the line «Fan, R.-E. et al. (2008). LIBLINEAR: A library for large linear classification. *JMLR*, 9, 1871–1874.» It must end with a backslash, like the lines around it.
(c) After the last line «Yarowsky, D. (1995). Unsupervised word sense disambiguation rivaling supervised methods. *Proc. ACL*, 189–196.» add a blank line and then this paragraph: «*Further methods and software used in our runs* (rule 4; full entries in `registry/references.md`): Beaumont et al. (2002), *Genetics* 162; Bingham et al. (2019), *JMLR* 20; Bishop (2006), *Pattern Recognition and Machine Learning*, Springer; Blei & Jordan (2006), *Bayesian Anal.* 1; Bradbury et al. (2018), JAX, github.com/jax-ml/jax; Breiman (2001), *Mach. Learn.* 45; Carvalho et al. (2010), *Biometrika* 97; Cawley & Talbot (2010), *JMLR* 11; Chang & Lin (2011), *ACM TIST* 2; Cover & Hart (1967), *IEEE Trans. Inf. Theory* 13; D'Agostino & Pearson (1973), *Biometrika* 60; Defazio et al. (2014), *NeurIPS* 27; Efron (1979), *Ann. Stat.* 7; Gelman & Rubin (1992), *Stat. Sci.* 7; Gelman et al. (2013), *Bayesian Data Analysis*, 3rd ed., CRC; Geurts et al. (2006), *Mach. Learn.* 63; Golub & Van Loan (2013), *Matrix Computations*, 4th ed.; Hoffman & Gelman (2014), *JMLR* 15; Mardia (1970), *Biometrika* 57; Phan et al. (2019), arXiv:1912.11554; Piironen & Vehtari (2017), *Electron. J. Stat.* 11; Platt (1999), in *Advances in Large Margin Classifiers*, MIT Press; Pritchard et al. (1999), *Mol. Biol. Evol.* 16; Qu et al. (2025), TabICL, *ICML*; Vehtari et al. (2021), *Bayesian Anal.* 16; Wilks (1938), *Ann. Math. Stat.* 9; Zhou et al. (2004), *NeurIPS* 16; Zhu & Ghahramani (2002), CMU-CALD-02-107.»
These changes cost about 0.35 page. Apply them only together with the cuts in the page-fit issue. If E2 is not in the document, also add «Figueroa et al. (2012), *BMC Med. Inform. Decis. Mak.* 12;» to (c); its status is 'used' for the E2 method.
- **Evidence:** registry/references.md: the status 'used' entries fan2008 ('liblinear: S001, S004'), qu2025, numpyro, bingham2019, jax, efron1979, cawley2010, figueroa2012, defazio2014, chang2011, platt1999, cover1967, breiman2001, geurts2006, zhou2004, zhu2002, blei2006, bishop2006, gelman2013, carvalho2010, piironen2017, hoffman2014, vehtari2021, gelman1992, golub2013, mardia1970, dagostino1973, pritchard1999, beaumont2002 and wilks1938 are absent from report_core.md. The 'Other packages… List them in the report's software note with versions' line is not followed. reports/tables/core_submission_history.md S004: solver=liblinear. requirements-lock.txt: jupyterlab 4.6.4, joblib 1.6.0, PyYAML 6.0.3, tabulate 0.10.0, threadpoolctl 3.7.0. S003 meta.json: python 3.11.17.
- **Resolution:** core_fixer: applied in condensed form. The software line now has Python 3.11, the LIBLINEAR solver, joblib, threadpoolctl, PyYAML, tabulate, JupyterLab, huggingface_hub, NumPyro, JAX, exact versions and the two Hugging Face model cards; licence names were dropped. Fan et al. (2008) added. The 'Also used in our runs' list adds all other 'used' entries, including Figueroa 2012, so the core stays complete if E2 is unowned. Compact run-on style without titles, to fit 6 pages.

### U2 (high)

- **Location:** Page fit: Section 3 (Figure 1 + Table 1 at the foot of page 2), Section 5 (Figure 2 on page 4)
- **Problem:** The core is now 6 pages, 5.52 used. The build keeps Table 1 and Figure 2 whole together with their captions. If more than about 3 lines are added before Section 3, Table 1 moves to page 3. That pushes Figure 2 to page 5 and leaves about 0.35 page blank.
Tested in the scratchpad:
- issue 1 plus the clarity edits, with no cuts: 7 pages (6.51; 6.37 with compact wording);
- the tested package below: 5.91, but with less than 1 line of room before Table 1 (one more filler line gives 7 pages, 6.37);
- room before Figure 2: at least 4 lines.
Members rewriting Sections 1–2 in their own words will easily cross this limit.
- **Fix:** Apply these cuts together with the additions:
(C1) REPLACE «*Table 1. The six uploads (S001–S003 on Kaggle day 2026-10-21 UTC, S004–S006 on 10-22; `registry/submissions.csv`, `reports/tables/core_journey.md`). CV: official 5 × 20, sd over repeats. Expected: pre-registered. z = (public − expected or CV)/SE, SE = Hanley–McNeil SE of the expected AUC on 395 rows (198/197, as in R440). S002 chose its columns with train+test features; S003 and S006 are fitted on the unlabelled test rows; S001, S004, S005 use no test features.*» WITH «*Table 1. The six uploads (S001–S003 on Kaggle day 2026-10-21 UTC, S004–S006 on 10-22; `registry/submissions.csv`). CV: official 5 × 20, sd over repeats. Expected: pre-registered. z = (public − expected or CV)/SE, with the Hanley–McNeil SE of the expected AUC on 395 rows. S002 chose its columns with train+test features; S003 and S006 are fitted on the unlabelled test rows; S001, S004 and S005 use no test features.*»
(C2) REPLACE «**With more time** we would obtain and archive the instructor's ruling on test features and upload the extended-grid QDA in advance. More labels would help the supervised models (E2), not the mixture, which is at the Bayes limit.» WITH «**With more labels** the supervised models would improve (E2), but not the mixture, which is already at the Bayes limit.» The instructor's ruling is not a 'more time' item, and Section 4 already covers it.
(C3) REPLACE «(3) All runs predate the first git commit, and an AI assistant did most of the work (below).» WITH «(3) An AI assistant did most of the work (below).» The git point is already made in §4, 'Run and provenance'.
(C4) REPLACE « Never a foundation-model run or S004. Public scores cannot change the rule, because the sd of their differences (3–6·10⁻⁴) exceeds any true gap.» WITH « Never a foundation-model run or S004.»
(C5) REPLACE «are within the sd of public differences, so by D010 they changed nothing.» WITH «are within the sd of public differences (3–6·10⁻⁴), so they could not change the rule (D010).»
To gain room, regenerate Figure 1 lower. In scripts/make_core_figures.py, line 230: figsize=(6.3, 2.55) → (6.3, 2.05). Line 262: subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.79) → (left=0.085, right=0.99, bottom=0.22, top=0.74), which keeps the same absolute space for the legend and labels. A simulation gave about 3 lines of room before Section 3 and still 6 pages. After every change, rebuild with --parts core and --out outside reports/, and check that Table 1 is still on page 2 and 'used' is at most 5.95.
- **Evidence:** Scratch builds: current file 5.52 (6 pages). Current file plus 3 filler lines before §3: still 5.52. All edits, no cuts (v_all): 6.51, 7 pages. Compact wording (v2): 6.37, 7 pages. Tested package (vP3): 5.91, 6 pages. Package plus 1 line (vP_hr1): 6.37, 7 pages. Package with Figure 1 at 2.05 in (vPs): +3 lines gives 5.91, +4 lines gives 6.37. Package plus 4 lines before Figure 2: 5.97.
- **Resolution:** core_fixer: applied. Did cuts C1-C5. Figure 1 height 2.55 → 2.1 in (FIG1_HEIGHT in make_core_figures.py, same absolute margins and fonts). Also moved 'Limits of CV resolution' into §2, placed Figure 2 after its paragraph and shortened short last lines. Result: 5.77 pages, 2.6 lines free above Figure 1 on page 2, about 11.8 lines free on page 6.

### U3 (high)

- **Location:** Between the title block and '## 1. Problem formulation'
- **Problem:** The text uses about 70 internal codes (S###, R###, r###, D###, B###, E#) and never explains them. A marker cannot tell that D010 is a selection rule written before the scores were seen, that B004 is a pre-registered prediction, or that R936 is an archived run. That makes the traceability and pre-registration story hard to follow.
- **Fix:** REPLACE «## 1. Problem formulation» (the heading line) WITH «*Codes:* S### = Kaggle submission; R### or r### = archived run (ids in `registry/experiments.csv`); D### = decision, B### = pre-registered belief (`registry/`); E1–E4 = extensions.» followed by a blank line and then «## 1. Problem formulation». This adds 2 lines and is included in the tested package.
- **Evidence:** report_core.md line 7 uses 'rule D010' before any definition. Lines 40–130 use R001…R936, B001–B008, D002–D013 and r930. package_deliverables.py ships registry/ and experiments/ (RECORD_DIRS), so the pointers resolve in the zip.
- **Resolution:** core_fixer: applied, after the title block.

### U4 (medium)

- **Location:** §5 'What the unlabelled rows add (Figure 2)', sentences (a) and (b)
- **Problem:** The text and Figure 2 disagree:
- The text cites 'QDA on the label-free block about 40' and 'strictly supervised QDA 80'. These are table-only rows, marked * in core_label_budget.md, and do not appear in the figure.
- It omits plotted rows: Bayesian QDA on the train+test block (80) and the label-only Bayesian QDA (160).
- '(b) supervised QDAs 0.66–0.70' includes the table-only r930 value 0.698 and leaves out the plotted 0.652.
- **Fix:** REPLACE «(a) Label budget (R206, re-run on 80 new folds with new label draws in R441): the mixture ranks the validation rows perfectly with 2 labels (400 of 400 evaluations), self-training needs about 20, QDA on the label-free block about 40, and strictly supervised QDA 80 (fixed regularisation) to 120 (nested, the S005 procedure). (b) On the ≈ 185 boundary rows per replica the mixture ranks almost as well as the true posterior (0.804 vs 0.806) and beats every other procedure in all 222 replicas: self-training 0.718, Bayesian QDA with a normal-inverse-Wishart prior (Murphy, 2007) 0.678, supervised QDAs 0.66–0.70 (R404).» WITH «(a) Labels needed (R206, re-run on 80 new folds with new label draws in R441): the mixture ranks the validation rows perfectly with 2 labels (400 of 400 evaluations), self-training needs about 20, Bayesian QDA with a normal-inverse-Wishart prior (Murphy, 2007) on the label-free block 80, and the label-only QDAs 120–160 (S005: 120). (b) On the ≈ 185 boundary rows per replica (true class probability 0.1–0.9) the mixture ranks almost as well as the true probabilities (0.804 vs 0.806) and beats every other procedure in all 222 replicas: self-training 0.718, Bayesian QDA 0.678, label-only QDAs 0.652–0.670 (R404).»
- **Evidence:** reports/tables/core_label_budget.md: plotted rows 2 / 20 / 80 / 120 / 120 / 160; the * table-only rows are 40 and 80. core_boundary.md: 0.804, 0.718, 0.678, 0.670, 0.664, 0.652; r930 0.698 is marked *. core_manifest.json 'rows marked * are table-only'. Runs 20261022-182022_r441_label_budget_rerun and 20261022-194238_r404_inductive_pooled.
- **Resolution:** core_fixer: applied. Text describes the plotted rows. The table-only fixed-reg rows are kept only with the C16 caveat and marked 'table only'.

### U5 (medium)

- **Location:** §3, first sentence after Table 1
- **Problem:** The body text never refers to Figure 1. Table 1 is mentioned only inside the two captions, so the reader is not told to look at them.
- **Fix:** REPLACE «Every upload asked one question and was pre-registered (beliefs B001–B007):» WITH «Figure 1 and Table 1 summarise the six uploads; each asked one question and was pre-registered (beliefs B001–B007):»
- **Evidence:** report_core.md: 'Figure 1' appears only in line 65 (its own caption). 'Table 1' appears only in lines 65 and 76 (captions).
- **Resolution:** core_fixer: applied. 'Figure 1 and Table 1 show the six uploads.'

### U6 (medium)

- **Location:** Title block (line 7) and §2 'Why this scheme'
- **Problem:** The first sentence a marker reads uses 'transductive Gaussian mixture' and 'strictly inductive QDA', and neither term is defined anywhere. 'Why this scheme' also has a dangling modifier: 'Re-running each scheme …, the minimum detectable difference … was'. The second fix also removes a one-word last line ('AUC.'), which offsets the extra line from the first.
- **Fix:** REPLACE «Final Kaggle selections (rule D010, default case): **S003**, transductive Gaussian mixture (public 0.99923), and **S005**, strictly inductive QDA (public 0.99953).» WITH «Final Kaggle selections (rule D010, default case): **S003**, a Gaussian mixture fitted to the unlabelled train and test features (transductive; public 0.99923), and **S005**, a QDA that uses the labelled rows only (strictly inductive; public 0.99953).»
AND REPLACE «Re-running each scheme with 50 split seeds on two neighbouring linear models, the minimum detectable difference» WITH «With 50 split seeds per scheme on two neighbouring linear models, the minimum detectable difference». Tested together: no net change in lines before Table 1.
- **Evidence:** Scratch build vP3: the title block becomes 3 lines and 'Why this scheme' loses its last line, so the page count is unchanged (5.91). D007 / registry/submissions.csv: S003 uses train+test features, S005 uses no test features.
- **Resolution:** core_fixer: applied in the title block and in 'Why this scheme'.

### U7 (medium)

- **Location:** §1 paragraph heading 'Structure found without labels'
- **Problem:** The heading contradicts Limitation (1) in §5: 'The block and K = 2 were chosen after R001/R002 had looked at the training labels'. It also contradicts the last sentence of the same paragraph, which reports label purities (R002). verify.md D2 asked for a disclosure of this in the Problem formulation.
- **Fix:** REPLACE «**Structure found without labels** (R001, R150).» WITH «**Structure in the features** (R001, R150; see also Limitations, Section 5).»
- **Evidence:** report_core.md line 40 vs line 123. research/reports/verify.md row D2: 'R002 checked label purity before the block + K=2 design … One disclosure sentence in Problem formulation'.
- **Resolution:** core_fixer: applied, same text as C9.

### U8 (medium)

- **Location:** §3 S002 bullet, last sentence
- **Problem:** 'The classes differ in covariance, not in means' is wrong. The class means differ on 4 informative columns, and a rule that uses only the means reaches AUC 0.947. LDA's CV of 0.878, quoted in the same bullet, already shows that the means carry signal.
- **Fix:** REPLACE «The classes differ in covariance, not in means.» WITH «The class covariances carry more signal than the class means (AUC 0.9998 vs 0.947 for rules that use only one of them; R154).»
- **Evidence:** reports/tables/r154_population_aucs.md: mean_only_pooled_cov 0.947035, cov_only_same_mean 0.999803. r154_class_means.csv: columns 46, 80, 100, 112 are active, with latent mean differences 0.63–0.75. Run R154 20261022-065605_r154_class_structure.
- **Resolution:** core_fixer: applied, same text as N4.

### U9 (medium)

- **Location:** §3 S003 bullet, last sentence
- **Problem:** 'Its public error (7.7·10⁻⁴) is 15 times the model's own Bayes error (R101)' uses 'Bayes error', which normally means the misclassification rate (0.33% here), for an AUC loss of 5·10⁻⁵. With the term read in its usual sense, the 15× claim makes no sense.
- **Fix:** REPLACE «Its public error (7.7·10⁻⁴) is 15 times the model's own Bayes error (R101): label noise, not ranking errors.» WITH «Its public shortfall, 1 − AUC = 7.7·10⁻⁴, is 15 times the loss of 5·10⁻⁵ that the fitted mixture predicts for itself (R101): a sign of label noise, not of ranking errors.»
- **Evidence:** experiments/20261022-064201_r101_bic_model_choice/notes.md: expected test AUC 0.999948 if labels follow the fitted posterior, so headroom 5e-5, and S003's public loss 7.7e-4 is 15x larger. research/reports/r2_audit.md G23: Bayes error rate 0.33%.
- **Resolution:** core_fixer: applied, same text as N16.

### U10 (medium)

- **Location:** §3 'What we learned'
- **Problem:** 'Two beliefs were wrong.' is followed by only one belief, B004, and B004's outcome is still 'pending'. 'round-1 claims' is internal jargon that a marker cannot decode. This paragraph carries the 'what was learned from multiple attempts' criterion.
- **Fix:** REPLACE «Two beliefs were wrong. B004 (S003's private AUC ≥ 0.999 and within 0.001 of public, 70%) was miscalibrated: with label noise modelled its probability is at most about 0.16 (R922);» WITH «One belief is already known to be miscalibrated: B004 (S003's private AUC ≥ 0.999 and within 0.001 of public, 70%); with label noise modelled its probability is at most about 0.16 (R922);»
AND REPLACE «We also corrected two round-1 claims:» WITH «We also corrected two of our own earlier claims:»
- **Evidence:** registry/beliefs.md: B004 Outcome 'pending'; B008 is the recalibrated version. R922 20261022-092146_r922_label_noise_private gives P(≥0.999) = 0.16.
- **Resolution:** core_fixer: applied. Both expectations are named (C6), and 'two earlier claims' replaces 'round-1 claims'.

### U11 (medium)

- **Location:** §4 'Expected private score', last sentence
- **Problem:** The checklist (B1) asks the report to state that large gaps between validation, public and private scores are not a failure. The report explains the gaps but never says this outright.
- **Fix:** REPLACE «so the private score is limited by label noise and likely about 0.002 below the public score.» WITH «so the private score is limited by label noise and likely about 0.002 below the public score: a gap that would be expected, not a failure.»
- **Evidence:** docs/ASSIGNMENT_CHECKLIST.md line 40 ('报告里要写明 large gaps between validation, public and private scores are not a failure'). R922: median 0.9973 vs public 0.99923.
- **Resolution:** core_fixer: applied. '... about 0.002 below the public score: an expected gap, not a failure.'

### U12 (medium)

- **Location:** 'Use of generative AI', last sentence (member action)
- **Problem:** The paragraph ends with an obligation ('every member must be able to explain every part'), not a fact. That is honest today, but a marker will read it as 'nobody on the team has reviewed this'. Rule 6 requires that members can explain every detail. The 'Verified by' column of registry/ai_usage.md is TODO in every row.
- **Fix:** Only when it is true, REPLACE «These checks do not replace our own review: under rule 6 every member must be able to explain every part (full log: `registry/ai_usage.md`).» WITH «Team members confirmed the decisions in `registry/decisions.md`, re-ran and rewrote the parts they own (contribution table) and can explain them (rule 6; full log: `registry/ai_usage.md`).» Keep the current sentence until then. This adds about 1 line after Figure 2, where there are about 4 lines of room.
- **Evidence:** registry/ai_usage.md: Verified by = TODO in all rows; the draft paragraph's two '[TODO, keep only when true]' clauses. report_core.md lines 126–129 (comment).
- **Resolution:** deferred: the reviewer conditions it on being true; every 'Verified by' cell in registry/ai_usage.md is still TODO. The current sentence stays, with a visible member TODO (core_fixer).

### U13 (low)

- **Location:** §1 'The generator' and §4 'Expected private score'
- **Problem:** 'Bayes AUC', 'posterior' and 'flip_y' are used without definitions. The current gloss 'flip rate f = flip_y' also misdescribes flip_y: it is the share of labels assigned at random, so about f/2 of them actually change. That is why the report's own '1 − f/2' holds.
- **Fix:** REPLACE «Under the fitted generator the Bayes AUC is 0.99995, the best linear rule reaches 0.947, and about 186 test rows (≈ 1%) have a posterior between 0.1 and 0.9 (R154).» WITH «Under the fitted generator the best possible (Bayes) AUC is 0.99995, the best linear rule reaches 0.947, and about 186 test rows (≈ 1%) have a true class probability between 0.1 and 0.9 (R154).»
AND REPLACE «(flat prior on the flip rate f = `flip_y`)» WITH «(flat prior on f, the share of labels assigned at random, `flip_y` in `make_classification`)»
- **Evidence:** scikit-learn make_classification: flip_y = fraction of samples whose class is assigned randomly. Report §4: 'Every near-Bayes model scores about 1 − f/2'. research/reports/r2_noise.md: mislabel rate 0.31% for posterior mean f 0.0063.
- **Resolution:** core_fixer: applied. 'best possible (Bayes) AUC', 'true class probability', and 'f, the share of labels assigned at random, flip_y in make_classification'.

### U14 (low)

- **Location:** §4 'Model' sentence
- **Problem:** 'with no scaler, selector or tuned hyperparameter' contradicts step 1, 'standardised with train+test statistics'.
- **Fix:** REPLACE «with no scaler, selector or tuned hyperparameter:» WITH «with no separate scaler or selector step and no tuned hyperparameter:»
- **Evidence:** configs/final_s003_gmm2_hv.yaml: 'scaler: none  # TransductiveGMM standardises the block itself with train+test statistics'.
- **Resolution:** core_fixer: applied through C8's wording.

### U15 (low)

- **Location:** §4 'Final selections', first sentence
- **Problem:** 'so D010 chooses it for robustness to the open rule question' is hard to parse ('it' = the second selection). It also does not say why a second selection can add expected AUC at all.
- **Fix:** REPLACE «A second selection adds at most 1.45·10⁻⁵ in expected private AUC (S006; S005 4.1·10⁻⁶; R936), about 1/20 of the private sd, so D010 chooses it for robustness to the open rule question about S003's use of the test features.» WITH «Kaggle ranks a team by the better of its two selections; on the replicas a second selection raises the expected private AUC by at most 1.45·10⁻⁵ (S006; S005 4.1·10⁻⁶; R936), about 1/20 of the private sd. D010 therefore uses it as insurance against the open rule question about S003's use of the test features, not to gain score.»
- **Evidence:** registry/decisions.md D010 context: r209 +1.45e-5, r201 +4.1e-6 (reports/tables/r936_choice.csv); private sd 3–6e-4.
- **Resolution:** core_fixer: applied. 'Kaggle ranks a team by the better of its two selections ... D010 uses it as insurance against the open rule question ..., not to gain score.'

### U16 (low)

- **Location:** Title block, 'Private LB: TODO (published after the deadline).' (member action)
- **Problem:** Kaggle reveals the private leaderboard only after the competition closes (29 Nov 23:59 HKT). The report is planned to be submitted by then, so this TODO may never be fillable, and the pre-export TODO check will then fail.
- **Fix:** REPLACE «Private LB: TODO (published after the deadline).» WITH «Private LB: not yet published.» If the Canvas deadline turns out to be after the reveal, members put in the private scores of S003 and S005 instead. This change makes the line shorter.
- **Evidence:** docs/COMPETITION_RULES.md rules 3 and 8; docs/ASSIGNMENT_CHECKLIST.md line 10 (course deadline planned 2026-11-29 23:59 HKT). build_report.py warning 'TODO left: core 6'.
- **Resolution:** core_fixer: TODO'): applied, same text as C23.

### U17 (low)

- **Location:** §3 bullet 'S005, S006'
- **Problem:** '(registered probability ≈ 0.20 and 0.25–0.28)' is cryptic: it does not say what event the probability refers to.
- **Fix:** REPLACE «slightly above S003 (registered probability ≈ 0.20 and 0.25–0.28);» WITH «slightly above S003, an outcome we had pre-registered as about 20% and 25–28% likely;»
- **Evidence:** registry/beliefs.md B006 ('高于 S003 的概率约 20%', i.e. about 20% probability of being above S003) and B007 ('约 25–28%', about 25–28%).
- **Resolution:** core_fixer: applied (report_core.md, S005/S006 bullet: 'an outcome we had registered as about 20% and 25–28% likely').

### U18 (low)

- **Location:** 'Use of generative AI', checks sentence
- **Problem:** 'an audit of 242 report numbers (R440 …)' suggests that this report's text was audited. In fact R440 ran on 2026-10-22, before the report was drafted on 2026-11-01, and checked the numbers that the report later drew on.
- **Fix:** REPLACE «a separate AI verification round and an audit of 242 report numbers (R440: 235 confirmed, 7 flagged).» WITH «a separate AI verification round and an audit, before drafting, of 242 numbers this report draws on (R440: 235 confirmed, 7 flagged).»
- **Evidence:** Run 20261022-184020_r440_core_numbers (2026-10-22). Hidden comment at report_core.md line 17: 'Draft written 2026-11-01'.
- **Resolution:** core_fixer: applied, same text as N15.

### U19 (low)

- **Location:** Figure 2 labels (scripts/make_core_figures.py lines 298, 304, 305) and the Figure 2 caption
- **Problem:** The panel (b) title says '(r404)' while panel (a) and the text use 'R441' and 'R404'. The axis label and annotation say 'Bayes posterior', while the edited text says 'true class probability'. The annotation text is 7.5 pt, the smallest on the page; it is legible, but only just.
- **Fix:** In make_core_figures.py:
- set_title('(b) Boundary rows, 222 replicas (R404)');
- set_xlabel('AUC on rows with true class probability 0.1–0.9');
- annotation f'true probabilities\n(ceiling) {bayes:.3f}' at fontsize 8;
- then regenerate the figure with the script.
In the Figure 2 caption, REPLACE «(b) AUC on test rows with Bayes posterior in (0.1, 0.9), 222 replicas (R404)» WITH «(b) AUC on test rows whose true class probability (Bayes posterior) is in (0.1, 0.9), 222 replicas (R404)». This is optional; the caption change was not part of the tested package.
- **Evidence:** Rendered page 4 at 200 dpi (fig2zoom-4.png in the scratchpad). make_core_figures.py lines 216–218 (font sizes 8–9) and line 298 (fontsize 7.5).
- **Resolution:** core_fixer: applied. Panel title '(R404)'; annotation 'true probabilities (ceiling) 0.806' at 8 pt. The x-label became 'AUC on rows with true probability 0.1-0.9', shorter than suggested because the full version collided with panel (a). Caption updated. Table-view labels updated too: the Bayes row is 'True class probability (Bayes posterior, ceiling)', and the reg-0.003 rows say the value came from R201 using all 250 labels.
