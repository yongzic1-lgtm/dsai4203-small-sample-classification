# Reflection fact sheet

> **Source material, not text to copy.**
>
> - Each member writes an individual reflection in their own words and uploads it as their own PDF on Canvas (template: `reports/reflections/TEMPLATE.md`). A missing reflection gives that member zero for the extension criterion.
> - Claim only your own work. Up to 2026-11-01 the archive shows that the AI assistant Claude Code (Anthropic, Claude Opus 5.5) did almost all of the operating work: it set up the project, designed and ran every experiment on Google Colab, wrote the registry entries and the extension drafts, and uploaded every submission on a member's account under that member's standing authorisation (D008; §1.1–1.3). Write what you personally did, decided, checked or learned, and say what the AI did.
> - Kaggle rule 6 allows generative AI, but its use must be disclosed and every member must be able to explain every detail. Do not write "I did X" unless you did X and can explain it.
> - Every fact below names its source. Open the source before you use a fact. Take numbers from the cited file or from `reports/tables/r440_core_numbers.csv`.
> - This sheet is itself AI output (Claude Code, 2026-11-01, written from the archive without new runs). If you use it, say so where you disclose AI use. Keep it inside the team (Kaggle rule 5).

**Short IDs.** S### = row of `registry/submissions.csv`; B### = `registry/beliefs.md`; D### = `registry/decisions.md`; I### = `registry/ideas.md`; R### = script `research/r###_*.py` and its run folder(s) `experiments/<date-time>_r###_*`. Journal entries (all dated 2026-10-22, in file order): J1 setup; J2 research day 1; J3 research round 1; J4 archiving before a restart; J5 verify round 1 and corrections; J6 fix round and research round 2; J7 upload of S005 and S006. "ai_usage row n" = the n-th row of the table in `registry/ai_usage.md` (13 rows on 2026-11-01).

## 0. Dates and limits that shape what you can write

| Item | Fact | Source |
|---|---|---|
| Kaggle closes | Sunday 29 Nov 2026, 23:59 HKT = 2026-11-30 00:59:59 JST | D006; `project.yaml` |
| Private leaderboard | Shown only after the close. Until then the archive has CV, expected and public scores only | `docs/ARCHIVE_RULES.md` §5 |
| Reflection deadline and length | Not archived; check Canvas. Use private scores in Q4 only if they are public when you write | `docs/ASSIGNMENT_CHECKLIST.md` §F |
| Canvas checkpoint | Date not archived. Kaggle display names must be stated there; the team is fixed at the checkpoint | Rules 1–2 in `docs/COMPETITION_RULES.md` §1 (read 2026-11-01; D014) |
| Team | `project.yaml` team block is TODO. On 2026-11-01 the Kaggle team had one member, and all six uploads came from that account. Teammates count as "inside the team" only after the Kaggle merger | `project.yaml`; `docs/COMPETITION_RULES.md` §3–4; D014 |
| Generative AI | Allowed; the report states which tool and for what; every member must be able to explain every detail of the submission | Rule 6 in `docs/COMPETITION_RULES.md` §1 |

## 1. Question 1: your main role and the extension you owned

### 1.1 What the archive records about people

- `registry/experiments.csv`: 214 runs (138 experiment, 69 analysis, 7 extension). Author `TODO_ID_1` in all 214. This is the default author in `project.yaml`; nobody set `SSC_AUTHOR`.
- `registry/submissions.csv`: S001–S006, author `TODO_ID_1`.
- `registry/compute_jobs.csv`: 90 Colab jobs (51 on account pro1, 38 on pro2, 1 without a label); author column empty.
- `registry/journal.md`: author "TODO" in J1–J7. `registry/beliefs.md`: author "TODO (AI-assisted)" in B001–B008. `registry/ideas.md`: proposer "AI (Claude Code)" in I001–I014.
- `registry/ai_usage.md`: all 13 rows name "AI (Claude Code)" as operator and leave the member (row 8: the extension owner) as TODO. "Verified by" is TODO or an AI re-check in every row.
- git: no commit exists (D004), so `git log` gives no evidence.
- `project.yaml` team block and the "Author:" line of all four extension drafts: TODO.
- Dates: all 214 runs, all 90 Colab jobs and every registry entry up to D013 are dated 2026-10-22. Nothing was archived from 2026-10-23 to 2026-10-31. Entries dated 2026-11-01 (D014, `docs/COMPETITION_RULES.md`, this sheet) come from an AI drafting round on that day.

So the archive cannot show any member's personal contribution. What counts is your own evidence (messages, notes, your own journal lines from now on) and your ability to explain the work.

### 1.2 Human actions the archive records, without a name

The archive calls this person "the user" (用户) or "the member". Only the person who did these may claim them.

| Action | Source |
|---|---|
| Asked that the setup make no git commit | D004 ("Decided by: 用户") |
| Chose Google Colab for all formal runs instead of the laptop, and made two Colab accounts (pro1, pro2) available | D005 |
| Works in Japan time; the archive clock is JST | D006 |
| Authorised the AI to upload submissions on its own within the plan (at most 3 per day, each pre-registered) | D008; ai_usage row 4 |
| Asked to stop the verify stage of research round 1 for a computer restart | J3 ("Did") |
| Logged in to Kaggle in the Chrome window the AI drives, so that S005 and S006 could be uploaded | J6, J7; ai_usage row 4 |
| Wanted all four extensions done | `research/reports/verify_rules.md` §4 ("四个都要") |

Team actions that the archive lists as still open: confirm the decisions whose decider or team confirmation is TODO (D001–D003, D007, D009–D014); save the Kaggle rules page as PDF (D014); fill "Member" and "Verified by" in `ai_usage.md`; set a git identity and make the first commit; sync the Windows clock (D012); confirm who owns the pro2 account; ask the instructor whether unlabelled test features may be used (D007, D010); fill the team block (J5 and J6 "Next"; `research/reports/fix_round1.md`).

### 1.3 Work streams and who operated them

| Work stream | What was done | Evidence | Operated by (archive) | ai_usage row |
|---|---|---|---|---|
| Setup | venv and lock file, data checksums, archive rules, `ssc` package, scripts, E1–E4 scaffolds, report templates, baseline configs, ideas I001–I009 | D001, D003, D004; `README.md`; `docs/`; J1 | Claude Code, main session | 1 |
| Compute | Colab CLI inside WSL; `colab/remote.py` (code bundles, log streaming); 90 jobs | D005, D012; `colab/README.md`; `registry/compute_jobs.csv` | Claude Code; the user chose Colab and provided the accounts | 2, 12 |
| Research day 1 | R001 feature structure, R002 informative block, R003 eight candidates under the official CV; class `ssc.transductive.TransductiveGMM` | `20261022-051627_r001_data_structure`, `20261022-053218_r002_informative_subspace`, the eight R003 runs; J2 | Claude Code | 2 |
| Research round 1 | generator (gen, R150–R156), mixtures (mix, R100–R106), supervised models (sup, R200–R249, 587 configurations), foundation and Bayesian models (fm, R250–R257) | `research/reports/{gen,mix,sup,fm}.md`; J3 | 6 parallel Claude Code agents (workflow `wf_13564004-279`) | 3 |
| Extensions E1–E4 | real-data runs and the four 2-page drafts | §1.4; `research/reports/ext12.md`, `ext34.md` | agents ext12 and ext34; corrected by fix_ext12 and fix_ext34 | 3, 8, 10 |
| Verification round 1 | six separate AI re-checks (leakage, reproduction, generator, candidates, extensions, rules), R900–R945, one synthesis | `research/reports/verify*.md`; J5 | 6 AI verifiers and 1 AI synthesis agent | 7 |
| Fix round and research round 2 | 29 corrections; R400–R404, R420–R426, R440–R441; code fixes and tests; final check | `research/reports/fix_round1.md`, `r2_*.md`; J6 | 9 agents (workflow `wf_5abda016-8ac`) | 9, 10, 11, 12 |
| Kaggle submissions | S001–S006 created with `make_submission.py`, uploaded through the Kaggle web page in the member's logged-in Chrome, recorded with `record_lb.py` | `registry/submissions.csv`; B001–B007; J2, J5, J7 | Claude Code, after authorisation (D008) | 2, 4, 13 |
| Registry | D001–D013, B001–B008, I001–I014, J1–J7, references (D014 added on 2026-11-01) | `registry/*.md` | Claude Code (main session and agents) | 1, 5, 9, 13 (D014: no row yet) |
| Archive tools | `scripts/archive_check.py`; snapshot `backups/snapshot_20261022-084506.zip` | J4 | Claude Code | 6 |
| Core report, contribution table, video, reflections | templates only at the start of 2026-11-01 | `reports/report_core.md`, `reports/contribution_table.md`, `video/PLAN.md`, `reports/reflections/TEMPLATE.md` | later drafts: see the rows added to `ai_usage.md` after row 13 | – |

### 1.4 The four extensions: what an owner must be able to explain

Each draft was written by an AI agent and corrected after verification. Each has "Author: TODO" and a visible AI-use line. The header comments of E1 and E2 say: "The owner must check and be able to explain every number"; the same holds for E3 and E4. Page counts are XeLaTeX trial exports (Calibri 11 pt, A4, 2.54 cm margins; `research/reports/fix_round1.md` §5).

**E1 Validation protocol design** (`reports/extensions/e1_validation_protocol.md`)

- Question: when only the split seed changes, which scheme can choose between two models 0.01 AUC apart?
- Runs: `20261022-064054_e1_validation_protocol` (linear pair: in-fold |corr| top-30 vs top-50 + L2 logistic regression, C = 0.05; CV 0.7910 vs 0.8011); `20261022-064148_e1_validation_protocol_qda_pair` (QDA reg 0.1 vs 0.3; 0.9958 vs 0.9832); comparison R306 `20261022-064412_r306_e1_pair_comparison`; pair screen R300 `20261022-063722_r300_e1_pair_screen`; check R940 `20261022-091211_r940_e1_pairs_check`. Code: `research/r300_e1_pair_screen.py`, `research/r306_e1_both_pairs.py`, `extensions/e1_validation_protocol/`.
- Results: 8 schemes × 50 split seeds. MDD = 2 sd(A − B): 20 % hold-out 0.057; 5-fold 0.030; 5 × 10 0.0083 [0.0070, 0.0104]; 5 × 20 0.0061 [0.0051, 0.0076]. More folds do not help; more repeats do (about 1/√R). At 80 % power 5 × 20 gives 0.0086 [0.0071, 0.0107]. The studied pair is only about 0.004 apart (R940); on a new 250-row sample the gap would move with sd 0.0105 (R941).
- Length 1.94–1.97 pages.

**E2 Learning curve** (`reports/extensions/e2_learning_curve.md`)

- Question: AUC with 50–250 training rows (10 subsets per size) and the expected AUC with 500 rows.
- Runs: L1 baseline `20261022-070248_e2_learning_curve`; QDA `20261022-070339_e2_learning_curve_qda_block`; GMM `20261022-070445_e2_learning_curve_gmm`; comparison R307 `20261022-070518_r307_e2_model_comparison`; label requirement R304 `20261022-070600_r304_label_requirement`; checks R941 `20261022-092201_r941_e2_replica_truth` and R942 `20261022-094142_r942_e2_coverage`.
- Results: L1 rises from 0.659 (n = 50) to 0.847 (n = 250). The power law gives 0.906 at 500 rows, but its nominal 90 % bootstrap interval covered the replica truth in only 21 % of replicas (R942; bias −0.031), so the draft quotes about 0.91–0.92 ± 0.04. QDA gains +0.0008. The GMM scores 1.0 at every size, and with only 2 labels (R304).
- Length 1.89–1.97 pages.

**E3 Information leakage** (`reports/extensions/e3_leakage.md`)

- Question: how far leaky feature selection (top-k |corr| on all 250 rows before CV) inflates CV compared with honest in-fold selection and with the leaderboard, and how the gap depends on k_feat and k_fold.
- Runs: `20261022-063908_e3_leakage` (Colab job `J20261022-064045-pro1-rext34`); S004 with B005; proxy R321 `20261022-072942_r321_e3e4_proxy_test`; replicas R944 `20261022-161150_r944_e3_replica_decomp`. Code: `extensions/e3_leakage/run.py`.
- Results: at k_feat = 20, leaky CV 0.867, honest CV 0.793, gap 0.073 (SD over 20 split seeds 0.014; Nadeau–Bengio corrected SE 0.020; p = 0.0003). S004 scored 0.79136 on the public LB: leaky CV 0.075 above it, honest CV 0.002 above. The gap is 0.001 at k_feat = 7, up to 0.090 at k_feat = 30–50 and 0 at 150. For 2 / 3 / 5 / 10 folds it is 0.087 / 0.082 / 0.073 / 0.063. Leaky CV would pick k_feat = 50, honest CV k_feat = 7.
- Length 1.74 pages.

**E4 Stability of the feature ranking** (`reports/extensions/e4_feature_stability.md`)

- Question: how often each feature enters the bootstrap top-k by |corr|, whether the sorted frequency drops sharply, and whether the stable features are worth keeping.
- Runs: `20261022-071414_e4_feature_stability` (B = 2000 resamples; k = 5, 10, 13, 20); R320 `20261022-074624_r320_e4_stable_vs_block`; null check R943 `20261022-100104_r943_e4_null_step`.
- Results: features 98, 17, 100, 112, 80, 77 and 46 enter the top-k in ≥ 98.5 % of resamples for k = 10, 13 and 20. The largest step is exceeded by 1 of 50 permuted-label curves (p ≈ 0.02–0.04). The stable set always equals the top-m set by |corr|. With L2 logistic regression it beats all 150 features (+0.063, p = 0.038), but the label-free 13-feature block beats it for every model family: L2 0.862 vs 0.832, QDA 0.996 vs 0.961, GMM 1.000 vs 0.983.
- Length 1.79–1.94 pages. The 4-panel figure has very small text (`fix_round1.md`).

### 1.5 Questions to settle before you write Q1

- Which one extension do you own? It must differ from your teammates' extensions and match the contribution table and the draft's "Author:" line.
- What did you do yourself (for example: set the scope, made or confirmed a decision, authorised or checked uploads, re-ran a run, rewrote a section, checked numbers, recorded the video)? Where is the evidence?
- What did the AI do in that part? Cite the ai_usage row(s).
- What did you check, and how? The "Verified by" cell of those rows should say the same thing.

## 2. Question 2: ideas considered, and whether they were accepted or rejected

All ideas I001–I014 were proposed by the AI (`registry/ideas.md`, column "Proposer"). Their status was also set by the AI; team acceptance is not recorded. If you had an idea yourself, append it to `ideas.md` with your id and the true date. Do not back-date.

### 2.1 Final-model ideas

| ID | Idea | Status | Evidence | Reason given in `ideas.md` (short) |
|---|---|---|---|---|
| I001 | L2 logistic regression on all 150 features (C = 0.05) | rejected | `20261022-063707_r300_l2_all` 0.777; `20261022-065651_r200_l2_all` 0.787; `research/reports/sup.md` | worse than L1 on all features (0.78–0.79 vs 0.85–0.87); on the 13-feature block only 0.93–0.94, far below QDA 0.9999: the signal is in the class covariances |
| I002 | sparse L1 logistic regression | accepted (baseline) | `20261022-050647_r003_sup_l1_all`; S001 | first step of the submission history, CV 0.847 |
| I003 | shrinkage LDA | rejected | `20261022-050653_r003_sup_lda_hv` 0.878; R200 runs 0.796–0.859 | LDA uses only mean differences; QDA on the same block reaches 0.9958–0.9999 |
| I004 | in-pipeline top-k \|corr\| filter + L2 logistic regression | rejected as final model; kept as the object of E1 and E3 | R300 `corr10`–`corr50` runs 0.791–0.812; `20261022-070429_r200_l2_corrk` 0.848; E3 honest CV 0.793, S004 public 0.79136 | the label-free variance block is better (L2 0.93 vs 0.79–0.85; QDA 0.9999 vs 0.993) |
| I005 | Lasso regression on the 0/1 target | pending (never tested) | none | low priority: I002 covers sparse linear models, and linear models stay ≤ 0.94 |
| I006 | tree models (random forest, extra trees, LightGBM, XGBoost) | rejected (negative result) | R203 runs 0.861–0.914 | axis-aligned splits cannot follow the 13-dimensional quadratic boundary |
| I007 | bootstrap stability selection inside the pipeline | rejected as a selector; kept as the object of E4 | E4 run; R320 | the stable set equals the top-m \|corr\| set; the block is better for L2, QDA and GMM |
| I008 | rank-average ensemble of linear models | pending (never tested) | none | the best linear model is about 0.94, QDA and GMM 0.9999–1.0 |
| I009 | use the 19,750 unlabelled test features | accepted | D007; R003; S003 | test features are competition data; unsupervised steps never see labels. The instructor has not been asked yet (D010) |
| I010 | supervised QDA on the 13 high-variance features | accepted | `20261022-050659_r003_sup_qda_hv`; S002 | CV 0.9958, 0.12 above LDA |
| I011 | transductive GMM on train+test features; labels only name the two components | accepted (final model) | `20261022-050708_r003_gmm2_hv`; S003 | CV 1.0000 in all 100 folds; public rank 1 at upload (J2) |
| I012 | improve the ~1 % boundary rows (feature discovery, BIC choice of K, semi-supervised EM, seed ensembles) | rejected | `research/reports/mix.md` (R100–R106) | no measurable gain: S003 is about 2e-7 below the Bayes limit; no refinement gained more than 1e-8, and losses reached 2.4e-8 (bagging) and 3.7e-6 (two components per class) (R103); the remaining public error most likely comes from label noise (about 5:1, R922) |
| I013 | reverse-engineer the data generator and build replica datasets | accepted | `research/reports/gen.md` (R150–R155); `research/lib_gen_replica.py` | generator identified (`make_classification` plus a per-column affine map); replicas match the real statistics |
| I014 | pretrained tabular foundation models (TabPFN v2, TabICL) on the 250 rows | rejected for submission; analysis only | `research/reports/fm.md` (R250–R253) | below QDA and Bayesian QDA (TabICL 0.9929 on all features, 0.9989 on the block; TabPFN v2 0.9851 / 0.9888); rule-4 status unclear |

### 2.2 Decisions and the alternatives they rejected

| Decision | What was chosen | Alternatives, and why they were rejected | Source |
|---|---|---|---|
| D001 | isolated venv `.venv` | conda env with `PYTHONNOUSERSITE=1` (easy to forget in each terminal); deleting the user site-packages (affects other projects) | D001 |
| D002 | stratified 5-fold × 20 repeats, seed 2026; paired differences and the Nadeau–Bengio test | single hold-out (too noisy); 10-fold (25 validation rows per fold); leave-one-out (no per-fold AUC); 5 × 10 (only 50 paired values). E1 supports the choice; the planned freeze decision after E1 is not written yet | D002; E1 |
| D003 | synthetic sandbox kept apart from the real archive | one archive with a data-source column (easy to misuse); no synthetic data (no end-to-end test) | D003 |
| D005 | formal runs on Colab, with each run's code bundle saved (not achieved for the 47 `--no-sync` runs or the runs before D005, S003 included; D012) | local runs (the user did not want them); Colab web notebooks (no automatic archive); waiting for the first commit (slows the work) | D005, D012 |
| D007 | allow unsupervised steps on train+test features | use only the 250 labelled rows (best supervised model at the time 0.9958) | D007 |
| D008 | at most 3 uploads per day, each pre-registered; do not chase the public LB | – (the public LB has about 395 rows; SE about 0.0015 near 0.999) | D008 |
| D009 → D011 | "CV is saturated and cannot separate tuned supervised QDA from the GMM" | D009's "indistinguishable": literally true but uninformative (in 222 replicas the CV test was never significant, yet supervised QDA was always slightly worse) | D011 |
| D010 | finals S003 + S005 (r201) by default; S003 + S006 (r209) if test features are explicitly allowed | S003 + r209 as the default: +1.45e-5 expected AUC, but it shares S003's rule risk. Foundation-model runs and S004 are never selected | D010; `research/reports/verify.md` §6 |
| D013 | if test features are forbidden: ext-grid QDA (R930) + r201 | r201 + r210 (about half the gain); train-only Bayesian QDA r255 (below r201) | D013; `research/reports/r2_inductive.md` |
| D014 (2026-11-01) | rules archived as a paraphrase; rule basis for D007 (rule 4 bans outside data; Foundational 18a counts the test set as competition data; 4b bans hand labelling); licence check of the final pipeline (numpy, scipy, scikit-learn, pandas: BSD-3-Clause) | copying the rules page word for word (someone else's long text); waiting for the instructor's answer before archiving | D014; `docs/COMPETITION_RULES.md` |
| Draft selection rule | `docs/VALIDATION_PROTOCOL.md` §5: tolerance band, then the simplest model, from a clean commit | not used: CV saturated at 1.0 (D007 "Consequences"), so D010 ranks by replica-based expected private AUC and by rule risk; no run comes from a commit (D004; `verify.md` E7) | VALIDATION_PROTOCOL §5; D010 |

### 2.3 Choices inside each extension

- **E1.** The pair was fixed by a rule written into R300 before E1 ran (of 11 neighbouring pairs, the one whose official-CV gap was closest to 0.01). This rule picked a pair whose gap on the official seed was a 1.84-sd draw; its real gap is about 0.004 (R940). A second pair near the AUC ceiling (QDA reg 0.1 vs 0.3) was added; its split noise is 4.5–6.1 times smaller. The criterion MDD = 2 sd means about 50 % power; the 80 % power version and the 28-pair check (R940) were added after verification. The GMM was not studied because its CV is saturated at 1.0. Sources: E1 draft; `research/reports/ext12.md` errata.
- **E2.** Three models on identical subsets instead of one. Curve forms compared: power law, log-linear, unweighted, without n = 50; their union is the "envelope". The R308 back-test was dropped as an argument (one sample cannot show the procedure's average behaviour), and the bootstrap interval was replaced by an honest ±0.04 after R942. Side result: L1 with C = 0.2 scores 0.8729 against 0.8469 at C = 0.1 (`20261022-063711_r300_l1_c02`). Sources: E2 draft; `ext12.md`.
- **E3.** Both meanings of "k" were covered: k_feat (10 values) × k_fold (2, 3, 5, 10). The protocol includes permuted-label nulls and the k_feat = 150 sanity check. The expected S004 score came from a proxy: AUC against the GMM's labels on all 19,750 test rows (R321); `ext34.md` notes that this proxy favours models close to the GMM. R944's replica decomposition was added after verification. One figure was removed to fit 2 pages. Sources: E3 draft; `research/reports/ext34.md`.
- **E4.** Four drop rules. *perm* was pre-registered as the main rule. *binom* is far too wide (under the null the top noise feature reaches a frequency of 0.84 at k = 10), and *half* admits pure-noise features in 72–100 % of null curves for k ≥ 10 (R943). Stability selection was never significantly better than honest top-m. R320 compared the stable set with the label-free block for three model families. One table was merged into the figure to fit 2 pages. Sources: E4 draft; `ext34.md`.

### 2.4 Ideas dropped for rule or licence reasons

- Foundation models: R250 checked licences. The TabPFN v2 weights (Prior Labs License, Apache-2.0 plus an attribution clause) and the TabICL checkpoint (BSD-3) download without a token. The default TabPFN-2.5 weights of the `tabpfn` package carry a non-commercial licence and need a token, so they were not used. Foundation-model runs are analysis only and never part of a submission (D010 item 4). Whether rule 4 allows pretrained models was not confirmed, and D014 notes that it is unclear whether the Prior Labs licence counts as an OSI licence under Foundational rule 6c (I014; D014 item 5; `research/reports/fm.md`).
- No attempt to recover the generator's random seed or the hidden test labels; replica rows never train a submitted model (D010 item 6; `research/reports/gen.md`, compliance paragraph).
- GMM pseudo-labels on the test rows served only as an analysis proxy, never to train a submitted model (`research/reports/ext34.md`).
- Doing all four extensions: only one extension per member is graded, so with 1–3 members the rest can only support the core report (`research/reports/verify_rules.md` §4).

## 3. Question 3: difficulties

The journal's "Difficulties" lines are the archived source for this question (`docs/ARCHIVE_RULES.md` §7). All of them describe the AI's operating problems or team-level problems; no member wrote one. Personal difficulties are not in the archive, so only you can describe yours.

### 3.1 Journal "Difficulties" lines

| Entry | Difficulty | What was done |
|---|---|---|
| J1 | The conda environment picked up old packages from the user-level site-packages | Isolated venv (D001) |
| J1 | scikit-learn 1.9 deprecated `LogisticRegression(penalty=...)` | Configs use `l1_ratio` with liblinear or saga |
| J1 | No git identity on the laptop; the user asked for no commit | D004: no commits; every run is recorded as `no-commit` |
| J2 | The Colab CLI does not run on Windows; WSL Ubuntu had no pip, and sudo needed a password | CLI installed in WSL with an existing micromamba |
| J2 | The first Colab session (US East) stalled: each `colab exec` took over 75 s; a new session (Asia) took 2–3 s | `remote.py` rewritten: one resumable log stream, merged status checks, automatic rebuild of a dead session (about 1 hour) |
| J2 | pandas writes different line endings on Windows and Linux, so the synthetic-data hashes differed | Compare by content |
| J2 | The gzip header timestamp forced a data re-upload every time | Fixed |
| J2 | A temporary HKT clock made the R003 run ids 1 hour early | Not renamed (append-only); explained in D006 |
| J2 | A workflow fix agent was duplicated by accident; two copies edited `src/ssc` at the same time | One copy was stopped |
| J3 | Two bugs in `remote.py`: `prepare()` lacked a `return`; `wsl --` let the outer shell expand `$A` and `$C` in the pro2 prefix to empty strings | Fixed (`wsl -e`); ext34 ran with `--no-sync` (code bundle `b413e6aa3bae4443`) |
| J3 | Seven High-RAM Colab sessions (sb1–sb7) that are not this project's were running | Left untouched |
| J5 | A usage limit interrupted the candidate verifier in the morning | Resumed in the afternoon with the existing scripts and runs; the replica job paused from 09:44 to 16:05 |
| J5 | R900 Null A failed (BrokenProcessPool: worker processes could not unpickle a cached helper) | Fixed and re-run; the failed run is archived as failed |
| J5 | The rules verifier could not run the `package_deliverables.py` dry run (blocked by the permission system) | Not run |
| J5 | Verifiers disagreed: S003's private expectation, the second final, and whether E3's ±0.014 is an SD or an SE | Settled item by item in `research/reports/verify.md` §3 |
| J5 | Time used: verification runs 09:10–10:06 and 16:05–17:23 JST; synthesis about 17:55–18:30 | – |
| J6 | A usage limit stopped the candidate and synthesis agents at about 10:15 JST | Resumed after the 13:10 reset from saved results |
| J6 | The pro2 account twice failed to get a second Colab machine (4 other sessions on the account) | The r2_inductive jobs ran one after another on one session |
| J6, J7 | No Kaggle API credentials on the laptop, so uploads went through the web page; after 16:10 the Chrome extension disconnected, and the reconnected Chrome was not logged in to Kaggle | S005 and S006 waited until a member logged in |
| J6 | Three large `.npz` files (12 MB) grew each code bundle from 0.5 MB to 13.5 MB (`logs/code_bundles` reached 75 MB) | Separate "research data" channel; bundles back to 1.3 MB (D013 item 4) |

### 3.2 Difficulties recorded elsewhere

| Difficulty | Evidence | Source |
|---|---|---|
| Three failed runs | `20261022-065046_r152_calibration` (workers could not unpickle a SciPy function); `20261022-071440_r201_qda_all` (scikit-learn's QDA needs more rows per class than features: about 90 rows per class against 150 features); `20261022-091033_r900_perm_null_a` | `notes.md` "Conclusion" of each run |
| Claims that verification found overstated or wrong | e.g. S003 private expectation 0.998 → about 0.997; "flip_y = 0 nearly impossible" → disfavoured about 5:1; D009's wording; the E2 interval covering 21 % instead of 90 %; 15 numbers in the extension drafts that did not match the archive | `research/reports/verify.md` §2, §4 |
| Page limits | The drafts first typeset at 2.16–2.71 pages; cut to 1.74–1.97. With Word's default paragraph spacing E1, E2 and E4 exceed 2 pages | `research/reports/verify_ext.md`; `fix_round1.md` §5 |
| Rule uncertainty | The rule text was not archived until 2026-11-01 (D014). The instructor has not confirmed the use of unlabelled test features (D007) or of pretrained models (I014); D010 has one case per possible answer | D007, D010, D014, I014 |
| CV saturation | The official CV cannot separate candidates above 0.9999 (222/222 replica tests not significant) | D011; `reports/tables/r936_cv_power.csv` |
| Public-LB noise | 395 rows; Hanley–McNeil SE 0.0011 near 0.9995, 0.023 near 0.79; the public differences among S003, S005 and S006 are within noise | `scripts/record_lb.py`; D010 |
| Clock skew | Laptop about 100 s ahead of the Colab VMs | D012 |
| Code provenance | 47 `--no-sync` runs have no code bundle; no run comes from a git commit | D012; `verify.md` §2 E5, E7 |
| Team set-up | No names, student IDs, Kaggle names or extension owners in the archive; one Kaggle participant; checkpoint and reflection dates not archived | `project.yaml`; `verify_rules.md` §4 |

Still open at the start of 2026-11-01: core report, contribution table, video and reflections were templates; D002's freeze decision (planned for 2026-11-01 in `docs/TIMELINE.md`) and the team confirmations were not written; the instructor had not been asked; the final selections on Kaggle are made only at the end (D010); B004 and B008 wait for the private LB.

## 4. Question 4: beliefs held before a leaderboard result

All beliefs B001–B008 were written by the AI before the matching upload (`registry/beliefs.md`, author "TODO (AI-assisted)"; ai_usage rows 4, 5 and 13). Use one only if you held it too at the time, and say who wrote it.

### 4.1 Pre-registered beliefs and their outcomes

| ID | Belief, as registered before the result | Confidence | Result | Outcome and lesson recorded |
|---|---|---|---|---|
| B001 | S001 (L1, all features) public ≈ 0.85 (CV 0.847) | 70 % within ±0.04 | 0.83589 | confirmed (−0.014 = −0.7 SE); the linear CV estimate was accurate |
| B002 | S002 (supervised QDA, 13 features) public ≈ 0.99 | 70 % | 0.99277 | confirmed (+0.6 SE); the signal is in the covariance structure |
| B003 | S003 public ≈ 0.999, not exactly 1.0, because about 1 % of the test rows lie on the boundary | 60 % | 0.99923 | confirmed. Lesson: "CV is saturated at 1.0; the remaining gap can only be found on boundary rows" (later contradicted, §4.3 a) |
| B004 | S003 private ≥ 0.999, and within 0.001 of public | 70 % | pending (private LB after the close) | verification gives this a probability of 0.14–0.17 (R922), so B004 is miscalibrated. It stays unchanged; B008 was added |
| B005 | S004 public ≈ 0.82, 90 % in 0.785–0.854 (taken from the 19,750-row GMM proxy, 0.819); at least 0.02 below the leaky CV 0.867 | 90 % / 80 % | 0.79136 | both parts confirmed. 0.029 below the point (−1.2 SE); the honest CV 0.793 was closer. Lesson: use the honest CV as the expectation |
| B006 | S005 (r201) public ≈ 0.9988, 80 % in 0.9980–0.9997; P(above S003) ≈ 20 % | 80 % | 0.99953 | confirmed (+0.6 SE), and above S003, which B006 gave about 20 % (R421, R425: 14–19 %) |
| B007 | S006 (r209) public ≈ 0.9991, 80 % in 0.9987–0.9996; P(above S003) ≈ 25–28 % | 80 % | 0.99956 | confirmed (+0.4 SE), and above S003, which B007 gave 25–28 % |
| B008 | S003 private: median ≈ 0.997, 90 % in 0.992–0.9996; P(≥ 0.999) ≈ 0.15; private ≈ 0.002 below public | 90 % | pending | – |

### 4.2 Expected and public score of each submission

Surprise = public − expected, in units of the Hanley–McNeil SE of a 395-row public AUC (179 positives) at the observed score, as printed by `scripts/record_lb.py`. `reports/tables/r440_journey.md` uses the SE at the expected score instead, so its values differ slightly (for example −1.35 for S004).

| Sub | Model (run) | CV mean (sd) | Expected public | Public | Public − expected | SE | Uploaded (JST) |
|---|---|---|---|---|---|---|---|
| S001 | L1 logistic, all features (`20261022-050647_r003_sup_l1_all`) | 0.8469 (0.0120) | 0.85 | 0.83589 | −0.0141 (−0.7 SE) | 0.0210 | 10-22 06:14 |
| S002 | QDA reg 0.1, 13-feature block (`20261022-050659_r003_sup_qda_hv`) | 0.9958 (0.0011) | 0.99 | 0.99277 | +0.0028 (+0.6 SE) | 0.0045 | 10-22 06:15 |
| S003 | transductive GMM (`20261022-050708_r003_gmm2_hv`) | 1.0000 (0) | 0.999 | 0.99923 | +0.0002 (+0.2 SE) | 0.0015 | 10-22 06:16 |
| S004 | E3 final model (`20261022-063908_e3_leakage`) | 0.7934 (0.0142) | 0.82 | 0.79136 | −0.0286 (−1.2 SE) | 0.0233 | 10-22 16:08 |
| S005 | strictly inductive QDA (`20261022-071234_r201_qda_block`) | 0.9999 (0.0002) | 0.9988 | 0.99953 | +0.0007 (+0.6 SE) | 0.0011 | 10-22 22:21 |
| S006 | self-training QDA-LW (`20261022-080632_r209_selftrain_qda_lw`) | 1.0000 (0) | 0.9991 | 0.99956 | +0.0005 (+0.4 SE) | 0.0011 | 10-22 22:22 |

Uploads per Kaggle day (UTC): S001–S003 on 2026-10-21, S004–S006 on 2026-10-22; the team limit is 3 per day (D008).

### 4.3 Where the leaderboard went against a belief

**(a) "The remaining gap is a ranking problem on the boundary rows."** This is B003's lesson; J2 says the same ("whether we can improve further depends on these boundary samples"), and I012 tried it. S003's public score of 0.99923 is far below the generator's clean-label Bayes AUC of 0.99995, while S003 itself is only about 2.6e-7 below the Bayes limit on replicas (R921). With clean labels a public score this low appears in about 0.1 % of simulated public splits (R103 in `mix.md`; R922: 0.00075). With sklearn's default label noise it appears in 83 % (R103). So the public score is the main evidence of label noise (flip_y MLE 0.003, likelihood support 0–0.015; R922), and no boundary refinement gained more than 1e-8 (I012). Sources: B003, J2, I012, `research/reports/mix.md`, `research/reports/verify_gen.md` §3.

**(b) B004: "S003 private ≥ 0.999" at 70 %.** The leaderboard has not tested it yet; private scores appear after the close. Verification that used the public score puts its probability at 0.14–0.17 (0.51 only under a log-uniform prior; `reports/tables/r922_posterior_by_prior.md`). The re-registered B008 expects about 0.997. If your reflection is due after the private LB appears, compare B004 and B008 with S003's private score (`private_lb` in `registry/submissions.csv`, recorded with `record_lb.py --private`).

**(c) B005: the expected S004 score came from a proxy.** The 0.82 was the model's AUC against the GMM's labels on all 19,750 test rows (R321). The public score was 0.79136: 0.029 below the expectation (−1.2 SE, inside the pre-registered 90 % range), while the honest CV (0.793) was within 0.002. The gap is within public-LB noise (SE 0.023), so the leaderboard alone does not prove the proxy wrong. The lesson recorded in B005 is to use the honest CV as the expectation. Sources: B005; `reports/tables/e3_leaderboard.csv`.

**(d) B006 and B007: S005 and S006 came out above S003.** Both point predictions were below S003's public score, and the chance of "above S003" estimated before the uploads was about 20 % for S005 (14–19 % in R421 and R425) and 25–28 % for S006. Both came out above it (+3.0e-4 and +3.3e-4). The rule D010, written before these scores were seen, treats public differences of this size as noise (sd of the difference to S003 3.2–6.5e-4; 0.0003 is about 12 pairs of rows). On 222 replicas S003 still has the highest expected private AUC (r201 about −2.5e-4, r209 about −7e-5; `reports/tables/r936_private.csv`). The final selection did not change. Sources: B006, B007, J7, D010.

**(e) "Only the transductive model reaches the top."** R003 and D007 compared the GMM with an under-tuned QDA (0.9958, S002 public 0.99277). Tuned strictly supervised QDA reached CV 0.9999 (r201; D009, D011), and S005 scored 0.99953 on the public LB, above S003. So a model that uses no test features can match S003 on the public rows. The replicas still put it lower in expectation (−2.57e-4), and the difference shows only on boundary rows (AUC 0.804 for S003 vs 0.664 for r201; R404). Sources: D009, D011, D013, `research/reports/sup.md`, `research/reports/r2_inductive.md`.

### 4.4 Using a belief honestly

- Every B row was written by the AI before the upload. If you did not hold that belief yourself, do not present it as yours. You can write about how you read the AI's prediction at the time, or about a belief you did hold (say when and how you held it).
- The template asks for a belief that the leaderboard showed to be wrong. No B row was refuted by a leaderboard score. Cases (a), (d) and (e) are where the public score went against an expectation or its explanation; (b) can only be settled by the private score.
- Allow for noise. The Hanley–McNeil SE of a 395-row public AUC is about 0.021 at 0.84, 0.0045 at 0.993 and 0.0011–0.0015 near 0.999 (§4.2). A gap below about 2 SE is ordinary noise.
- To have a belief of your own, append it to `registry/beliefs.md` now, with your id and today's date, before the private LB appears (for example about S003's or S005's private score). Do not edit earlier rows.

## 5. Before you upload your reflection

- Written in your own words; no sentences copied from this sheet, the extension drafts or the research reports.
- Every claim about your own work is true, and you can explain it in detail (rule 6).
- AI use is stated: the tool (Claude Code, Anthropic, Claude Opus 5.5), what it did in your part, and what you checked. It matches `registry/ai_usage.md` and the contribution table.
- Your extension is named and matches the contribution table.
- Q2 has at least one rejected idea with its evidence; Q3 at least one difficulty with what was done about it; Q4 one belief with the leaderboard evidence and an allowance for noise.
- Every number traces to a run id, sub id or table.
- Private scores are used only if they were public when you wrote.
- Length and deadline as on Canvas; exported to PDF; uploaded by you.
