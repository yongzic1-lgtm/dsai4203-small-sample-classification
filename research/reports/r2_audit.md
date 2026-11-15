# r2_audit：核心报告数字核查 + R206 标签预算独立重跑（研究第 2 轮）

- 作者：AI 研究代理 r2_audit（Claude Code，Claude Opus 5.5），2026-10-22 JST。
- 计算：Colab 账号 pro1，会话 `r2aud`（用完已停）。作业：`J20261022-182157-pro1-r2aud`（R441）、`J20261022-183505-pro1-r2aud`（R440 第一版）、`J20261022-184159-pro1-r2aud`（R440 定稿）。
- 新 run（都是 `kind = analysis`，都写了 Conclusion）：
  - R441 `20261022-182022_r441_label_budget_rerun`：R206 标签预算曲线的独立重跑。
  - R440 `20261022-184020_r440_core_numbers`：核心报告数字核查 + R206/R304/R441 对比 + journey 表。它替代第一版 `20261022-183328_r440_core_numbers`（数字相同；定稿只改了 journey 表的一个措辞，并多查 2 条数据事实）。
- 代码：`research/r441_label_budget_rerun.py`、`research/r440_core_numbers.py`、`research/r440_archive_pack.json.gz`（R440 在 Colab 上读的档案副本，见 §English 1.2）。
- 表/图：
  - `reports/tables/r440_core_numbers.{csv,md}`：claim, value, source file, verified yes/no, note, 外加 id、section、archived_value、status、claimed_in。
  - `reports/tables/r440_r249_recheck.csv`：r249 对比表 64 行逐行重算。
  - `reports/tables/r440_journey.{csv,md}`：可以直接放进报告的 journey 表。
  - `reports/tables/r440_label_budget_compare.{csv,md}`、`r440_label_budget_first_hit.{csv,md}`、`reports/figures/r440_label_budget_compare.png`。
  - `reports/tables/r441_label_budget.{csv,md}`、`reports/figures/r441_label_budget.png`。
- 我没有上传 Kaggle，没有改 `registry/`、`decisions.md`、`beliefs.md`，也没有改第 1 轮的报告。要改的地方列在英文 §5。

---

## 中文摘要（给队友）

**一句话**：核心报告会用到的 242 个数字里，235 个与档案一致（按报告里印出的位数）；r249 对比表 64 行全部能从 `fold_scores.csv` 重算出来。R206 的标签预算结论在新的折和新的标签抽样上重现了。要改的主要是 2 句措辞，外加一个限定："严格监督约 80 个标签"只对 reg 固定为 0.003 的 QDA 成立，真正的 S005 流程要约 120 个。

**1. R206 独立重跑（R441）**
- 设计：用官方 CV 的 repeat 4–19（R206 用的是 0–3），共 80 个新的验证折。每个预算从 200 行训练折里重新抽 5 次标签（新的随机种子）。候选流程用核查者独立重写、已逐位验证过的 `r931_cand_lib`，R206 的对照模型用 sklearn 重新写，没有用 R206 的代码。
- 结果（达到平均 AUC ≥ 0.99 的最少标签数）：

  | 模型 | R206 | R441 |
  |---|---|---|
  | 直推 GMM（S003） | 4 | 4（2 个标签时已经 1.0，400/400） |
  | 自训练 QDA-LW，阈值 0.75 | 20 | 20（0.9959） |
  | 自训练 QDA-LW，阈值 0.99（S006） | – | 20（0.9991） |
  | QDA reg 0.003，train+test 块 | 40 | 40（0.9925） |
  | 严格监督 QDA reg 0.003，折内块 | 80 | 80（0.9939） |
  | **严格监督 QDA，嵌套调 reg（S005 = r201）** | – | **120**（80 个时均值 0.984，中位数 0.998） |
  | 严格监督 QDA，嵌套 log-loss（r210） | – | 120（80 个时只有 0.938） |
  | 严格监督贝叶斯 QDA（r255 trainonly） | – | 160 |

- 与 R206 的逐格比较：两边都完全有定义的 37 个（模型 × 预算）格子里，最大 |z| = 1.98，没有一格超过 2 个 fold-cluster SE。
- R304（E2）：GMM 在 2 个标签时 50/50 都是 1.0，R441 也是。QDA reg 0.1 的中位数在 75 个标签时过 0.99，R441 在 80 个（中位数）或 120 个（均值），两者一致（设计不同：R304 在其余 250 − n 行上打分）。
- **结论**：D009 证据 1 现在已被独立重现，可以引用。但报告里"strictly supervised QDA needs about 80 labels"要写成"about 80–120"：80 来自 reg 固定为 0.003（这个值是从 R201 的嵌套选择里拿的）；真正的 S005 流程要 120，40 个标签时 400 次里 371 次根本拟合不了。

**2. 数字核查（R440）**
- 核查了 242 条：数据事实、生成器（gen）、S001–S006 的 CV / public / sha256、E3、全部监督消融（sup、r249）、基础模型和贝叶斯模型（fm）、复刻和最终选择（r936）、验证协议（E1、置换零假设）、标签预算。
- 7 条没通过：
  1. sup.md："在全部 150 个特征上，每种方法都不超过 0.88"。**错**。fm 的 TabICL v2 在全部特征上是 0.9929，TabPFN v2 0.9851，horseshoe 0.9261。只在 sup 这条线里成立（最高 0.875）。
  2. fm.md："horseshoe（0.926）是最好的线性模型"。**说过头**。只在全部 150 个特征上成立；在 13 列块上，L2 logistic 是 0.938（r207）。
  3. fm.md：QDA reg 0.03 写的是 0.9990，档案是 0.998950，应写 0.9989。不影响结论。
  4. fm.md："与 S003 判在同一侧的测试行占 99.5%"，档案是 99.554%，应写 99.6%。不影响结论。
  5. "严格监督约 80"对真正的 S005 流程不成立（见上）。
  6. public 榜 395 行：只有 Kaggle 页面上写，档案里没有。请存档比赛页面。
  7. fm.md："BQDA 有 1,493 行测试分数恰好等于 1.0"：需要 `test_pred.npy`，这次没查。
- 通过但要加限定的：
  - R156 的 flip_y 0.0044 和 private 0.9977 数字没错，但已被 R922 取代（0.997，90% 0.992–0.9996）。
  - r254 边界行 AUC（GMM 0.825 对 BQDA 0.702）只有 6 个复刻。
  - `r209_*_r003` 的自训练对比（log-loss 0.0109 → 0.0120）不是嵌套诚实的估计（reg 0.003 来自 R201）。
  - `r206_label_budget.md` 里 40 个标签时严格监督 QDA 的 0.9476 只来自 60 次里的 6 次（其余拟合失败）。不要引用这个数（图里没画它）。
  - gen.md 写"R² ≥ 1 − 4e-14"，档案是 1 − 4.07e-14，严格说略低于这个界。写成"1 − R² ≈ 4e-14"。
  - sup.md 中文写 GPC-ARD 0.978（是 fm 的 `r255_gpc_ard_hv`），英文写 0.976（是 sup 的 `r202_gpc_ard_block`）。一句话只引用一个 run。

**3. Journey 表**（`reports/tables/r440_journey.md`，可直接贴进报告 §3）
- S001 → S002 → S003 → S004，加上已建好但还没上传的 S005、S006，以及 3 个没提交的候选。
- 列：CV（sd）、预登记的 expected public、public、public − CV、surprise（以 expected 处的 Hanley–McNeil SE 计）、复刻上相对 S003 的 private 差距。
- S001–S004 的 surprise 都在 ±1.4 SE 以内：−0.72、+0.55、+0.14、−1.35。

**4. 建议**
- 主会话：D009 证据 1 改引 R206 + R441；verify.md 的 C13 从 unverified 改成 confirmed。
- sup.md、fm.md 的负责人：按英文 §5 改 6 处。
- 写核心报告时，数字请从 `reports/tables/r440_core_numbers.csv` 里取（每条都有 source file）。

---

## English

### 1. What was done

#### 1.1 Runs and jobs

| Run | Job | What |
|---|---|---|
| `20261022-182022_r441_label_budget_rerun` | `J20261022-182157-pro1-r2aud` (5.2 min, 8 cores) | label-budget curve on new folds and new label draws; 11 models × 10 budgets × 80 folds × 5 draws = 40,480 evaluations |
| `20261022-184020_r440_core_numbers` | `J20261022-184159-pro1-r2aud` | final audit run (242 claims, r249 recheck, label-budget comparison, journey table) |
| `20261022-183328_r440_core_numbers` | `J20261022-183505-pro1-r2aud` | first audit run (240 claims). Superseded by the final run: same numbers; the final run adds claims D10–D11 and says "not submitted" instead of "pending" for the three candidates that have no sub_id |

Local runs: only smoke tests (< 10 s each) with `SSC_ARCHIVE_ROOT` pointing to the scratchpad. The session `pro1/r2aud` was stopped after each job.

#### 1.2 How the audit reads the archive

The Colab VM receives code and data, but not `experiments/` or `reports/`. `research/r440_core_numbers.py --pack` therefore copies, on the laptop, the files the audit needs into `research/r440_archive_pack.json.gz`. This is a pure file copy: no Run is created and nothing is written to the archive. The pack holds 206 files with the sha256 of each one's bytes, and the sha256 of the six submission CSVs. On Colab the script first checks every sha256, then recomputes:
- CV means and sds from `fold_scores.csv`. The sd is taken over the 20 repeat means (ddof 1), as in `ssc.validation`.
- paired differences with `ssc.validation.paired_compare`: mean difference, split SE, Nadeau–Bengio corrected p and fold win rate.
- everything else from the `summary.json` or run table that produced the number.

A claim is "verified: yes" when the archived value equals the claimed value at the claimed precision. For example, a claim of 0.9999 must lie within ±0.00005.

### 2. (a) Label budget: independent re-run of R206 (R441)

#### 2.1 Design

| | R206 (`20261022-074712`) | R441 (this round) |
|---|---|---|
| Folds | official repeats 0–3 (20 folds) | official repeats 4–19 (80 new folds) |
| Label draws | 3 per (fold, budget), rng 100000 + … | 5 per (fold, budget), `default_rng([4410, repeat, fold, n, draw])` |
| Budgets | 4–200 | 2–200 (2 = one label per class) |
| Code | `r206_label_budget.py` + `lib_sup` | candidate procedures from `r931_cand_lib` (bit-identical to the archived test predictions, r932); R206-equivalent models rewritten from sklearn |
| Scoring | full 50-row validation fold | same |

The "first budget" rule uses only budgets where every evaluation was defined. A mean over the few fits that succeed would be biased.

#### 2.2 Results

First budget with mean validation AUC ≥ 0.99 (`reports/tables/r440_label_budget_first_hit.csv`):

| Model | Uses test features | R206 | R441 | R441 mean AUC at that budget |
|---|---|---|---|---|
| Transductive GMM K=2 (S003) | yes | 4 | 4 (2 on R441's grid) | 1.0000 (400/400 evaluations at 2 labels) |
| Self-training QDA-LW, threshold 0.75 | yes | 20 | 20 | 0.9959 |
| Self-training QDA-LW, threshold 0.99 (S006 procedure) | yes | – | 20 | 0.9991 |
| QDA reg 0.003, train+test block | block only | 40 | 40 | 0.9925 |
| QDA reg 0.1, train+test block (S002 model) | block only | 120 | 120 | 0.9940 |
| QDA reg 0.003, in-fold block (strict) | no | 80 | 80 | 0.9939 |
| QDA, in-fold block, nested reg by inner AUC (S005 = r201) | no | – | 120 | 0.9993 (80 labels: mean 0.9842, median 0.9984) |
| QDA, in-fold block, nested reg by inner log-loss (r210) | no | – | 120 | 0.9966 (80 labels: 0.9376) |
| Bayesian QDA EB, train+test block (r255) | block only | – | 80 | 0.9980 |
| Bayesian QDA EB, in-fold block (strict) | no | – | 160 | 0.9951 |
| QDA-LW, in-fold block (strict) | no | never | never | 0.9870 at 200 |

Cell-by-cell comparison with R206 (`reports/tables/r440_label_budget_compare.csv`): of the 37 (model, budget) cells that are fully defined in both runs, the largest |z| is 1.98 (QDA-LW in-fold at 10 labels: 0.624 vs 0.656). No cell differs by more than 2 combined fold-cluster SEs. The pre-registered failure condition (> 3 SE, or a first-hit shift of more than one grid step) did not occur.

Comparison with R304 (E2; `reports/tables/r304_label_requirement.csv`). R304 scores on the other 250 − n rows, so only the direction is comparable:
- GMM: R304 has AUC 1.0 at 2 labels in 50/50 subsets. R441 has 1.0 at 2 labels in 400/400 draws.
- QDA reg 0.1: R304's median passes 0.99 at 75 labels, with mean 0.9899. R441's median passes at 80, its mean is 0.9897 at 80 and 0.9940 at 120.

Figure: `reports/figures/r440_label_budget_compare.png`. The left panel shows R441 (solid) against R206 (dashed). The right panel shows the candidate procedures with R304's medians.

#### 2.3 What this means

- D009 evidence 1 is now independently reproduced. verify.md C13 can move from "unverified" to "confirmed (R441)".
- The headline numbers stand: GMM about 4 labels (2 suffice), self-training about 20, supervised QDA with a label-free block about 40.
- "About 80 labels for strictly supervised QDA" holds only with reg fixed at 0.003, a value taken from R201's nested choices on all outer folds. The honest strictly supervised procedure we would submit (S005) needs about 120. At 80 labels its nested tuning sometimes picks a poor reg (10% quantile 0.975). At 40 labels it is undefined in 371 of 400 draws, because the in-fold block has more columns than a class has rows. Suggested wording: "Strictly supervised QDA needs about 80–120 labels (80 with a fixed, well-chosen regularisation; 120 when the regularisation is also tuned by nested CV, R441)."
- Log-loss tuning (r210) is fragile at small budgets (0.938 at 80 labels). This adds to verify.md's observation that r210 is fragile to mislabelled rows.

### 3. (b) Audit of the core-report numbers (R440)

#### 3.1 Coverage and outcome

`reports/tables/r440_core_numbers.csv` has 242 claims: 235 verified, 7 not. Every row names the file it was checked against.

| Section | Claims | Verified | Sources |
|---|---|---|---|
| data (rows, features, positive rate, Hanley–McNeil SEs) | 11 | 10 | data/raw, `ssc.validation.auc_standard_error` |
| generator (gen.md as corrected by verify) | 44 | 44 | R150, R151, R153, R154, R156, R920, R921, R922 |
| submissions S001–S006 (CV mean/sd from fold scores, sha256, public) + E3 | 31 | 31 | `registry/submissions.csv`, run folders, `submissions/*.csv` |
| supervised (sup.md, r249; incl. the r249 row recheck as one claim) | 56 | 55 | 30 sup runs, `r249_comparison.csv`, `r932_pairwise_agreement.csv`, `r930_paired_vs_gmm.csv` |
| validation protocol (E1 MDD, permutation nulls, S003 boundary rows) | 7 | 7 | `e1_paired_difference.csv`, `r900_perm_null_{A,B}_summary.csv` |
| fm (TabPFN v2, TabICL v2, horseshoe, BQDA, DP-GMM, GPC) | 62 | 58 | 15 fm runs, `r254_*`, `r255_eb_choice_frequency.csv` |
| replicas / final selection (r936) | 17 | 17 | `r936_{private,choice,cv_power,public_abc}.csv` |
| label budget (R206, R304, R441) | 14 | 13 | `r206_label_budget.csv`, `r304_label_requirement.csv`, R441 evals |

The r249 recheck (`reports/tables/r440_r249_recheck.csv`) recomputes every one of the 64 rows from `fold_scores.csv` and `summary.json`: CV mean and sd, OOF log-loss, and the paired difference, split SE and corrected p against r210. All 64 rows match. This covers every supervised ablation the report is likely to quote: LDA, GNB, linear models, poly-2 logistic, SVM, kNN, GPC, RF, ExtraTrees, LightGBM, XGBoost, label spreading and propagation, self-training, and the QDA variants.

#### 3.2 Claims that failed

| id | Claim (where) | Archive | Verdict and fix |
|---|---|---|---|
| P31 | "no method exceeded 0.88 on all 150 features" (sup.md, Text for the report; Chinese summary point 3) | sup-track maximum 0.875 (`r208_qda_all_lw`); TabICL v2 all features 0.9929 (`20261022-074251_r251_tabicl_all`), TabPFN v2 0.9851, horseshoe 0.9261 | refuted as a project-wide statement. Write "no model in the supervised track exceeded 0.88 on all 150 features; the pretrained TabICL v2 reached 0.993 without any feature selection" |
| F21 | "a regularised-horseshoe logistic regression … (0.926, the best linear model)" (fm.md, Text for the report) | best linear model 0.9383 (`20261022-074147_r207_l2_block_tt_wide`, block) | overstated. Write "the best linear model on all 150 features" |
| F13 | classical QDA reg 0.03 = 0.9990 (fm.md tables) | 0.998950 (`20261022-070545_r256_qda_reg0.03_hv`) | rounding: 0.9989 |
| F27 | BQDA on the same side of 0.5 as S003 for 99.5% of test rows (fm.md) | 0.99554 (`r255_bqda_eb_hv` summary, `agreement_with_s003.side_agree_all`) | rounding: 99.6% |
| L14 | "strictly supervised ~80 labels" applied to the S005 procedure | 120 (R441) | see §2.3 |
| D5 | public LB = 395 rows | 0.02 × 19,750; the share is on the Kaggle page only | not archivable. Archive the competition page (verify.md G2 already asks for the rules text) |
| F42 | BQDA has 1,493 test scores exactly 1.0 (fm.md) | not checked | needs `test_pred.npy`, which the pack does not carry. Low priority: the tie is harmless if no negative sits in it (fm.md) |

#### 3.3 Verified, but quote with a qualifier

| id | Number | Qualifier |
|---|---|---|
| G38, G39 | R156: flip_y posterior mean 0.0044, S003 expected private 0.9977 | correct for R156's 4-point prior, superseded by R922: private mean 0.9968, median 0.9973, 90% [0.9923, 0.9996] (G34–G36) |
| G19 | class_sep 0.71 (R150) | superseded by R920: 0.703, LR 95% [0.639, 0.760] (G17–G18) |
| G42 | S003 regret 3e-7 (R153, 100 replicas) | quote R921: 2.6e-7 (1000 replicas, G26) |
| G8 | dependencies "R² ≥ 1 − 4e-14" (gen.md) | archived 1 − R² = 4.07e-14, marginally above 4e-14. Write "1 − R² ≈ 4e-14" |
| P7 | GPC-ARD 0.976 (sup, `r202_gpc_ard_block`) vs 0.978 (fm, `r255_gpc_ard_hv`, train+test block) | sup.md's Chinese summary prints 0.978 and its English text 0.976. Both numbers are right for their runs; name the run |
| P36 | self-training on a tuned base: log-loss 0.0109 → 0.0120 | not a nested-honest comparison (reg 0.003 from R201); sup.md says so |
| F38–F40 | boundary-row AUC on replicas: GMM 0.825, BQDA 0.702; overall 0.9949 vs 0.9948 | 6 replicas only (verify.md C12); r936 (222 replicas) supports the direction |
| F7c | BQDA log-loss 0.0094 | fold-mean CV log-loss, not OOF; do not set it against sup's OOF 0.0071 (verify.md C7) |
| L5 | R206 table: strict QDA reg 0.003 at 40 labels = 0.9476 | mean of 6 of 60 evaluations; the other 54 fits are undefined. Do not quote it (the figures already omit it) |
| P21–P22 | r210 1.0000 ± 0.0001 | exact 0.99998 ± 0.00007; 1 fold of 100 below 1.0 |
| P49–P51 | D009 wording "95% interval about ±2–5e-4; 0 folds better, 1–4 worse" | r210: ±1.6e-4, r201: ±5.1e-4; worse in 1 and 4 folds. So "about ±2–5e-4" is right for r201 and generous for r210; write "±1.6–5.1e-4" |

#### 3.4 Numbers the core report can now use as they are

Most often needed, all verified (ids in the CSV):
- Data: 250 training rows, 113 positives (0.452), 19,750 test rows, 150 features, p/n = 0.6. Hanley–McNeil SE: 0.065 for one 50-row fold at AUC 0.80; 0.023 / 0.021 for a 395-row public AUC of 0.79 / 0.82; 0.0015 at 0.999.
- Generator: 12 informative columns (list in G1), found label-free with a 13.0-nat margin (R920); 8 redundant columns; 130 noise columns; one cluster per class; hypercube. class_sep is 0.70 (0.64–0.76). The 13-column block spans the informative space. Bayes AUC is 0.99995, the best linear rule 0.947, the Bayes error 0.33%.
- Journey: S001 0.8469 ± 0.0120 → public 0.83589; S002 0.9958 ± 0.0011 → 0.99277; S003 1.0000 → 0.99923; S004 (E3) 0.7934 ± 0.0142 → 0.79136. E3: leaky CV 0.867, gap 0.073 (SD over split seeds 0.014, corrected SE 0.020, p = 0.0003).
- Supervised: strict QDA (S005) 0.9999 ± 0.0002; r210 1.0000 ± 0.0001; 0.9995 ± 0.0006 after selection (r205); 587 configurations, 63 candidates. LDA 0.86–0.88, GNB 0.87, best linear 0.938, poly-2 0.930, SVM 0.951, kNN 0.902, GPC-ARD 0.976, trees ≤ 0.914 (LightGBM 0.9136, XGBoost 0.9033), label spreading 0.957.
- FM: TabPFN v2 0.9851 (all features) / 0.9888 (block); TabICL v2 0.9929 / 0.9989; horseshoe 0.9261 / 0.9673; Bayesian QDA 1.0000 (0.9990 strictly inductive).
- Replicas: private difference vs S003 for r209 −6.9e-5, r255 −1.16e-4, r210 −1.68e-4, r201 −2.57e-4. The second final adds at most 1.45e-5 (S003 + r209).

### 4. Journey table (figure-ready)

Generated by R440 (`reports/tables/r440_journey.md`, CSV with more columns in `r440_journey.csv`):

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

*Caption.* Official 5×20 CV (sd over repeats). Public LB from `registry/submissions.csv`. Surprise = (public − expected) / Hanley–McNeil SE of the expected AUC on 395 rows (198/197). Pending rows show the replica-ABC prediction given S003's public 0.99923 (`r936_public_abc.csv`, CSV-formatted scores). Last column = expected private-AUC difference to S003 on 222 parametric replicas (`r936_private.csv`, tn0, csv8, R156 weights); S003 itself is expected at about 0.997 (R922, flat prior).

Notes for the report:
- All four uploaded submissions landed within 1.4 Hanley–McNeil SEs of their pre-registered expectation. The largest surprise is S004 (−1.35 SE). Its pre-registered 0.82 was E3's proxy estimate (verify.md F8); the public score sits at the honest CV (0.793) and inside the pre-registered 90% range 0.785–0.854 (`registry/submissions.csv` notes).
- "Public − CV" is negative for all four, but each gap is smaller than one public SE (S001 −0.011 vs SE 0.020; S004 −0.002 vs 0.021). At this resolution it is not evidence of CV optimism.
- Rows 5–6 will receive their public scores when the main session uploads them. Re-run `research/r440_core_numbers.py` (pack, then the Colab step) afterwards to refresh this table.

### 5. Corrections list (file → exact correction)

I edited none of these files; owners or the main session apply them.

1. `research/reports/sup.md`, Text for the report, paragraph 1, last sentence: "and no method exceeded 0.88 on all 150 features" → "and no model in this track exceeded 0.88 on all 150 features (the pretrained TabICL v2 reaches 0.993 on all features without selection; fm)". Chinese summary "意味着什么" point 3: "在全部 150 个特征上，每种方法都不超过 0.88" → "在全部 150 个特征上，本线的模型都不超过 0.88（fm 的 TabICL v2 不做特征选择也有 0.993）". Source: R440 P31.
2. `research/reports/sup.md`, Text for the report, paragraph 2: "Supervised QDA needs about 40 labels, or about 80 when the feature block must also be found from the labelled rows" → "… or about 80–120 when the feature block must also be found from the labelled rows (80 with reg fixed at 0.003, a value taken from R201; 120 for the nested S005 procedure; R206, re-run in R441)". Chinese point 4 "严格监督约 **80** 个" → "严格监督约 **80–120** 个（reg 固定 0.003 时 80；S005 的嵌套流程 120；R441 重现）". Source: R440 L12, L14; R441.
3. `research/reports/sup.md`, Chinese point 2: "GPC-ARD 0.978" → "GPC-ARD 0.976" (the sup run `r202_gpc_ard_block`; 0.978 is fm's `r255_gpc_ard_hv`). Source: R440 P7.
4. `research/reports/fm.md`, Text for the report, paragraph 1: "(0.926, the best linear model)" → "(0.926, the best linear model on all 150 features; on the 13-column block L2 logistic reaches 0.938)"; Chinese "全特征版本的 0.9261 是目前最好的**线性**模型" → "…是全部 150 个特征上最好的线性模型（13 列块上 L2 logistic 是 0.938）". Source: R440 F21.
5. `research/reports/fm.md`, key-number table and report table: QDA reg 0.03 "0.9990" → "0.9989"; Text for the report: "99.5% of all test rows" → "99.6% of all test rows". Source: R440 F13, F27.
6. `research/reports/gen.md`, Table "n_redundant" row: "R² ≥ 1−4e-14" → "1 − R² ≈ 4e-14" (archived 4.07e-14). Source: R440 G8.
7. `research/reports/verify.md` §2 C13 (main or verify owner): verdict "unverified" → "confirmed by R441 (`20261022-182022_r441_label_budget_rerun`): same first budgets as R206 for all matched models, max |z| 1.98 over 37 cells; the nested S005 procedure needs 120, not 80". §7.1 rows "R206 label-budget curve" and "Supervised ablations" → "done (R441, R440)".
8. `registry/decisions.md` D009 (main session, by a superseding decision or a Status note as the rules allow): evidence (1) can cite R206 + R441 as reproduced, with the "80–120" wording of item 2. The CV-resolution wording proposed in verify.md §4 item 2 can say "±1.6–5.1e-4" instead of "±2–5e-4" (R440 P49–P50).
9. `reports/tables/r206_label_budget.md` cannot be edited (script output). When citing it, do not use the 40-label value of `sup_qda_infold_r003` (0.9476 from 6/60 evaluations). Source: R440 L5.
10. `research/r440_archive_pack.json.gz` (658 kB) travels in every Colab code bundle. Delete it once the core report is frozen; `--pack` rebuilds it from the archive. I did not delete it.

### 6. Open points

- S005/S006 public scores: after the upload, refresh the journey table (pack + Colab run). The table code reads `registry/submissions.csv` and adds the scores and surprises.
- F42 (BQDA ties at 1.0) remains unchecked. It only matters if BQDA is ever submitted.
- The public-LB size (395) and the 2% share should be archived from the Kaggle page together with the rules text (verify.md G2).
- R441 used the official splits (repeats 4–19). A label-budget curve on parametric replicas (`lib_gen_replica`) would show whether 80–120 labels is a property of these 250 rows or of the generator. Not needed for the report.
