# verify_leak：标签泄漏审查（R900–R901）

- 作者：AI 代理（Claude Code），verifier label `leak`；计算：Colab 账号 pro2、会话 `vleak`（High-RAM CPU，8 核），2026-10-22 JST，作业 `J20261022-091532/093125/094954-pro2-vleak`（`J20261022-091214` 是一次失败的启动，见下文）。会话已 `down`。
- 代码：`research/r900_perm_null.py`（标签置换零分布，Null A / Null B）、`research/r901_id_order_dupes.py`（id、行序、重复行、文件格式探针）。
- Run（都是 `kind = analysis`）：`20261022-091506_r900_perm_null_a`、`20261022-092943_r900_perm_null_b`、`20261022-094812_r901_id_order_dupes`；失败的 `20261022-091033_r900_perm_null_a`（pickling 错误，已在 notes.md 说明，不用）。
- 表/图：`reports/tables/r900_perm_null_{A,B}_{summary,per_perm}.csv`、`reports/figures/r900_perm_null_{A,B}.png`、`reports/tables/r901_{duplicates_exact,near_duplicates}.csv`、`reports/tables/r901_id_order_probes.json`。

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要第 4 条和英文 §4.4）。依据：`research/reports/verify.md` §1.2、§3 C5。

1. **E3 gap uncertainty**（摘要"发现的问题"第 4 条；§4.4）。Old: "不是均值的标准误（SE = 0.014/√20 ≈ 0.0032）" / "not the SE of the mean (0.014/√20 ≈ 0.0032)", which read as the SE to report → New: adds that 0.0032 is split noise conditional on the 250 rows, and that the inferential (Nadeau–Bengio corrected) SE is 0.020 (t 3.72, p 0.0003). Source: `experiments/20261022-063908_e3_leakage/cells.csv`, row k_feat 20, k_fold 5: gap_mean 0.0733, gap_std 0.0143, corrected_t 3.724, corrected_p 0.00033, so SE = 0.0733 / 3.724 = 0.0197. Report the spread as "SD over 20 split seeds 0.014" and the inference as "corrected SE 0.020, p = 0.0003".

---

## 中文摘要（给队友）

**一句话：我尽力想证明我们的最好方案有标签泄漏，但没有找到。** S003（直推式 GMM）、r155、r201/r210（嵌套 QDA）、r209（自训练）、r255（贝叶斯 QDA）和 E3 的"诚实"流程，在代码路径、置换检验、id/行序/重复行检查这三个方面都通过了。找到的问题都不是"验证折的标签进了模型"，而是研究者层面的选择偏差和表述问题，影响在 1e-4 量级或者更小。

### 做了什么

1. **逐行读代码**（引用见英文部分第 1 节）。所有流程都经过 `cross_validate`，它只把训练折的 `y[train_idx]` 交给 `fit`（`src/ssc/validation.py:137`）。所以验证标签要想进模型，只有两条路：(a) 在 CV 外面用全部 250 个标签算了什么，再"烤"进模型里（比如在全部行上选特征）；(b) 模型从别的地方拿到真标签（读 `train.csv` 的 target、全局缓存等）。逐个流程查了这两条路，都没有。另外核对了代码包：存档 run 用的代码和现在的代码在这些路径上一致。
2. **两种标签置换检验**，都用官方 CV（5 折 × 20，seed 2026），每种 20 次置换：
   - **Null A（经典置换）**：用打乱的标签训练、也用打乱的标签评分。能抓住路 (a)。
   - **Null B（交叉置换）**：用打乱的标签训练，用**真标签**评分。能抓住路 (b)，这是 Null A 看不到的：如果模型偷偷读真标签，Null A 的评分标签本身是乱的，照样显示 0.5。
   - 两个**阳性对照**：`ctrl_e3_leaky`（E3 的泄漏流程：在全部 250 行上选 top-20 |corr| 特征）和 `ctrl_disk_reader`（故意写坏的模型：不管传进来的标签，直接读 `train.csv` 的真标签在全部 250 行上拟合 QDA）。外加一个随机打分的基准。
3. **id / 行序 / 重复行 / 文件格式探针**（R901）。

### 关键数字

| 流程 | 真标签 CV AUC（重算 = 存档） | Null A：20 次置换的均值 ± sd，最大值，>0.6 的比例 | Null B：均值，AUC>0.5 的置换数 / 20 |
|---|---|---|---|
| S003 `s003_gmm2_hv` | 1.0000 = 1.0000 | 0.496 ± 0.041，max 0.559，0 | 0.573，11/20（双峰：标签只决定分量方向） |
| r155 `gmm2_inf_tight` | 1.0000 = 1.0000 | 0.495 ± 0.041，max 0.556，0 | 0.573，11/20 |
| r201 `qda_block` | 0.99990 = 0.99990 | 0.481 ± 0.028，max 0.541，0 | 0.518，10/20 |
| r210 `qda_block_nll` | 0.99998 = 0.99998 | 0.477 ± 0.041，max 0.561，0 | 0.516，11/20 |
| r209 `selftrain_qda_lw` | 1.0000 = 1.0000 | 0.491 ± 0.036，max 0.557，0 | 0.480，8/20（双峰） |
| r255 `bqda_eb_hv` | 1.0000 = 1.0000 | 0.479 ± 0.035，max 0.568，0 | 0.538，10/20 |
| E3 诚实流程 | 0.7934 = 0.7934 | 0.485 ± 0.040，max 0.545，0 | 0.504，10/20 |
| **阳性对照** E3 泄漏流程 | 0.8668 = 0.8668 | **0.697 ± 0.023，min 0.641，20/20 > 0.6** | 0.512，12/20 |
| **阳性对照** 读盘作弊 | 1.0000 | 0.506 ± 0.032，0（Null A 看不出来） | **1.000，20/20** |
| 随机打分 | 0.508 | 0.500 ± 0.008，0 | 0.508（构造上是常数） |

怎么读：诚实的流程在 Null A 下都在 0.5 附近，没有一次超过 0.6；泄漏对照稳定在 0.70，读盘对照在 Null B 下每次都是 1.000。所以两种检验都有检出能力，而我们的流程两种都没触发。诚实流程在 Null A 下略低于 0.5（0.477–0.496）是分层 CV 在零假设下已知的悲观偏差，不是泄漏（泄漏只会往上推）。

R901 探针：
- train 的 id 是 0–249，test 是 250–19999（train 就是 id 序列的前 250 行），但 AUC(id, y) = 0.552（p 0.16），按 id 五等分的正例比例 0.40–0.52（p 0.70），游程检验 p 0.60。S003 的 test 预测和 test id 无关（Spearman −0.0013，十等分 p 0.17）。id 和 150 个特征的最大 |Spearman| 是 0.016，低于 4 SE（0.028）。
- **没有任何完全重复的行**（150 列、13 列块、12 个信息列，train 内部、test 内部、train–test 之间都是 0）。最近邻距离：train→test 和 test→test 的分布一样（比 test→test 1% 分位更近的 train 行：2 / 3 / 1 个，期望 2.5）。
- 对抗验证（区分 train 和 test 的分类器）AUC 0.499 ± 0.025：train 是随机抽样。CSV 的小数位数和类别无关（AUC 0.456，p 0.23）。
- 高方差块**只用 test 的特征**也能得到完全相同的 13 列；块里有 3 列和标签几乎不相关（|corr| 排名 85、44、33 / 150）。12 个"信息列"里有第 61 列（|corr| 0.006，排名 139），只靠标签的选法不可能选到它。所以这两个列集合确实是不看标签找到的。
- 重算的 S003 test 预测与存档 `test_pred.npy` 的差 < 6e-11，提交 CSV 与它的差 < 5e-9，id 顺序与 test.csv 一致。

### 发现的问题（都不是折内泄漏，影响都很小）

1. **研究者层面的选择**：R001 画过"方差 vs 与标签的 |corr|"散点图（`research/r001_data_structure.py:51,82`），R002 用标签看过分量纯度。所以"用高方差块 + K=2 GMM"这个设计决定，是在看过全部 250 个标签之后做的。代码本身不看标签，但设计环节看过。影响很小：只用 test 特征就能得到同一个块，BIC 也选 K=2（R101），public LB 0.99923 用的是从来没见过的标签。报告里应该如实写一句。需要注意的是阈值 1.5 离噪声列的最大比值（1.473）很近，块里最小的比值是 1.675：在 20,000 行上结果是确定的，但在 200–250 行上会多选 3–10 列噪声（这也是 r201 折内块有 16–23 列的原因）。
2. **r209 的 `sup_qda_tt_r003` 和 `selftrain_qda_r003`**（CV 1.0）把 reg 固定成 0.003，而 0.003 是 R201 在全部外层折上的众数选择（`r209_semisup_followup.py:46,49` 自己写了 "informed by R201"）。这是一个在 CV 外、用全部标签选出的超参数，不能当作嵌套诚实的估计引用。它们不是我们的候选方案，但别把它们写成"诚实 CV 1.0"。
3. **"最好的监督模型 r210 CV 0.99998"**：r210 是看过 R209 结果之后才写的（换了内层指标），sup 线一共试了 587 个配置。选择感知的估计是 r205（46 个配置的整个菜单都嵌套）0.99955 ± 0.0006。报告里监督模型的水平写成 0.9995–0.9999 更稳妥。D009 说"不可区分"，这个结论没问题。
4. **E3 的"0.073 ± 0.014"**：均值对（`cells.csv` 里 k_feat 20 / 5 折是 0.0733），但 ± 0.014 是单次重复的 gap 在 20 个 split seed 之间的 SD，不是均值的标准误（SE = 0.014/√20 ≈ 0.0032）。但 0.0032 只是"给定这 250 行"时换 split 的噪声；用于推断的 Nadeau–Bengio 校正 SE 是 0.020（t 3.72，p 0.0003；`cells.csv`）。报告里要写清楚 ± 指的是什么。用官方 5×20 的置换检验，泄漏流程比诚实流程高 0.21（0.697 vs 0.485），和 E3 自己的零分布（0.705 vs 0.499）一致。
5. **可复现性**：r255 和 E3 是用 `--no-sync` 跑的，`meta.json` 里没有代码包。我用 `code_sha`、当时的代码包和 R900 的逐位重算确认了代码没变。但 E3 的最终模型还等着提交，以后可能提交的 run 最好不要用 `--no-sync`。

### 建议

- S003 / r155 / r201 / r210 / r209 / r255 / E3 诚实流程：**从泄漏角度没有反对理由**，可以照常作为候选。
- 报告的 Validation protocol 一节可以加一段"Leakage audit"，引用 R900 的两种置换检验和两个阳性对照（这正好也能给 E3 当证据）。同时写上第 1、3、4 条。
- 以后的研究 run 不要再用"在全部外层折上选出来的固定超参数"，否则就用嵌套。

---

## Details (English)

### 0. Scope and method

Task: try to show that any of our top pipelines leaks label information, i.e. that the label of a validation row can influence its own prediction, directly or through anything (selection, block, prior, basis) computed with labels from outside the training fold. Three lines of evidence: (1) a static code-path audit with file:line references, (2) two label-permutation nulls under the official CV with positive controls, and (3) id / order / duplicate / file-format probes. Everything was computed on Colab (pro2/vleak, 8-core High-RAM). The local machine was used only for smoke tests into a scratch archive and for reading archived files.

### 1. Code-path audit

**Common CV path.** `ssc.validation.cross_validate` (src/ssc/validation.py:143-184) calls `_fit_one` (validation.py:134-140). `_fit_one` passes only `X[train_idx], y[train_idx]` to `model.fit` (validation.py:137) and scores `X[val_idx]` (validation.py:138). Validation labels are used only by `roc_auc_score` (validation.py:170). `test_pred="refit"` (validation.py:180-182) is the only fit on all 250 rows, and it is used for test predictions only. A validation label can therefore reach its own prediction only through (a) a quantity computed from the full label vector outside `cross_validate` and frozen into the estimator, or (b) labels obtained through another channel (disk, a global cache, a hard-coded column list chosen with labels). Every pipeline below was checked for both.

| Pipeline (archived run) | Where labels enter | Label-free parts | (a) | (b) | Verdict |
|---|---|---|---|---|---|
| S003 = R003 `gmm2_hv` (`20261022-050708_r003_gmm2_hv`), `TransductiveGMM("highvar", 2)` | `fit`: component class-1 rates from the training-fold labels with Beta smoothing, `p0 = y.mean()` of the fold (src/ssc/transductive.py:103-112, lines 109-110) | `unlabeled_features` reads only the FEATURE columns of train+test (transductive.py:26-35, line 35, no `target`); `highvar` = std over the 20,000 rows (transductive.py:42-45); standardisation/PCA on the 20,000 rows (`_basis`, transductive.py:49-61); mixture fitted on those features only and cached (`_mixture`, transductive.py:64-69) | none | none: no label is read or cached anywhere in transductive.py | clean |
| r155 `gmm2_inf_tight` (`20261022-080453_r155_gmm2_inf_tight`), `TransductiveGMMTight(columns="informative")` | `fit`: same component-rate rule (research/lib_gen_replica.py:606-614) | informative columns from `identify_structure(X_all)` (lib_gen_replica.py:136-281): its only input is the feature matrix; the GMM posterior used for the hypercube z-test is label-free (lib_gen_replica.py:176-178); called from `UnsupervisedGMM.fit` (lib_gen_replica.py:536-540). `ref.scores(..., ds.y)` in r155 (research/r155_candidate_gmm_tight.py:47) uses all 250 labels, but only for the S003-agreement diagnostics, not for the CV | none | none | clean; the 12 columns include col 61 (|corr| 0.006, rank 139 of 150), which no label-based rule would pick (R901) |
| r201 `qda_block` (`20261022-071234_r201_qda_block`) | `Tuned.fit` runs `GridSearchCV(cv=RepeatedStratifiedKFold(5, 2))` on the X, y it receives (research/lib_sup.py:234-263, lines 242-246), i.e. on the 200-row outer training fold from validation.py:137 | `HighVarColumns` (std on the fitting rows, lib_sup.py:92-120) and `StandardScaler` are pipeline steps re-fitted in every inner and outer fit (research/r201_sup_quadratic.py:50-51, 77) | none | none | clean. Archive check: `fold_fit_info.csv` has 100 rows; each is keyed by the SHA1 of the training matrix that `Tuned.fit` received, and all 100 match the 100 official outer training folds one-to-one (0 unmatched, 0 duplicates). In-fold supports have 16-23 columns, so the block really is re-derived from 200 rows. So the inner CV is inside the outer training fold. |
| r210 `qda_block_nll` (`20261022-082047_r210_qda_block_nll`) | as r201, inner `scoring="neg_log_loss"` (research/r210_sup_tiebreak_lw.py:44-46) | as r201 | none | none | clean (same 100/100 fold mapping) |
| r209 `selftrain_qda_lw` (`20261022-080632_r209_selftrain_qda_lw`) | `PoolSelfTraining.fit`: `y_pool = -1` for all 20,000 rows, then the training-fold rows get their labels (lib_sup.py:427-428). Validation and test rows stay unlabelled (research/r209_semisup_followup.py:56-57) | block and standardisation from train+test features (`pool_space`, lib_sup.py:316-323); rows are located by exact bytes (lib_sup.py:303-313). R901: no duplicate rows, so the mapping is one-to-one | none | none | clean |
| r255 `bqda_eb_hv` (`20261022-070200_r255_bqda_eb_hv`) | `BayesQDA.fit` (research/lib_fm.py:325-364): the NIW prior hyperparameters (kappa0, nu0, S0) maximise the closed-form marginal likelihood of the FITTING rows (training fold) over a fixed grid (lib_fm.py:337-349). S0 is the pooled within-class covariance of the fitting rows with their labels (lib_fm.py:314-323, 341), m0 is the fitting-row mean (line 335), and the SVD projection uses the fitting rows (lines 329-333) | block + train+test standardisation via `SubspaceModel` (research/r255_bayes_qda.py:63) | none | none | clean. The "empirical Bayes" prior comes from the training fold only. The all-250-row EB fit (r255_bayes_qda.py:154) feeds only the descriptive table of `r255_bqda_eb_analysis`. |
| E3 honest / final model (`20261022-063908_e3_leakage`) | `CorrelationTopK.fit` on the fitting rows (src/ssc/features.py:56-70) inside `Pipeline(scaler, CorrelationTopK, model)` (extensions/e3_leakage/run.py:140); the leaky variant freezes columns from all rows (run.py:131-139) | scaler fitted in-fold | none (honest), yes by design (leaky) | none | honest pipeline clean; selection is inside the folds |

Other paths that read labels outside a fold, and why none of them matters here: `lib_mix.ReplicaGenerator` loads all 250 labels to orient replica components (research/lib_mix.py:476-481) and `lib_gen_replica.fit_parametric` uses them for orientation only (lib_gen_replica.py:312-316). Both are simulation code, not CV of a candidate. `r100` uses labels only in its marked descriptive sections.

**Code identity.** On the audited paths, the code that produced each archived run is the current code. R003 bundle `b8534fcd0d68fc89` (D005) is byte-identical for `src/ssc/{transductive,validation,models,data}.py` and `research/r003_*.py`. Bundles `f84e098deaf7a1bf` (r201), `126a725bb7ba4972` (r209), `f4064b1c399a0de8` (r210) and `5577ac7d589756ab` (r155) differ from the current tree only in docstrings and in the later `Tuned(scoring=...)` argument, whose default `"roc_auc"` reproduces the old behaviour. r255 and E3 were launched with `--no-sync` (`meta.json compute.code_bundle = "unsynced (--no-sync)"`). For r255 the recorded `code_sha` (lib_fm.py `9a8fb86c4d4e37e6`) matches bundle `05afe965914950b0`, whose BayesQDA differs only by the later `predictive` option (default `"t"` = old behaviour). For E3, every bundle created 06:35-06:39 (just before the 06:40 job) differs from the current `run.py` only in a figure title; `features.py`, `validation.py` and `models.py` are identical. The strongest identity check is that R900's true-label rows reproduce every archived CV AUC exactly (difference 0, Section 2).

### 2. Label-permutation nulls (R900, official CV)

Design (research/r900_perm_null.py). Permutation p uses `y_p = default_rng(9000 + p).permutation(y)`, p = 0..19. Pipelines are exact reconstructions of the archived runs, with `test_pred="none"`.
- **Null A**: fit and score on `y_p`, splits `make_splits(y_p, official)`. It detects path (a). Positive control `ctrl_e3_leaky` = E3's leaky procedure (top-20 |corr| columns chosen on all 250 rows of `y_p`, then scaler + FixedColumns + L2 logreg C=0.1).
- **Null B**: fit on `y_p`, score against the TRUE labels, official splits of the true labels (signature `5708ce5c3828`). It detects path (b), which Null A cannot: there the evaluation labels are themselves permuted, so a model that secretly uses the true labels still scores about 0.5. Positive control `ctrl_disk_reader` ignores the labels it is given and fits QDA(reg 0.01) on the block of all 250 rows with the true labels read from `train.csv`.
- `ctrl_random`: random scores, the chance floor. Its seed depends on the training rows, so in Null B (fixed splits) it is the same 0.508 in every permutation.

Null A (`20261022-091506_r900_perm_null_a`, `reports/tables/r900_perm_null_A_summary.csv`). Per-permutation mean AUC over the 100 folds:

| pipeline | real (= archived) | mean | sd | min | max | frac > 0.6 | #perm > 0.5 |
|---|---|---|---|---|---|---|---|
| s003_gmm2_hv | 1.0000 (1.0000) | 0.4960 | 0.0408 | 0.4489 | 0.5588 | 0 | 9 |
| r155_gmm2_inf_tight | 1.0000 (1.0000) | 0.4950 | 0.0405 | 0.4457 | 0.5558 | 0 | 9 |
| r201_qda_block | 0.99990 (0.99990) | 0.4811 | 0.0277 | 0.4366 | 0.5412 | 0 | 4 |
| r210_qda_block_nll | 0.99998 (0.99998) | 0.4771 | 0.0407 | 0.3965 | 0.5607 | 0 | 6 |
| r209_selftrain_qda_lw | 1.0000 (1.0000) | 0.4914 | 0.0355 | 0.4436 | 0.5575 | 0 | 10 |
| r255_bqda_eb_hv | 1.0000 (1.0000) | 0.4794 | 0.0350 | 0.4160 | 0.5685 | 0 | 4 |
| e3_honest | 0.7934 (0.7934) | 0.4850 | 0.0398 | 0.4066 | 0.5446 | 0 | 7 |
| **ctrl_e3_leaky** | 0.8668 (0.8668) | **0.6970** | 0.0235 | **0.6408** | 0.7360 | **1.0** | 20 |
| ctrl_disk_reader | 1.0000 | 0.5059 | 0.0315 | 0.4432 | 0.5567 | 0 | 11 |
| ctrl_random | 0.5081 | 0.5002 | 0.0079 | 0.4838 | 0.5140 | 0 | 10 |

- Every honest pipeline stays at chance. No permutation mean exceeds 0.6 (overall max 0.569). The spread (sd 0.028-0.041) matches the known-honest reference e3_honest (0.040). It is larger than the random floor (0.008) because all 100 folds of one permutation share the same chance association between features and permuted labels. The permutation p-value of every real AUC is 1/21 = 0.048, the minimum possible with 20 permutations.
- The honest pipelines sit slightly BELOW 0.5 (0.477-0.496; r201 and r255 have 4/20 permutations above 0.5, sign-test p 0.012). This is the known pessimistic bias of (stratified) CV under the null (e.g. Parker, Guenter & Bedo 2007, BMC Bioinformatics 8:326): rows that the training fold over-represents by chance are under-represented in the validation fold. Leakage pushes the other way.
- The positive control has power. `ctrl_e3_leaky` is 0.697 ± 0.023 (20/20 above 0.6), consistent with E3's own null at k_feat 20 / 5-fold (leaky 0.705, honest 0.499, 30 permutations x 5 seeds). Leakage of the E3 kind would have been seen.
- `ctrl_disk_reader` is 0.506: Null A alone is blind to label channels outside the CV, so Null B is needed.

Null B (`20261022-092943_r900_perm_null_b`, `reports/tables/r900_perm_null_B_summary.csv`):

| pipeline | mean | sd | min | max | frac > 0.6 | frac < 0.4 | #perm > 0.5 (sign-test p) |
|---|---|---|---|---|---|---|---|
| s003_gmm2_hv | 0.5725 | 0.3685 | 0.040 | 1.000 | 0.50 | 0.35 | 11 (0.82) |
| r155_gmm2_inf_tight | 0.5725 | 0.3685 | 0.040 | 1.000 | 0.50 | 0.35 | 11 (0.82) |
| r201_qda_block | 0.5184 | 0.1138 | 0.324 | 0.744 | 0.25 | 0.15 | 10 (1.00) |
| r210_qda_block_nll | 0.5157 | 0.1311 | 0.285 | 0.782 | 0.30 | 0.20 | 11 (0.82) |
| r209_selftrain_qda_lw | 0.4795 | 0.3892 | 0.000 | 0.995 | 0.40 | 0.45 | 8 (0.50) |
| r255_bqda_eb_hv | 0.5383 | 0.1283 | 0.338 | 0.837 | 0.30 | 0.15 | 10 (1.00) |
| e3_honest | 0.5040 | 0.0404 | 0.421 | 0.571 | 0 | 0 | 10 (1.00) |
| ctrl_e3_leaky | 0.5124 | 0.0551 | 0.411 | 0.610 | 0.05 | 0 | 12 (0.50) |
| **ctrl_disk_reader** | **1.0000** | 0 | **1.000** | 1.000 | **1.0** | 0 | **20 (2e-6)** |
| ctrl_random | 0.5081 | 0 | 0.508 | 0.508 | 0 | 0 | (constant by construction) |

- Trained on permuted labels, the honest pipelines show no preference for the true orientation: the number of permutations above 0.5 is consistent with Binomial(20, 1/2) for all of them. The GMM pipelines and self-training are bimodal (near 0 or near 1). Their label-free structure separates the true classes, and the permuted training labels only pick the sign; a pipeline that took its orientation from the true labels would sit at 1.0 in 20/20. The QDA-type pipelines spread over 0.29-0.84, because a QDA fitted to random labels picks up the covariance structure in a random direction.
- `ctrl_disk_reader` = 1.000 in 20/20 permutations, so a channel that brings the true labels in from outside the training fold would have been detected.
- Fraction above 0.6 is not a meaningful leakage statistic in Null B, because honest bimodal pipelines put half their mass near 1. The symmetric statistics are the count above 0.5 and frac > 0.6 vs frac < 0.4.

### 3. Ordering, id, duplicates, file format (R901, `20261022-094812_r901_id_order_dupes`)

- **ids**: train ids 0..249 and test ids 250..19999, both sorted; together they are exactly 0..19999. The train set is the first 250 rows of the id sequence. If the generator had not shuffled its rows, the classes would be ordered and this would leak, but nothing in the data shows any order. AUC(id, y) = 0.552 (Mann-Whitney p 0.16), AUC(id mod 2) 0.512, AUC(id mod 10) 0.548 (p 0.19); positive rate per id quintile 0.40/0.40/0.46/0.48/0.52 (chi-square p 0.70); runs test z 0.53 (p 0.60); lag-1 autocorrelation -0.039 (permutation p 0.52).
- **S003 vs test id / order**: Spearman(test id, S003 score) = -0.0013; predicted-positive rate per id decile 0.481-0.522 (chi-square p 0.17); runs test of the predicted labels along the file order p 0.27; lag-1 p 0.26. Id vs every feature on all 20,000 rows: max |Spearman| 0.016 (col 26), 0 of 150 above 4 SE (0.028); id vs mixture posterior -0.0006.
- **Duplicates**: 0 exact duplicate rows within train, within test and between train and test, in all 150 columns, the 13-column block and the 12 informative columns (`r901_duplicates_exact.csv`). Near duplicates, with train+test standardisation (`r901_near_duplicates.csv`): the train-to-test NN distance matches the test-to-test reference (3,000 random test rows). Train rows closer than the reference's 1% quantile: 2 / 3 / 1 (expected 2.5). Train rows closer than the reference minimum: 0 / 0 / 0. KS p 0.064 / 0.91 / 0.99. Smallest train-to-test distance in all 150 columns: 12.03 (reference median 13.60, minimum 11.42).
- **Random subset**: adversarial validation (L2 logreg, train vs test, 5x3 CV) AUC 0.499 ± 0.025 (all 150), 0.511 (block), 0.511 (informative); KS test of the mixture posterior train vs test p 0.24; S003 predicted test positive rate 0.501 vs train 0.452 (binomial p 0.13; the mixture weight is 0.4998, consistent with a balanced generator).
- **File format**: decimals per value 1-6 in both files (row mean 5.888 train / 5.889 test); AUC of row-mean decimals vs y 0.456 (p 0.23), of string length 0.465 (p 0.34); max per-column |AUC - 0.5| of the decimal count 0.051 (col 39, over 150 columns); min number of distinct values per column 19,889, so no column is categorical.
- **Block label-freeness**: the std/median-std rule reproduces exactly the 13 columns from all 20,000 rows and from the 19,750 test rows alone; 250 train rows alone give 19 columns, a superset. On 20,000 rows the 13th ratio is 1.675 and the 14th 1.473, so the 1.5 threshold sits in a narrow label-free gap, 0.027 above the largest noise ratio. Ranks of the block columns by |corr| on the 250 labels: 98→1, 100→3, 112→4, 80→5, 77→6, 46→7, 49→8, 67→9, 8→10, 26→17, 64→33, 30→44, 15→85; 9 of 13 overlap the top-13 |corr|. The 12 informative columns overlap the top-12 |corr| in only 4 (61→139, 142→103, 52→98).
- Recomputed S003 test predictions equal the archived `test_pred.npy` to 6.2e-11. The S003 submission CSV equals it to 5e-9, with ids in `test.csv` order.

### 4. Researcher-level (meta) issues: not fold leakage, but worth stating

1. **Label-informed design of the label-free pipeline.** R001 plotted std against |corr| with the training target (research/r001_data_structure.py:51, 82) and R002 checked component label purity, before the "high-variance block + K=2 GMM" design was fixed. The code never sees labels; the design loop did. The effect is bounded: the same block comes from test-only features, BIC picks K=2 without labels (R101), 3 of the 13 columns are poorly label-correlated (ranks 85/44/33), and the public LB (0.99923) is scored on labels nobody has seen. Severity low; one sentence in the report. D007's "unsupervised steps never see labels" is true for the code.
2. **Fixed hyperparameter selected on all labels.** r209 `sup_qda_tt_r003` and `selftrain_qda_r003` fix reg = 0.003 because it was the modal nested choice of R201 over the 100 outer folds (research/r209_semisup_followup.py:46, 49). That value was chosen with every label, then evaluated on the same folds, so their CV 1.0 is not a nested-honest estimate. They are not candidates; just do not cite them as honest. `selftrain_qda_lw` (tuning-free Ledoit-Wolf, threshold 0.99, which was already in R204's a-priori grid) is fine.
3. **Winner's curse in the supervised track.** r210 was written after reading R209 (research/r210_sup_tiebreak_lw.py:1-13), and the track evaluated 587 configurations. The selection-aware estimate is r205 (`20261022-072747_r205_select_supervised`, the whole 46-configuration menu nested), 0.99955 ± 0.0006, against r210's 0.99998. Quote the supervised level as about 0.9995-0.9999, not 0.99998. D009's "indistinguishable" stands.
4. **E3 uncertainty label.** "0.073 ± 0.014": the mean is right (`cells.csv`, k_feat 20, 5-fold: 0.0733). But 0.014 is the SD of a single repeat's gap across 20 split seeds (`gap_std`), not the SE of the mean (0.014/√20 ≈ 0.0032). That 0.0032 is split noise conditional on the 250 rows; the inferential (Nadeau–Bengio corrected) SE is 0.020 (t 3.72, p 0.0003; `cells.csv`). The report should say which. Under the official-CV null the leaky-minus-honest gap is 0.21 (0.697 - 0.485), consistent with E3's own null.
5. **`--no-sync` runs** (r255, E3). The code identity was recovered (Section 1). E3's final model still awaits a submission, and runs that may be submitted should keep a code bundle.

### 5. What I could not check

- The public/private split of the test set and the hidden test labels (Kaggle-side) cannot be probed; the public score is treated as given.
- Permutation tests only see leakage that the pipeline code can express. A human looking at the public LB and then changing a model is outside their scope; the submission log (3 submissions, D008) makes it negligible so far.
- R900 does not include the r102/r205 pipelines. By reading their code (research/lib_mix.py:296-317 `SemiSupEM` uses only `rows_in_all(X)` of the training fold, lib_mix.py:671-685 `SupSelectGMM` selects on the fold's rows), they follow the same pattern, but they were not permutation-tested.

### 6. Reproduce

```powershell
.venv\Scripts\python.exe colab\remote.py run -a pro2 -s vleak --high-mem -- research/r900_perm_null.py --null A
.venv\Scripts\python.exe colab\remote.py run -a pro2 -s vleak --high-mem -- research/r900_perm_null.py --null B --skip-real
.venv\Scripts\python.exe colab\remote.py run -a pro2 -s vleak --high-mem -- research/r901_id_order_dupes.py
```
Job wall time on 8 cores (registry/compute_jobs.csv): Null A 938 s, Null B 943 s, R901 36 s.
