# ext12 报告：扩展 E1（验证方案设计）与 E2（学习曲线），真数据完整运行

作者：Claude Code（agent ext12），2026-10-22。计算全部在 Colab（账号 pro1，会话 rext12，CPU High-RAM 8 核），会话已 `down`。
扩展报告草稿：`reports/extensions/e1_validation_protocol.md`、`reports/extensions/e2_learning_curve.md`（作者行仍是 `Author: TODO`，等队员认领）。

## Errata (verify round 1, 2026-10-22)

第 1 轮核查（`research/reports/verify_ext.md`，复核 run R940 `20261022-091211_r940_e1_pairs_check`、R941 `20261022-092201_r941_e2_replica_truth`、R942 `20261022-094142_r942_e2_coverage`；汇总见 `research/reports/verify.md` §4）发现下文有几处数字和结论要更正。下文正文保持原样，只在原处改了第 9 条（:49 那一行）；fix round 2 又在原处重写了英文 "Text for the report"（第 12 条）。格式是 旧 → 新。行号（如 :49）是加入本节之前的原行号，现在都顺延 26 行（:49 → :75；fix round 2 加了第 12 条之后）。两份扩展草稿已经按新值改写。

**E1**
1. :34 "split 噪声小 4.5–6 倍" → "小 4.5–6.1 倍"（比值 4.53 / 5.81 / 5.84 / 6.09 / 5.76，来自 `e1_paired_difference.csv` 与 `e1_qda_paired_difference.csv`）。
2. :35 "reference 5x50 为 −0.0029（SE 0.0014）" → "SE 约 0.0019"。0.0014 低估了 reference 的 split 噪声（verify_ext §3 第 5 条）。:35 的"真实 CV 差距"和 :71 的 "the reference 5 x 50 gap of the linear pair is only −0.003" → 50 次执行共 1000 个 repeat 合并的差距 **−0.0044 ± 0.0004**（R940），即这一对只差约 0.004。
3. :35 "官方种子上的 −0.0101 偏离约 2 sd" → "偏离 1.84 sd"，即 (−0.0101 + 0.0044) / 0.0031。
4. :35 "NB 校正 t 检验在 50 次中从未判这一对显著" → "3-fold、5-fold 各有 1/50 次显著（frac 0.02）；10-fold、5x10、5x20 为 0 次"（hold-out 不做这个检验）。所以应写成"最多 50 次中 1 次"。
5. :29、:71 "只有官方 5x20 的 MDD 整个 CI 都低于 0.01" → 必须写明判据和功效：MDD = 2 sd，约等于 50% 功效（α = 0.05）。按 80% 功效（2.8 sd），5x20 的 MDD 是 0.0086 [0.0071, 0.0107]，按同一规则属于"不确定"，28 对里只有 4 对通过（R940）。如果只看"排对顺序"，5x10 对 28 对的 p_correct 都 ≥ 0.93（R940）。
6. :29 "两个相似的线性模型（AUC ≈ 0.80）差 0.01 时" → 研究的这一对实际只差约 **0.004**（R940），p_correct 在 0.01 处是正态平移外推 Φ(0.01/sd)。MDD 只取决于 sd(A − B)，与差距无关；R940 的 28 对里有 25 对 5x20 的 MDD 整个 CI 低于 0.01，所以 MDD 的结论本身不受影响。
7. 补充（原文没有）：split 噪声只是下界。换一份 250 行数据，这一对的总体差距 sd 为 **0.0105**，是 5x20 split 噪声的 3.4 倍（R941，基于模型，只作参考）。

**E2**
8. :46、:73、:77 "0.906 [0.881, 0.916]" 的"90% 区间" → 只是名义 90%。R942 在 100 份复制样本上原样重跑，这个区间只覆盖复制总体真值（0.923，R941）的 **21%**，点估计平均偏低 0.031（RMSE 0.044）。报告应写"约 0.91–0.92，诚实的不确定度约 ±0.04（envelope 0.86–0.97）"。:48 的增益区间 [+0.022, +0.047] 同样偏窄；"每个 bootstrap 里增益都为正"由拟合边界 b ≥ 0 保证，没有信息量。
9. :49 "所以报告里给 0.88–0.92，把 0.95 视为乐观。这推翻了我事先的假设（以为 power law 区间会覆盖不足）" → "R942：power law 的名义 90% 区间只有 21% 覆盖复制总体真值（偏差 −0.031）；事先的'覆盖不足'假设平均而言成立；报告写约 0.91–0.92 ± 0.04"。已在原处更正。
10. :73 "A back-test that hides the largest sizes supports this form, which errs by +0.003 to +0.006, while a log-linear curve overshoots by up to +0.039" → 删除这条推论。单个样本上的回测不能代表平均表现；在复制样本上 power law 平均偏低（−0.031），log-linear 平均偏高（+0.017），真值在两者之间（R942）。
11. :46、:78 QDA "0.997 [0.995, 0.999]" 也是名义区间（R942 覆盖率 57%），但 QDA 的偏差只有 −0.0005，所以"500 行只多不到 0.002"的结论不变。

**Fix round 2, 2026-10-22**
12. **(fix round 2, 2026-10-22)** 英文 "Text for the report" 已按改正后的草稿 `reports/extensions/e1_validation_protocol.md`、`e2_learning_curve.md` 在原处重写（依据 `research/reports/fix_round1.md` §2 第 11 行、§7）。上面"下文正文保持原样"不再适用于英文部分；中文摘要仍以本节为准。
    - E1 段：加了 MDD 判据和功效（MDD = 2 sd ≈ 50% 功效；第 5 条）；"confirms decision D002" → "supports"；80% 功效下 5x20 的 MDD 0.0086 [0.0071, 0.0107]、5x10 对 28 对的 p_correct ≥ 0.93（第 5 条，R940）；QDA 对 split 噪声小 4.5–6.1 倍（第 1 条）；"the reference 5 x 50 gap … is only −0.003 … partly split noise" → 合并 1,000 个 repeat 的差距 −0.0044（SE 0.0004），官方种子的 −0.0101 是 1.84 sd（第 2、3 条）；补了 split 噪声只是下界（sd 0.0105，3.4 倍，R941）和 NB 检验最多 50 次中 1 次显著（第 4、7 条）。
    - E2 段：Old "An inverse power law predicts 0.906 [0.881, 0.916] … A back-test that hides the largest sizes supports this form, which errs by +0.003 to +0.006, while a log-linear curve overshoots by up to +0.039" → New: 0.906 带名义 90% 区间，曲线形式（log-linear 0.945、去掉 n = 50 为 0.869、envelope 0.857–0.974），R942 覆盖率 21%、偏差 −0.031（RMSE 0.044），报告写约 0.91–0.92 ± 0.04（第 8、10 条）。QDA "gains less than 0.001" → "less than 0.002 (+0.0008)，replica bias −0.0005"（第 11 条，与草稿一致）。
    - 表：表头标明 power law 和名义 90% 区间，L1 行加 "quote ≈ 0.91–0.92 ± 0.04 (R942)"；来源行把 R308 换成 R941 `20261022-092201_r941_e2_replica_truth`、R942 `20261022-094142_r942_e2_coverage`。
    - 数字核对自 `reports/tables/e1_paired_difference.csv`、`e1_qda_paired_difference.csv`、`r940_e1_pairs.csv`、`e2_extrapolation.csv`、`e2_qda_extrapolation.csv`、`r942_e2_coverage.csv` 和 R941 的 summary.json（`e1_pair_gap`，t = 250，sd 0.01045）。

## 中文摘要（给队员）

### 做了什么
1. **R300 选 E1 的候选对**（`research/r300_e1_pair_screen.py`，job `J20261022-063839-pro1-rext12`）：用官方 5-fold x 20 跑了 14 个监督候选（每个都是带 test_pred 的 experiment run），比较 11 对相邻候选，按**事先写进脚本的规则**（线性区间中差距最接近 0.01 的一对；QDA reg_param 中最接近 0.01 的一对）选 A/B：
   - 主对（线性区间）：in-fold |corr| top-30 vs top-50 + L2 logreg C=0.05：0.7910 vs 0.8011，差 −0.0101（`20261022-063705_r300_corr30_l2`、`20261022-063706_r300_corr50_l2`）。正好就是脚手架默认的那一对，所以 `config.yaml` 的模型没改，只补了理由注释。
   - 第二对（天花板区间）：QDA reg 0.1 vs 0.3（block 上）：0.9958 vs 0.9832，差 +0.0126。
   - 复现性检查：`r300_l1_c01` = 0.84693、`r300_qda_r01` = 0.99578，与 R003 完全一致。
2. **E1 完整运行**（`research/r306_e1_both_pairs.py`，job `J20261022-064231-pro1-rext12`）：两对各 50 个 split seed × 8 个方案 + reference 5×50，各 18,000 次拟合。
3. **E2 完整运行**（`research/r307_e2_all_models.py`，job `J20261022-070426-pro1-rext12`）：在**相同子集**上跑 (a) L1 基线、(b) QDA block、(c) TransductiveGMM，并画对照图。
4. 补充分析：**R304** 标签需求（2–200 个标签，job `J20261022-070740-pro1-rext12`）；**R308** 外推回测（只用小 size 拟合，预测已观测的大 size，job `J20261022-071012-pro1-rext12`）。

### 关键数字
**E1**（主对 run `20261022-064054_e1_validation_protocol`，QDA 对 run `20261022-064148_e1_validation_protocol_qda_pair`）

| 方案 | 每次决定的拟合数 | 线性对 std(A−B) | 线性对 MDD [95% CI] | QDA 对 MDD [95% CI] |
|---|---|---|---|---|
| hold-out 20% | 2 | 0.0287 | 0.057 [0.048, 0.072] | 0.021 [0.018, 0.026] |
| 3-fold | 6 | 0.0130 | 0.026 [0.022, 0.032] | 0.0057 [0.0048, 0.0072] |
| 5-fold | 10 | 0.0149 | 0.030 [0.025, 0.037] | 0.0051 [0.0043, 0.0064] |
| 10-fold | 20 | 0.0149 | 0.030 [0.025, 0.037] | 0.0051 [0.0043, 0.0063] |
| 5x10 | 100 | 0.0042 | 0.0083 [0.0070, 0.0104] | 0.0014 |
| 5x20（官方） | 200 | 0.0031 | **0.0061 [0.0051, 0.0076]** | 0.0011 |

- 结论：两个相似的线性模型（AUC ≈ 0.80）差 0.01 时，**只有官方 5x20 的 MDD 整个 CI 都低于 0.01** → D002 成立；5x10 处于边界；单次 k-fold 不论 k 多大 MDD 都约 0.03；hold-out 约 0.06（只有 64% 概率排对）。
- 增加 k 没用（3/5/10-fold 的 std 都是 0.013–0.015），增加 repeat 有用（约 1/√R）。
- 单次 5x20 run 内部的 `se_split`（0.0030）与真正的 seed 间 std（0.0031）一致 → `compare_runs.py` 的 split SE 可信。
- 配对：5x20 的差值 std 从不配对的 0.0050 降到 0.0031。
- noshuffle：50 个 seed 都是同一个 split，spread = 0，而这个 split 给出的 A−B = −0.0097（reference 只有 −0.0029），是假精度。
- 接近天花板的 QDA 对：split 噪声小 4.5–6 倍，3-fold 就够；hold-out 仍不够。
- **意外/教训**：线性这一对的真实 CV 差距很小，reference 5x50 为 −0.0029（SE 0.0014），50 次 5x20 的平均为 −0.0044。官方种子上的 −0.0101 偏离约 2 sd。“从 11 对里挑差距最接近 0.01 的”本身就挑中了被噪声放大的一对（winner's curse）。MDD 结论不受影响，但说明单次 5x20 的点估计也有约 ±0.006 的 split 噪声。NB 校正 t 检验在 50 次中从未判这一对显著。

**E2**（(a) `20261022-070248_e2_learning_curve`，(b) `20261022-070339_e2_learning_curve_qda_block`，(c) `20261022-070445_e2_learning_curve_gmm`，对照 `20261022-070518_r307_e2_model_comparison`）

| n（有效训练行数） | (a) L1 mean ± std | (b) QDA mean ± std | (c) GMM |
|---|---|---|---|
| 50 (40) | 0.659 ± 0.100（24.5% 折为常数预测） | 0.973 ± 0.022 | 1.000 ± 0 |
| 100 (80) | 0.735 ± 0.040 | 0.990 ± 0.005 | 1.000 |
| 150 (120) | 0.810 ± 0.028 | 0.992 ± 0.003 | 1.000 |
| 200 (160) | 0.830 ± 0.021 | 0.995 ± 0.0014 | 1.000 |
| 250 (200) | 0.847 ± 0.006（只有 CV 噪声） | 0.996 ± 0.0005 | 1.000 |
| 外推 500 | **0.906** [0.881, 0.916]；envelope [0.857, 0.974] | **0.997** [0.995, 0.999] | 1.000 |

- L1 基线还缺数据：250→500 预测增益 +0.040 [+0.022, +0.047]；但曲线形式不确定性（log-linear 0.945，去掉 n=50 为 0.869）远大于 bootstrap 区间。
- R308 回测：只用 n = 50–150 拟合，power law 对 x = 160/200 的预测误差只有 +0.003/+0.006（L1）、−0.002/−0.003（QDA），区间都覆盖；log-linear 高估 +0.02–0.04。*（本条结论已更正，2026-10-22，见文首 Errata 第 9 条）* 但一份样本上的回测说明不了这套流程平均的表现。R942（`20261022-094142_r942_e2_coverage`）在 100 份 250 行复制样本上原样重跑 E2：power law 的名义 90% 区间只有 **21%** 覆盖复制总体的真值，点估计平均偏低 0.031（RMSE 0.044）。所以我事先的假设（power law 区间会覆盖不足）平均而言是对的。报告里应写 **约 0.91–0.92，诚实的不确定度约 ±0.04**（envelope 0.86–0.97），不再写 0.88–0.92。
- QDA 在 40 行训练时就有 0.973，约 100 行后几乎平，500 行只多 +0.0008。这部分推翻了 r302 的假设（原以为 QDA 很吃数据）。
- GMM：所有 size、所有子集都是 1.0。R304（`20261022-070600_r304_label_requirement`）显示 **2 个标签（每类 1 个）时 50 组全部 AUC = 1.0**。QDA 在每类行数少于 13 维时根本拟合不了（n ≤ 20），75 个标签时中位数才过 0.99；L1 在 20 个标签以内是常数预测。
- 副产品：L1 的 C=0.1 偏强。R300 中 C=0.2 的官方 CV 为 0.873（`20261022-063711_r300_l1_c02`），比 C=0.1 高 0.026。

### 对团队的意义
- D002（官方 5x20）可以从“暂定”改为“确认”（证据：上面的 E1 run）。这需要队员在 `registry/decisions.md` 写新条目（我没有改 registry）。`docs/VALIDATION_PROTOCOL.md` 的 `TODO(E1)` 可以用这些数字填。
- 监督候选之间 0.01 的差别必须看配对差的 split SE，而且要警惕“从很多候选里挑最大差距”。
- 最终模型（GMM）在标签上没有提升空间；剩下的不确定性在边界样本上，学习曲线看不到。

### 需要注意 / 问题
- `colab/remote.py` 在 06:24 版本里 `prepare()` 少了 `return want`，导致 `run` 在启动前崩溃（AttributeError）。我没有改 colab/，而是用 scratchpad 里的一个包装脚本（只把 prepare 的返回值补上，哈希取自实际上传的包），跑了 R300 和 E1 两个 job。06:40 维护者修好之后，其余 job 用的是原版 remote.py。所有 run 的 `meta.json` 都记录了正确的 code_bundle。
- E1 job 的日志流在本机卡住了（作业在 VM 上已 exit 0），最后用 `remote.py fetch J20261022-064231-pro1-rext12 --data raw` 取回。所以 compute_jobs.csv 里这一行的 wall_seconds = 0。
- run_id 的时间戳比 job_id 早 1–2 分钟（VM 时钟），不影响任何结果。
- 扩展 run 没有 test_pred；R300 的 14 个监督候选有 test_pred，但都远差于 GMM，不适合提交。
- 我新增的文件只在 `research/`（r300–r308、lib_ext12），另外改了 `extensions/e1_validation_protocol/config.yaml` 和 `extensions/e2_learning_curve/config.yaml`（只加注释，模型设置未变），以及两个扩展 README 的 Results log。`run.py` 没改。

### 用到的库/方法（请加入 `registry/references.md`）
scikit-learn（StratifiedKFold、StratifiedShuffleSplit、LogisticRegression/liblinear、QuadraticDiscriminantAnalysis、GaussianMixture）、SciPy（`curve_fit`、chi2、t 分布）、NumPy、pandas、matplotlib、joblib；Nadeau & Bengio 2003（校正 t 检验）、Hanley & McNeil 1982、Bengio & Grandvalet 2004、Kohavi 1995、Figueroa et al. 2012（inverse power law 学习曲线）、Efron 1979（bootstrap）、Dempster, Laird & Rubin 1977（EM / GMM，**registry 里还没有**）、Friedman 1989（regularised discriminant analysis，QDA 的 reg_param，**registry 里还没有**）。没有用外部数据；无监督步骤只用 train+test 的特征（D007）。

## Text for the report

**Validation protocol (E1 evidence).** We executed each candidate scheme with 50 split seeds while keeping the model seeds fixed. The schemes were a single stratified hold-out, k-fold with k = 3, 5, 10, and repeated 5-fold. For each seed we scored two neighbouring candidates on identical splits; the pairs were chosen by a pre-registered rule from 11 pairs screened under the official CV. For two similar linear models (in-fold top-30 vs top-50 features + L2 logistic regression, AUC ≈ 0.80), the minimal detectable difference MDD = 2 x std(A − B) was 0.057 for a 20 % hold-out and 0.026–0.030 for a single 3-, 5- or 10-fold run. Repeated 5-fold brought it to 0.0083 with 10 repeats and **0.0061 [95 % CI 0.0051, 0.0076] with 20 repeats** (run `20261022-064054_e1_validation_protocol`). A true gap equal to the MDD is detected with only about 50 % power at α = 0.05. By this criterion only the official 5 x 20 scheme resolves a 0.01 AUC difference with its whole confidence interval, which supports decision D002. The verdict depends on the criterion: at 80 % power (2.8 sd) even 5 x 20 is inconclusive (0.0086 [0.0071, 0.0107]), while for ordering alone 5 x 10 already gets a 0.01 pair right with probability ≥ 0.93 for all 28 pairs of 8 linear candidates (R940, `20261022-091211_r940_e1_pairs_check`). More folds do not reduce the noise; more repeats do. The split SE that a single repeated run reports (0.0030) matches the seed-to-seed spread (0.0031), so `compare_runs.py` can be trusted. Near the ceiling the problem is much easier: for QDA with reg 0.1 vs 0.3 (AUC 0.996 vs 0.984) split noise is 4.5–6.1 times smaller in every k-fold scheme and even 3-fold has MDD 0.0057 (run `20261022-064148_e1_validation_protocol_qda_pair`). Two cautions emerged. First, the studied linear pair is only about 0.004 apart: pooling all 1,000 repeats gives A − B = −0.0044 (SE 0.0004; R940), so the −0.0101 seen under the official seed was a 1.84-sd draw, favoured by picking the pair closest to 0.01. Second, split noise is conditional on these 250 rows and is a lower bound: across 250-row replica samples the population gap of the pair has sd 0.0105, 3.4 times the 5 x 20 split noise (R941, `20261022-092201_r941_e2_replica_truth`, model-based), and the Nadeau–Bengio corrected test, which takes this view, finds the linear difference significant in at most 1 of 50 executions.

**Learning curves and label requirement (E2 evidence).** On identical stratified subsets of 50–250 rows (10 per size, 5-fold x 4 CV inside each), the supervised L1 baseline rises from 0.659 to 0.847 and is still data-limited. An inverse power law gives 0.906 for 500 training rows, with a nominal 90 % bootstrap interval [0.881, 0.916]; the curve form matters more (log-linear 0.945, the fit without n = 50 0.869; envelope 0.857–0.974). A coverage simulation that replays the whole procedure on 100 replica 250-row samples shows that the nominal interval covers the replica truth at 500 rows (0.923) in only 21 % of samples, and that the power-law point is low by 0.031 on average (RMSE 0.044; log-linear +0.017; R941, R942 `20261022-094142_r942_e2_coverage`). We therefore quote about 0.91–0.92 with an honest uncertainty of about ±0.04 (envelope 0.86–0.97). Supervised QDA on the 13-column block is already at 0.973 with 40 training rows and would gain less than 0.002 (+0.0008) from 250 to 500 (replica bias −0.0005, R942). The transductive GMM scores 1.0 at every size, and with only two labels (one per class) it ranks the remaining rows perfectly in all 50 trials (run `20261022-070600_r304_label_requirement`). Modelling the unlabelled feature distribution therefore lowers the label requirement by orders of magnitude, from roughly 75 labels (QDA, median AUC 0.99) to 2.

| Model | AUC at n = 50 (40 train rows) | AUC at n = 250 (200) | Predicted AUC at 500 training rows (power law; nominal 90 % interval) | Labels needed for median out-of-subset AUC >= 0.99 |
|---|---|---|---|---|
| L1 logistic, all 150 features | 0.659 ± 0.100 | 0.847 ± 0.006 | 0.906 [0.881, 0.916]; quote ≈ 0.91–0.92 ± 0.04 (R942) | not reached (<= 200) |
| QDA reg 0.1, 13-column block | 0.973 ± 0.022 | 0.996 ± 0.0005 | 0.997 [0.995, 0.999] | 75 |
| Transductive GMM K = 2 | 1.000 ± 0 | 1.000 ± 0 | 1.000 | 2 |

*Sources: runs `20261022-070248_e2_learning_curve`, `20261022-070339_e2_learning_curve_qda_block`, `20261022-070445_e2_learning_curve_gmm`, `20261022-070600_r304_label_requirement`, `20261022-092201_r941_e2_replica_truth`, `20261022-094142_r942_e2_coverage`. The R308 back-test (`20261022-071154_r308_e2_backtest`) is no longer used as an argument (Errata 10).*
