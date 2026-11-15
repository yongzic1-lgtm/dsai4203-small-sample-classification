# ext34 报告：E3 信息泄漏 + E4 特征排序稳定性（真实数据，Colab 完整运行）

作者：ext34（Claude Code，AI 代理）· 2026-10-22 · Colab 账号 pro1，会话 rext34（CPU High-RAM，8 核；已 `down`）

## 勘误（2026-10-22 晚，fix_ext34，依据第 1 轮核查 `research/reports/verify.md`、`verify_ext.md`、`verify_leak.md`）

正文已就地改正；下面列出改了什么、依据是哪个文件。
1. **E3 "gap 0.073 ± 0.014"**：± 0.014 是单次重复的 gap 在 20 个 split seed 之间的 SD，不是 SE。推断用的是 Nadeau–Bengio 校正 SE = 0.0733 / 3.72 = **0.020**（p = 0.0003）；0.014/√20 = 0.003 只是"给定这 250 行"的 split 噪声（`experiments/20261022-063908_e3_leakage/cells.csv`，k_feat 20，5 折）。
2. **E3 LB 不再 pending**：E3 最终模型已作为 S004 上传（2026-10-22 16:08 JST），public **0.79136**（`registry/submissions.csv`；`reports/tables/e3_leaderboard.csv` 已用 `--refresh-lb` 刷新）。leaky − public = +0.075，约 **3.2** 个 public SE（SE 0.0233 = 该表的 `public_lb_se_approx`，Hanley–McNeil，AUC 0.79，395 行；verify_ext 用 AUC 0.82 处的 0.021 得到 3.6）；honest − public = +0.002；proxy 0.820 − public = +0.028（1.2 SE），落在预登记的 90% 区间 0.785–0.854 内（B005）。原文"约两个 public SE"指的是 leaky 相对 proxy 的 0.047。
3. **E3 小 k_feat 的解释**（R944 `20261022-161150_r944_e3_replica_decomp`，100 份复制样本）：平均而言 honest CV 对它自己的模型无偏（|bias| ≤ 0.004），小 k_feat 的平均 gap 很小（k_feat = 2 为 0.015）且来自泄漏乐观。但本样本 k_feat = 2 的 0.056 在复制样本里排第 96 百分位；gap 同样 ≥ 0.04 的 8 份复制样本里，超额来自 honest CV 偏低（bias −0.037），和本样本对代理标签的分解一致（leaky − proxy +0.011，honest − proxy −0.045）。所以"强特征之间的 winner's curse"平均只解释约 0.014。k_feat = 7 的 gap ≈ 0 也部分是运气（复制样本中位数 0.006）。
4. **E3 补充**：代理标签上最优的是 k_feat = 10（0.864）；honest CV 选 k_feat = 7（代理 0.852），leaky CV 选 k_feat = 50（代理 0.825）。
5. **E3 数字**（扩展报告表 1 与正文）：k_feat 5 的 leaky 0.817 → 0.816；k_feat 10 的 gap 0.026 → 0.025；k_feat 13 的 gap 0.045 → 0.044；null leaky "0.62–0.71" → 0.61–0.71；"~48 个特征"是纯噪声下的期望，真实标签下有 61 个特征的 |corr| > 1/√250；置换标签下 gap 在 k_feat = 20–30 是平台，不是"在 30 达到峰值"；honest 相对 proxy 偏低 0.027 → 0.026（proxy_auc_hard 0.8195；0.027 是对 395 行模拟均值 0.8200 的差）。
6. **E4 drop 的位置**：≥ 0.986 / 98.6% → ≥ 0.985 / 98.5%（80 和 77 在 k = 10 为 0.9855）；"k ≥ 8 时位置不随 k 变" → 只跑了 k = 5/10/13/20，应写"k = 10/13/20"。而且 config 预登记的主规则是 perm（`drop.primary_rule: perm`），m* = 7/8/9/17，会随 k 变；gap/half 规则不随 k 变，但 half 在置换标签下 k ≥ 10 时 72–100% 的曲线会收进纯噪声特征，perm 只有 6–8%（R943 `20261022-100104_r943_e4_null_step`）。"sharp drop" 成立但余量不大：每个 k 的最大跳都只被 50 条置换曲线中的 1 条超过（p ≈ 0.02–0.04）。
7. **E4 "6 个块特征从不 stable"** 只对 top-10 / freq ≥ 0.5 的定义成立；perm 规则下 67 在 k = 13/20、8 和 26 在 k = 20 是 stable 的（`reports/tables/e4_stable_sets.csv`）。
8. **E4 数字与补充**：表 2 中 all 150 的 L2 ± 0.019 → ± 0.018，stable 的 ± 0.006 → ± 0.005；stability selection − honest top-m 在 k = 20 的 −0.007 → −0.006；"far above any linear model" → R300 最好的线性模型（块上 L2，C = 1，`20261022-063713_r300_block_l2_c1`）是 0.898，QDA 0.917 只高 0.018。补充主规则下的有用性检验：k = 20 时 perm 集（17 个特征，每折重选）比 all 150 只高 +0.028（p = 0.30，`e4_paired.csv`）。
9. **"需要主会话处理"第 1 条已完成**（S004）。`extensions/e3_leakage/run.py` 现在在参考 k_fold 行的 split signature 等于官方 signature 时把 `official_cv: true` 写进 summary.json，make_submission 就不会再打印误导性的警告。只影响以后的 run；`20261022-063908_e3_leakage` 的文件夹是 append-only，不补写。
10. **两个扩展报告**已按上述改正，加上可见的引用和一行 AI 说明，并压到 2 页以内（XeLaTeX，Calibri 11 pt，A4，2.54 cm 页边距：E3 1.74 页，E4 1.89 页；按 Word 默认段后 8 pt、行距 1.08 估计约 1.95 / 2.06 页）。为此 E3 去掉了图 `e3_gap_vs_kfeat.png`（2 折和 10 折的 gap 改成表 1 的一列），E4 去掉了原表 1（各规则的 m* 已在图的面板标签和正文里）。

## 中文摘要（给队友）

**做了什么**
- 按真实数据配置并完整运行 E3、E4（`extensions/e3_leakage/config.yaml`、`extensions/e4_feature_stability/config.yaml`，改动理由写在两个 README 的结果记录里）。另写两个研究脚本：`research/r320_e4_stable_vs_block.py`（stable 集与无标签高方差块的比较 + 三类模型的有用性检验）、`research/r321_e3e4_proxy_test.py`（用 GMM 伪标签在 19,750 行 test 上估计 E3/E4 模型的 test AUC）。共享代码 `research/lib_ext34.py`。
- 草拟英文扩展报告：`reports/extensions/e3_leakage.md`、`reports/extensions/e4_feature_stability.md`（≤ 2 页，四个标题，Author: TODO）。

**E3 关键数字**（run `20261022-063908_e3_leakage`，5-fold = 官方 splits，20 个 split seed）
- k_feat = 20：leaky 0.867 ± 0.008，honest 0.793 ± 0.014（± 为 20 个 split seed 间的 SD），gap 0.073（SD 0.014；Nadeau–Bengio 校正 SE 0.020，p = 0.0003；20 个 seed 全部 leaky 更高）。
- gap 对 k_feat 非单调：k_feat = 7 时 gap = 0.001（7 个强特征 98, 17, 100, 112, 80, 77, 46 每折都被重新选中，fold overlap 0.999；恰好为 0 部分是运气，R944 复制样本中位数 0.006）；k_feat = 2 时 0.056（本样本偏高：复制样本第 96 百分位，主要是 honest CV 偏低；平均只有 0.015，来自泄漏乐观，R944）；k_feat = 30-50 时最大 0.090；k_feat = 150 时恰好为 0（sanity PASS）。
- k_fold：真实数据 k_feat = 20 的 gap 为 0.087 / 0.082 / 0.073 / 0.063（2/3/5/10 折），原因是 honest 随训练行数变多而变好（0.767 → 0.805）；置换标签下 gap 不随 k_fold 变（0.201 / 0.206 / 0.210），与一阶理论一致。
- null：honest 0.49-0.51，leaky 最高 0.71（k_feat = 30）。四个 sanity check 全部 PASS。
- leaky CV 会选错模型：leaky 最好是 k_feat = 50（0.894），honest 最好是 k_feat = 7（0.834）；proxy test 上 k_feat = 7 得 0.852，k_feat = 50 得 0.825，代理最优是 k_feat = 10（0.864）。
- **LB 对照**：`experiments/20261022-063908_e3_leakage/test_pred.npy`（k_feat = 20，在全部 250 行上拟合）已作为 S004 提交，public **0.79136**（2026-10-22 16:08 JST）。leaky − public = +0.075（约 3.2 个 public SE，SE 0.0233），honest − public = +0.002。提交前的预期：proxy test AUC 0.820（R321 `20261022-072942_r321_e3e4_proxy_test`，19,750 行 GMM 伪标签），随机 395 行模拟的 public = 0.820 ± 0.021（90%：0.785-0.854，B005）；实际低 0.028（1.2 SE），在区间内。

**E4 关键数字**（run `20261022-071414_e4_feature_stability`，B = 2000，k = 5/10/13/20）
- 有 sharp drop（每个 k 的最大跳只被 50 条置换曲线中的 1 条超过，p ≈ 0.02–0.04，R943）：7 个强特征在 k = 10/13/20 时频率 ≥ 0.985；49 居中（0.04/0.65/0.79/0.92）；其余 ≤ 0.28（k = 10）。预登记的主规则 perm 给 m* = 7/8/9/17，会随 k 变（k = 20 时 9 个 |corr| 0.13–0.18 的弱特征超过噪声包络）；gap 规则 7/8/8/8，half 7/8/8/9（k = 10/13/20 时不变，但 half 在噪声下 k ≥ 10 时 72–100% 的曲线收进纯噪声特征）；binom 7/17/22/28（严重偏宽：置换标签下 top 噪声特征在 k = 10 时频率的 95% 分位可达 0.84）。
- stable 集永远等于 top-m |corr|（Jaccard 1）——稳定性没有提供 |corr| 排名之外的信息。
- 有用性（L2 logreg，官方 CV）：honest stable（k = 10）比 all150 高 0.063（p = 0.038）；主规则 perm 在 k = 20 的 17 特征集（每折重选）只高 +0.028（p = 0.30）；naive − honest = −0.001（k = 10）/ +0.034（k = 5）/ +0.069（k = 20 的 17 特征 perm 集）；stability selection 从不显著好于 honest top-m。
- **与无标签高方差块比较**（R320 `20261022-074624_r320_e4_stable_vs_block`）：stable 集（k = 10）= {17, 46, 49, 77, 80, 98, 100, 112} = 7 个块特征 + 17。**17 是块特征的精确线性组合**（无标签 R² = 1 − 3e-14）；同样的还有 6, 52, 57, 61, 142, 144——共 20 个特征张成 12 维块子空间，正好对应 R001 的 8 个零特征值。所以 R002 说的"块外特征也与目标相关"并不意味着块不完整：块的张成已经包含它们。6 个块特征（8, 15, 26, 30, 64, 67）从不进入 top-10 / freq ≥ 0.5 的 stable 集（15/30/64 甚至低于 chance；但在 perm 规则下 67 在 k = 13/20、8 和 26 在 k = 20 算 stable），但它们的类别信息在方差里（GMM 两分量间 d ≈ 0，log 方差比达 −1.0/+0.63/+0.83），单独就能让 QDA 达到 0.917、GMM 达到 0.932（R300 最好的线性模型，块上 L2 C = 1，是 0.898）。honest CV：块 > stable，L2 0.862 vs 0.832（p = 0.018），QDA 0.996 vs 0.961（p = 0.0006），GMM 1.000 vs 0.983（p = 0.009）；块 ∪ stable = 块。结论："stable" ≠ "值得保留"，对协方差型信号（本队最终模型）而言，边际 |corr| 排名不是选特征的依据。

**对团队的意义**：(1) 验证协议"所有用到标签的步骤都放进 pipeline"有了定量证据（报告 §2 可引用 E3）。(2) 最终模型使用的块已经是完整的信息子空间；把 17 等块外特征加进 GMM/QDA 不会带来新信息（R320：QDA +0.0004，p = 0.44）。(3) proxy test AUC（GMM 伪标签）可以在不花提交次数的情况下比较非直推式模型，但它偏向与 GMM 假设一致的模型（如 QDA-on-block），不能替代 LB。

**需要主会话处理**
- （已完成）提交 E3 最终模型一次（`scripts\make_submission.py --run 20261022-063908_e3_leakage --desc "E3 final model corr_topk(20)" --expected-public 0.82`）。该 run 是 `kind = extension`，summary 里没有 `official_cv: true`，make_submission 打印了警告（5-fold 行确实用的是官方 splits，split_sig 5708ce5c3828）。结果：S004，public 0.79136；`--refresh-lb` 已运行，报告已改；run.py 已改为以后的 run 自动写 `official_cv`（勘误第 9 条）。
- `registry/references.md` 建议把 ambroise2002、meinshausen2010、efron1979 标为 used，并新增 Hastie et al. 2009 的 7.10.2 节引用（我不能改 registry/）。
- E3 那次作业因为 `colab/remote.py` 在 06:24–06:40 之间的 `prepare()` 缺少 `return want` 而崩溃，之后改用 `--no-sync` 运行（VM 上的代码 = `logs/code_bundles/b413e6aa3bae4443.tar.gz`，已写进 notes.md）；remote.py 在 06:40 被修好。

**运行清单**

| run_id | 内容 | Colab job |
|---|---|---|
| 20261022-063908_e3_leakage | E3 完整网格 + null + 最终模型 test_pred | J20261022-064045-pro1-rext34 |
| 20261022-071414_e4_feature_stability | E4 完整（B = 2000） | J20261022-071554-pro1-rext34 |
| 20261022-072731_r320_e4_stable_vs_block | R320 第一次（数值有效，图被替代） | J20261022-072910-pro1-rext34 |
| 20261022-074624_r320_e4_stable_vs_block | R320 重跑（同样的数值 + 块张成列 + 可读的图） | J20261022-074803-pro1-rext34 |
| 20261022-072942_r321_e3e4_proxy_test | GMM 伪标签 proxy test AUC（E3 各 k_feat、E4 各集合） | J20261022-073122-pro1-rext34 |

**修改过的文件**：两个扩展的 config.yaml；两个 run.py 的配色修复（第 4 个颜色；E3 null 图标题重叠，在正式 run 之后修，只影响以后的图）；README 结果记录；两个扩展报告草稿；以及 research/lib_ext34.py、r320、r321。

**库与方法**：scikit-learn 1.9.1（LogisticRegression、QuadraticDiscriminantAnalysis、GaussianMixture、StandardScaler、StratifiedKFold、roc_auc_score）、NumPy、pandas、SciPy、matplotlib、joblib；方法：Ambroise & McLachlan 2002（选择偏差）、Hastie et al. 2009 §7.10.2、Nadeau & Bengio 2003（corrected t-test）、Hanley & McNeil 1982（AUC SE）、Efron 1979 / Efron & Tibshirani 1993（bootstrap）、Meinshausen & Bühlmann 2010（stability selection）、Dempster et al. 1977 / GMM-EM（直推式 GMM，D007）。**合规**：没有使用外部数据；无监督步骤（方差块、GMM、R²）只用 train+test 特征（D007 允许）；GMM 伪标签只用于分析，不用于训练任何提交模型；没有提交 Kaggle。

---

## Text for the report

**E3 (information leakage).** Choosing the top-k features by |corr| on all 250 rows before
cross-validation inflates the CV AUC of an L2 logistic model by 0.073 at k_feat = 20 (SD over 20 split
seeds 0.014; Nadeau-Bengio corrected SE 0.020, p = 0.0003; leaky 0.867 vs honest 0.793 on the official
5-fold x 20 splits), and by up to 0.090 at k_feat = 30-50. The submitted k_feat = 20 model (S004) scored
0.79136 on the public LB: the leaky CV was 0.075 too high (3.2 LB standard errors), the honest CV only
0.002. The inflation is not monotone in k_feat. It vanishes at k_feat = 7, because the seven strong
features (|corr| 0.34-0.40) are re-selected inside every fold, and it grows above 7, as noise features
picked for their chance correlation with the validation labels enter the model. On average (100 replica
datasets, R944) the small gap below 7 is leak optimism; in our sample the gap at k_feat = 2 (0.056, a
96th-percentile value) comes mostly from a low honest estimate. With permuted labels the honest CV stays
at 0.49-0.51, while the leaky CV reaches 0.71. To first order the gap does not depend on the number of
folds (null: 0.201 / 0.206 / 0.210 for 3 / 5 / 10 folds). On the real data it shrinks from 0.087 to 0.063
between 2 and 10 folds, because larger training parts make the honest in-fold selection more accurate.
Leaky CV would also have chosen k_feat = 50 instead of the honest choice k_feat = 7 (AUC against the
team's GMM proxy labels on 19,750 test rows 0.825 vs 0.852; the proxy optimum is k_feat = 10, 0.864).

**E4 (stability of the ranking).** Over 2000 bootstrap resamples the same seven features enter the
top-k in >= 98.5% of resamples for k = 10, 13 and 20, and the sorted frequency then drops sharply
(largest step 0.30-0.49, exceeded by 1 of 50 permuted-label curves, p = 0.02-0.04). The strong group is
above the drop for every k >= 10. Under the pre-registered permutation rule the drop sits at 7-9 features
up to k = 13 and at 17 for k = 20, where weak features (|corr| 0.13-0.18) clear the noise envelope. The
stable set always equals the top-m set by |corr|. It beats all 150 features for a linear model (+0.063
honest CV; the 17-feature set at k = 20 only +0.028, p = 0.30), but it is not the set worth keeping. Six
of the 13 label-free high-variance features never enter the top-10 stable set, yet their class signal
lies in the variance. On its own this part of the block gives QDA 0.917 and a Gaussian mixture 0.932, and
the full block beats the stable set for every model family. Conversely, the second most stable feature
(17) is an exact linear combination of the block and adds nothing to it. Being stably selected measures
how reproducible a feature's marginal correlation is within this sample; being worth keeping depends on
the model and has to be judged by honest CV.

| | E3: leaky CV | E3: honest CV | E3: gap | E4: L2 logistic | E4: QDA | E4: GMM |
|---|---:|---:|---:|---:|---:|---:|
| k_feat = 7 / stable set (7.6 feat.) | 0.835 | 0.834 | 0.001 | 0.832 | 0.961 | 0.983 |
| k_feat = 20 / label-free block (13) | 0.867 | 0.793 | 0.073 | 0.862 | 0.996 | 1.000 |
| k_feat = 50 / block minus stable (6.4) | 0.894 | 0.803 | 0.090 | 0.765 | 0.917 | 0.932 |
| k_feat = 150 / all 150 | 0.769 | 0.769 | 0 | 0.769 | n/a | 1.000 |

*Official CV (5-fold x 20 seeds, seed 2026), mean over seeds. E3 run 20261022-063908_e3_leakage; E4 runs
20261022-071414_e4_feature_stability and 20261022-074624_r320_e4_stable_vs_block; GMM on all 150 =
20261022-050732_r003_gmm2_all. Public LB of the E3 model: S004 = 0.79136.*
