# fm 报告：预训练表格基础模型与贝叶斯方法（R250-R257）

作者：AI 代理 "fm"（Claude Code），账号 pro2，会话 rfm（L4 GPU VM，12 核）。所有计算都在 Colab 上完成。8 个作业，见 `registry/compute_jobs.csv` 中的 `J20261022-0641..0752-pro2-rfm`。本地只做过不超过 40 秒的冒烟测试，结果写进 scratch 档案。

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要和 "Text for the report"）。依据：`research/reports/verify_rules.md` §1、`research/reports/verify_cand.md` §1、`research/reports/verify.md` §2 C7、C12、G5。

1. **Pretraining data**（合规说明第 1、2 条；Text for the report 第 1 段和 "Rule compliance"）。Old: "TabPFN v2 只在合成先验数据上预训练（…约 1.3 亿个…合成数据集）…TabICLv2 也是只用合成数据预训练…两者都没有在真实的外部数据集上训练"、"（合成）数据"、"both pretrained exclusively on synthetic prior data"、"never trained on real external data … (on synthetic data)" → New: "pretrained by their authors on prior-generated data (Hollmann et al. 2025; Qu et al. 2026); never trained on this competition's data". Reason: no archived source. The archived model cards (`experiments/20261022-063927_r250_fm_licence_check/tabpfn_README.txt`, `tabicl_README.txt`) do not state the training data of these checkpoints (verify_rules §1). The "about 130 million datasets" detail was removed for the same reason.
2. **TabICLv2 venue**（合规说明第 1 条；用到的库/方法）。Old: "ICML 2026" → New: "arXiv preprint arXiv:2602.11139". Source: `experiments/20261022-063927_r250_fm_licence_check/tabicl_README.txt:34` (BibTeX `journal={arXiv preprint arXiv:2602.11139}`).
3. **"Lowest log-loss"**（对团队的意义第 1 条）。Old: "CV 1.0000，log-loss 最低" → New: lowest CV (fold-mean) log-loss in the fm track (0.0094); self-training QDA-LW has a lower OOF log-loss (0.0071). Sources: `20261022-070200_r255_bqda_eb_hv` summary.json `cv_logloss` 0.00937; `20261022-080632_r209_selftrain_qda_lw` summary.json `oof_logloss` 0.00709. The comparison with the GMM (0.009 vs 0.017) is like-for-like (GMM fold-mean CV log-loss 0.0171 in `20261022-064436_r102_ref_s003` summary.json) and stays.
4. **Boundary-row replica result**（结论"在边界行上"第 2 条；"Where unlabeled data still matter"）。Old: "0.825 vs 0.702 … 0.795; 0.9949 / 0.9948" without sample size → New: marked as unverified and based on few replicas; r936 cited for the direction. Note: the synthesis says "6 replicas". `reports/tables/r254_replica_auc.csv` has 6 replicas in total, but the quoted numbers are the means of the 3 replicas of the K = 2 generator (`G1_gmm2`). The 3 replicas of the DP-GMM generator (`G2_dpgmm`) give 0.830 vs 0.701. Supporting evidence: `reports/tables/r936_cv_power.csv` (r255's true AUC below S003's in 222/222 replicas, mean gap −1.2e-4).
5. **(fix round 2, 2026-10-22) "Best linear model"**（结论"Horseshoe"第 1 条；Text for the report 第 1 段）。Old: "全特征版本的 0.9261 是目前最好的**线性**模型"、"(0.926, the best linear model)" → New: "全特征版本的 0.9261 是全部 150 个特征上最好的线性模型（13 列块上 L2 logistic 是 0.938）"、"(0.926, the best linear model on all 150 features; on the 13-column block L2 logistic reaches 0.938)". Source: R440 claim F21 (`reports/tables/r440_core_numbers.csv`, run `20261022-184020_r440_core_numbers`; `research/reports/r2_audit.md` §5 item 4); `20261022-074147_r207_l2_block_tt_wide` auc_mean 0.93830.
6. **(fix round 2, 2026-10-22) QDA reg 0.03**（关键数字表）。Old: "0.9990" → New: "0.9989". Source: R440 claim F13 (r2_audit §5 item 5); `20261022-070545_r256_qda_reg0.03_hv` auc_mean 0.998950. The table in "Text for the report" has no reg 0.03 entry, so nothing changed there.
7. **(fix round 2, 2026-10-22) Side agreement on all test rows**（Text for the report "Where unlabeled data still matter"）。Old: "99.5% of all test rows" → New: "99.6% of all test rows". Source: R440 claim F27 (r2_audit §5 item 5); `20261022-070200_r255_bqda_eb_hv` summary.json `agreement_with_s003.side_agree_all` 0.99554.
8. **(fix round 2, 2026-10-22) Boundary-row result confirmed in direction**（结论"在边界行上"第 2 条；"Where unlabeled data still matter"）。Old: "没有人复核过。方向得到 r936 的支持"、"and have not been re-verified. A larger replica study supports the direction" → New: the GMM 0.825 vs BQDA 0.702 result (r254, few replicas) is confirmed in direction by r404 on 222 replicas: on the test rows whose true Bayes posterior is in (0.1, 0.9) (about 185 per replica), S003 0.804 vs r255 (Bayesian QDA) 0.678, S003 better in 222/222 replicas; on r254's own definition (rows with an S003 score in (0.1, 0.9)) 0.814 vs 0.686. Source: r404 `20261022-194238_r404_inductive_pooled`, `reports/tables/r404_boundary.csv` (`bnd_auc_clean_mean` 0.80415 / 0.67772, `bnd_auc_clean_share_s003_better` 1.0; `s003bnd_auc_clean_mean` 0.81399 / 0.68589), `research/reports/r2_inductive.md` §3 Table B1. This replaces the "unverified" status of item 4.

## 中文摘要（给队员）

### 做了什么
1. **基础模型的许可与下载检查（R250，analysis）**：
   - TabPFN v2 的权重（`ModelVersion.V2`，HF 仓库 `Prior-Labs/TabPFN-v2-clf`）不需要任何 token 就能匿名下载，HF 上没有 gated。许可是 Prior Labs License，即 Apache-2.0 加一条署名要求。
   - tabpfn 9.2 的默认模型属于 TabPFN-2.5 及以后的系列，权重为非商用许可。在没有 token 的情况下它抛出 `TabPFNLicenseError`，要求先在 ux.priorlabs.ai 接受许可并设置 `TABPFN_TOKEN`。按规则**没有使用这个模型**。
   - TabICL 2.2.0 用的 checkpoint 是 `tabicl-classifier-v2-20260212`，BSD-3 许可，HF 上没有 gated，可以匿名下载。
   - 整个过程没有输入任何凭据：环境里删除了所有 HF/TabPFN token，并关闭了浏览器登录。
2. **TabPFN v2 和 TabICL 的官方 CV（R251）**：只用每折 200 行带标签数据作为 context，不用测试集特征。分别在"全部 150 个特征"和"13 维 block"上跑。测试集预测用 refit 模式（在全部 250 行上拟合一次）。
3. **正则化 horseshoe 贝叶斯稀疏 logistic（R253、R257）**：用 NumPyro NUTS 采样。每折 500 次 warmup 加 500 个样本；全数据拟合用 4 条链，各 1000+1000。考察后验纳入和可识别性，并与 L1 logistic、block 模型比较。
4. **Dirichlet-process 变分 GMM（R254）**：在 20,000 行无标签数据上拟合。看有效分量数、标签无关的 BIC、与 S003 在边界行上的一致性。另外做了复刻模拟（参数 bootstrap），比较 GMM、DP-GMM、贝叶斯 QDA 的排序能力。
5. **贝叶斯 QDA（Normal-Inverse-Wishart，Student-t 后验预测）和 GP 分类器（R255）**：只用 250 个标签，回答"不用无标签数据，能否弥补 QDA 0.996 到 GMM 1.000 的差距"。R256 是对照实验，用来拆开"贝叶斯积分"和"收缩强度/目标"这两个因素。

### 关键数字（官方 CV 5×20，seed 2026，分割全部相同）

| 模型 | run_id | CV AUC ± sd(repeat) | CV log-loss | 用到测试特征？ |
|---|---|---|---|---|
| 贝叶斯 QDA，EB 先验，block | 20261022-070200_r255_bqda_eb_hv | **1.0000 ± 0.0000** | **0.0094** | 只用于 block 选列和标准化（无标签） |
| 同上，改成高斯 plug-in（无 Student-t） | 20261022-070536_r256_bqda_eb_plugin_hv | 1.0000 ± 0.0000 | 0.0095 | 同上 |
| 贝叶斯 QDA，block 和标准化全部只在训练折内计算 | 20261022-070202_r255_bqda_eb_trainonly | 0.9990 ± 0.0011 | 0.041 | **完全不用** |
| 经典 QDA reg 0.01 / 0.03 / 0.1(R003) / 0.3 | r256_qda_reg0.01_hv 等 | 0.9998 / 0.9989 / 0.9958 / 0.9832 | 0.024 / 0.046 / 0.088 / 0.178 | 同 block |
| TabICL，block | 20261022-074643_r251_tabicl_hv | 0.9989 ± 0.0008 | 0.047 | 同 block |
| TabICL，全部 150 个特征 | 20261022-074251_r251_tabicl_all | 0.9929 ± 0.0021 | 0.120 | 否 |
| TabPFN v2，block | 20261022-073935_r251_tabpfn2_hv | 0.9888 ± 0.0031 | 0.146 | 同 block |
| TabPFN v2，全部 150 个特征 | 20261022-073743_r251_tabpfn2_all | 0.9851 ± 0.0041 | 0.167 | 否 |
| GP 分类器（ARD RBF），block | 20261022-070207_r255_gpc_ard_hv | 0.9778 ± 0.0046 | 0.248 | 同 block |
| Horseshoe，150 线性项 + 91 个 block 二次项 | 20261022-073222_r253_hs_quadhv | 0.9673 ± 0.0054 | 0.248 | 只用于选列 |
| Horseshoe，全部 150 个特征 | 20261022-073018_r253_hs_all | 0.9261 ± 0.0070 | 0.347 | 否 |
| DP-GMM（直推），block | 20261022-071054_r254_dpgmm_hv | 1.0000 | 0.0171 | 是（直推） |
| 参照：S003 GMM K=2（R003） | 20261022-050708_r003_gmm2_hv | 1.0000 | 0.0171 | 是 |

### 结论
- **贝叶斯 QDA 在 CV 上和直推 GMM 打平**：只用 250 个标签，AUC 1.0000，log-loss 0.0094，比 GMM 的 0.0171 还低。与 QDA(reg 0.1) 的配对差为 +0.0042（split-SE 0.00025，corrected p 0.12）。即使完全不用测试集，也有 0.9990。
- **起作用的不是"贝叶斯积分"**：R256 显示，同一个先验换成 plug-in 后结果相同（0.0095）。真正的原因是 **R003 的 QDA reg 0.1 收缩过强**，reg 0.01 时已经有 0.9998。EB（type-II 最大似然，只用训练折）在全部 100 折里都选中"向 pooled 协方差收缩、约 4 个伪观测"。这个强度是自动选出来的，不靠 CV 调参。
- **在边界行上，GMM 仍然更好**：
  - 贝叶斯 QDA 与 S003 在 204 个边界行上只有 73.5% 判在同一侧（Spearman 0.55）。在非边界行上几乎完全一致（pseudo-AUC 0.99998）。
  - 复刻模拟中，在 GMM 的约 200 个边界行上，直推 GMM 的 AUC 是 0.825，DP-GMM 0.824，只用 250 个标签的贝叶斯 QDA 0.702，两者平均 0.795。总体测试 AUC 则几乎相同（0.9949 / 0.9948）。注意：这组数只来自 K=2 生成器的 3 个复刻（加上 DP-GMM 生成器共 6 个，后者是 0.830 / 0.701；`reports/tables/r254_replica_auc.csv`），方向已在 222 个复刻上得到确认（r404，`reports/tables/r404_boundary.csv`，`research/reports/r2_inductive.md`）：在真实贝叶斯后验落在 (0.1, 0.9) 的测试行（每个复刻约 185 行）上，S003 的 AUC 是 0.804，贝叶斯 QDA（r255）是 0.678，222/222 个复刻里 S003 都更好；按 r254 的定义（S003 分数在 (0.1, 0.9) 的行）是 0.814 vs 0.686。r936 也支持这一方向：222 个复刻里，贝叶斯 QDA（r255）的真实 AUC 每次都低于 S003（平均低 1.2e-4，`reports/tables/r936_cv_power.csv`）。
  - 所以无标签数据的价值正好体现在边界行上，而 CV 看不出这一点。S003 仍应作为主提交。
- **DP 先验没有带来新东西**：γ 从 0.001 到 100、截断取 10 或 20，有效分量数都正好是 2（权重 0.499 / 0.499）。plug-in BIC 为 343,920，与 K=2 的 343,911 基本相同；与 S003 在边界行上 99% 一致。这说明 K=2 的结构是稳健的，可以作为 S003 的冗余版本，没有必要另外提交。
- **基础模型**：
  - TabICL 的表现强于 TabPFN v2。不做任何特征选择、只用 150 个原始特征时，TabICL 得到 0.9929，是"不用无标签数据、不做选择"的模型里最好的。
  - 但两者都不如贝叶斯 QDA 或轻正则 QDA。它们在边界行上的排序与 GMM 基本无关（Spearman 0.19–0.38）。
  - 测试集上有 6–23% 的行分数落在 (0.1, 0.9) 区间，而 S003 只有 1%。
- **Horseshoe**：
  - 全特征版本的 0.9261 是全部 150 个特征上最好的线性模型（13 列块上 L2 logistic 是 0.938）。相对 L1 +0.079（split-SE 0.0025，p 0.0005，win 0.99），相对 R003 的 block L2（`r003_sup_logreg_hv`，C=1，0.8977）+0.028（p 0.03；`experiments/20261022-073018_r253_hs_all/paired_vs_references.csv`），说明 block 之外也有线性信号。（fix round 2：注明参照 run；调好的 block L2 r207 是 0.938，比 horseshoe 高。）
  - 但它远低于二次模型。加入 block 二次项后是 0.967，仍低于 QDA。
- **可识别性（R257）**：
  - 全数据 4 条链没有混合，β 的 R-hat 为 4.6。原因是 150 个特征里有 8 个（近似）精确线性相关：null space 涉及全部 13 个 block 列，以及 block 外的 6、17、52、57、61、142、144。不同链用了同一线性预测的不同表示，线性预测量 η 的 R-hat 最大只有 1.17。
  - 用 QR 列主元去掉共线列之后，R-hat 降到 1.003。明确被纳入的特征是 46、6、80、30、142、100（1-κ > 0.9，90% 区间不含 0）。其中 6 和 142 在 block 之外，但它们是 block 的线性组合。

### 对团队的意义
1. 最终提交仍以 S003（直推 GMM）为主。如果想要第二个"方法互补"的提交（D008），**贝叶斯 QDA EB（r255_bqda_eb_hv）** 是合适的候选：它是监督模型，标签只用于拟合，CV 1.0000，CV（逐折平均）log-loss 在 fm 这条线里最低（0.0094）。但自训练 QDA-LW（r209）的 OOF log-loss 更低（0.0071），两个数的口径不同。但复刻模拟表明它在边界行上弱于 GMM，所以预期私榜分数略低于 S003，主会话决定时应考虑这一点。另外，它的测试概率里有 1,493 行恰好等于 1.0，形成并列。提交时应改用 log-odds（`decision_function`）来避免并列；如果这些并列行里没有负例，就不影响 AUC。
2. 报告里"QDA 0.996 到 GMM 1.000 的差距"应改写为：**这个差距主要来自 QDA 的正则化选择**。用 EB 先验或轻正则时，监督 QDA 的 CV 也是 1.0。直推方法的优势只在边界行，证据来自复刻模拟，CV 无法检验。
3. 基础模型（TabPFN v2、TabICL）**建议只作为分析对比，不用于最终提交**，原因见下面的合规说明。

### 合规说明（基础模型与规则 4）
- 规则 4 原文："You may not use data from outside this competition to train your model"，并要求引用所有用到的方法。
- TabPFN v2 和 TabICLv2 由各自的作者用先验生成的数据预训练（Hollmann et al., Nature 2025；Qu et al., 2026，arXiv preprint arXiv:2602.11139），从未在本比赛的数据上训练。模型卡写明默认 checkpoint 与 `v2_default` 的内容完全相同。存档的两份模型卡都没有写这些 checkpoint 具体用了什么预训练数据，所以报告里不要写"只用合成数据"（verify_rules §1）。
- 不过，它们的权重是在本比赛之外的数据上训练出来的。严格解读规则 4 的话，"用外部数据训练的预训练模型"可能被视为违规。规则原文没有明确讨论预训练模型。
- 第二点是署名：TabPFN v2 的许可要求在分发或公开使用时注明 "Built with PriorLabs-TabPFN" 并附上许可。仅内部基准测试不需要署名，但写进报告就应当署名。TabPFN-2.5 及以后的版本需要登录并接受非商用许可，我们没有使用。
- **建议**：基础模型只出现在报告的 Analysis 里作为对照（并给出引用和署名），不作为最终提交。如果想用，请先问讲师。从性能看也没有用的必要：它们都不如我们自己的贝叶斯 QDA 或 GMM。

### 发现的基础设施问题（主会话需要修，我没有改 `colab/`）
1. **`colab/remote.py` 的 `prepare()` 没有 `return want`**：每次 `run` 同步完代码后都会在 `synced.get(...)` 处报 AttributeError。我的绕法是先 `remote.py up`（它不使用返回值），再 `run --no-sync`。代价是 meta.json 里的 `code_bundle` 记成 "unsynced (--no-sync)"。为此我在每个 run 的 config 里写了 `code_sha`（lib_fm.py 和脚本的 sha256），代码包本身也保存在 `logs/code_bundles/`。
2. **非默认账号（pro2）无法使用**：`_colab_prefix()` 里的 `$A`、`$C` 会被 wsl.exe 外层的 shell 提前展开成空串，报错为 `mkdir: cannot create directory ''`。我的绕法是在 PowerShell 里设 `$env:A`、`$env:C`，再设 `WSLENV=A/u:C/u`，只改进程环境变量，不改任何文件。
3. **会话忙时运行 `up` 会触发 self-heal**：它会误判会话失去响应并重建 VM，正在跑的作业随之丢失。我丢了一个作业：`J20261022-064754-pro2-rfm`（TabPFN 的 fold_mean 版本），它没有任何 run 进入档案，也没有登记在 compute_jobs 里。规则应该是：会话上有作业在跑时，不要对它执行 `up`，也不要在同一会话上并行启动作业。

### 用到的库/方法（请加入 `registry/references.md`）
- TabPFN：Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637, 319–326. doi:10.1038/s41586-024-08328-6。软件为 tabpfn 9.2.0（代码 Apache-2.0），权重为 TabPFN-2（Prior Labs License：Apache-2.0 加署名 "Built with PriorLabs-TabPFN"）。
- TabICL：Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2025). TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. *ICML 2025*；以及 Qu et al. (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv preprint arXiv:2602.11139。软件为 tabicl 2.2.0（BSD-3），checkpoint 为 tabicl-classifier-v2-20260212。
- PyTorch 2.14.1：Paszke et al. (2019), *NeurIPS*。huggingface_hub 2.2.0。
- NumPyro 0.22.0：Phan, Pradhan, Jankowiak (2019), arXiv:1912.11554；Bingham et al. (2019), Pyro, *JMLR* 20。JAX 0.10.2：Bradbury et al. (2018)。NUTS：Hoffman & Gelman (2014), *JMLR* 15。
- Horseshoe：Carvalho, Polson, Scott (2010), *Biometrika* 97；正则化 horseshoe 与 τ0 的选择：Piironen & Vehtari (2017), *Electronic Journal of Statistics* 11。R-hat / ESS：Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), *Bayesian Analysis* 16。
- 变分 DP-GMM：sklearn `BayesianGaussianMixture`；Blei & Jordan (2006), Variational inference for Dirichlet process mixtures, *Bayesian Analysis* 1；Bishop (2006), *PRML* 第 10.2 节。
- NIW 共轭模型、Student-t 后验预测、边际似然：Murphy, K. P. (2007), Conjugate Bayesian analysis of the Gaussian distribution（技术报告）；Gelman et al. (2013), *Bayesian Data Analysis* 第 3 版第 3.6 节。Empirical Bayes / type-II ML：Bishop (2006) 第 3.5 节。经典 QDA 与 RDA：Friedman (1989), *JASA* 84。
- GP 分类（Laplace）：Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning* 第 3 章；sklearn `GaussianProcessClassifier`。
- QR 列主元：Golub & Van Loan (2013), *Matrix Computations* 第 4 版第 5.4 节（scipy.linalg.qr）。

---

## Text for the report

**Pretrained and Bayesian alternatives.** We compared the transductive Gaussian mixture (S003) with supervised alternatives that use only the 250 labels, all under the official protocol (repeated stratified 5-fold × 20, identical splits; Table X). Two pretrained tabular foundation models, TabPFN v2 (Hollmann et al., 2025) and TabICL v2 (Qu et al., 2026), both pretrained by their authors on prior-generated data and never trained on this competition's data, were used as in-context learners on each fold's 200 training rows. On the raw 150 features, without any feature selection, they reach 0.985 and 0.993 AUC. That is far above L1 logistic regression (0.847) and a regularised-horseshoe logistic regression sampled with NUTS (0.926, the best linear model on all 150 features; on the 13-column block L2 logistic reaches 0.938). However, they remain below a Bayesian QDA. That model gives each class a Normal-Inverse-Wishart prior and scores rows with the exact Student-t posterior predictive, and its prior hyperparameters are chosen per fold by maximising the closed-form marginal likelihood. On the high-variance block it reaches CV AUC 1.0000 on all 100 folds, with a lower CV log-loss than the GMM (0.009 vs 0.017), and 0.999 when even the block rule and scaling are computed only on the training fold. Control runs show that the Student-t predictive adds nothing (the plug-in Gaussian gives the same result). The earlier QDA (0.996) was simply over-regularised towards the identity: with reg_param 0.01 it scores 0.9998. The empirical-Bayes prior picks the right amount of shrinkage towards the pooled covariance automatically, without tuning on validation folds. A Dirichlet-process variational mixture on the 20,000 unlabeled rows keeps exactly two effective components for every concentration from 0.001 to 100, and it agrees with S003 on 99% of the boundary rows. This supports the K = 2 structure.

**Where unlabeled data still matter.** Because cross-validated AUC is saturated, we used a parametric bootstrap: 20,000 rows were drawn from the fitted mixtures, and only the first 250 labels were revealed. Overall test AUC is the same for the transductive GMM and the supervised Bayesian QDA (0.9949 vs 0.9948). On the roughly 200 rows near the class boundary, however, the GMM ranks far better (AUC 0.825 vs 0.702), and averaging the two does not help (0.795). These replica numbers come from only 3 replicas of the K = 2 generator (6 with a DP-mixture generator, which gives 0.830 vs 0.701). A study on 222 replicas confirms the direction (r404): on the test rows whose true posterior lies in (0.1, 0.9), about 185 per replica, the GMM reaches 0.804 and the Bayesian QDA 0.678, and the GMM is better in all 222 replicas (0.814 vs 0.686 when the boundary rows are chosen by S003's scores, as above). In r936 (222 replicas) the Bayesian QDA's overall true AUC was also below S003's in every replica, by 1.2 × 10⁻⁴ on average. On the real test set, the Bayesian QDA agrees with S003 on only 74% of the 204 boundary rows. We therefore keep the transductive GMM as the primary submission. The supervised Bayesian QDA is a methodologically independent check: it falls on the same side of 0.5 as S003 for 99.6% of all test rows, and its pseudo-AUC against S003's confident labels is 0.99998. A side finding concerns feature identification. The 150 features contain 8 near-exact linear dependencies (seven non-block features are linear combinations of block features), so the horseshoe coefficients are not identified. Chains settled on different but equivalent sparse representations (R-hat 4.6 on the coefficients, at most 1.17 on the linear predictor). After removing collinear columns by pivoted QR, the chains mix (R-hat 1.003) and six terms are clearly included (46, 6, 80, 30, 142, 100).

**Rule compliance.** The foundation models were downloaded anonymously under their open licences (TabPFN-2: Prior Labs License, Apache-2.0 with attribution, "Built with PriorLabs-TabPFN"; TabICL: BSD-3). The default TabPFN-2.5+ weights require a login and licence acceptance, so they were not used. The models were never trained on this competition's data. Their weights were, however, fitted by their authors outside this competition, on prior-generated data that the archived model cards do not document for these checkpoints, and rule 4 does not address pretrained models explicitly. We therefore report them only as an analysed comparison and do not use them in a submission.

| Model (250 labels unless noted) | CV AUC (± sd over repeats) | CV log-loss | Boundary-row side agreement with S003 |
|---|---|---|---|
| Transductive GMM K = 2 (S003; uses 20k unlabeled rows) | 1.0000 ± 0.0000 | 0.017 | 100% |
| DP variational GMM (20k unlabeled rows), 2 effective components | 1.0000 ± 0.0000 | 0.017 | 99% |
| Bayesian QDA, empirical-Bayes NIW prior (block) | 1.0000 ± 0.0000 | 0.009 | 74% |
| Bayesian QDA, block and scaling from the training fold only | 0.9990 ± 0.0011 | 0.041 | 63% |
| QDA, reg_param 0.01 / 0.1 / 0.3 (block) | 0.9998 / 0.9958 / 0.9832 | 0.024 / 0.088 / 0.178 | 58% / 57% / 54% |
| TabICL v2 (block / all 150 features) | 0.9989 / 0.9929 | 0.047 / 0.120 | 57% / 51% |
| TabPFN v2 (block / all 150 features) | 0.9888 / 0.9851 | 0.146 / 0.167 | 49% / 51% |
| GP classifier, ARD RBF (block) | 0.9778 ± 0.0046 | 0.248 | 48% |
| Regularised-horseshoe logistic (150 linear + 91 block quadratic terms / 150 linear) | 0.9673 / 0.9261 | 0.248 / 0.347 | 51% / 54% |
| L1 logistic, all 150 features (R003) | 0.8469 ± 0.0120 | — | — |

Run ids: r255_bqda_eb_hv 20261022-070200, r255_bqda_eb_trainonly 20261022-070202, r256_qda_reg* 20261022-0705(43/46), r251_tabicl_* 20261022-0742(51)/0746(43), r251_tabpfn2_* 20261022-0737(43)/0739(35), r255_gpc 20261022-070207, r253_hs_* 20261022-0730(18)/0732(22), r254_dpgmm_hv 20261022-071054, replica table `reports/tables/r254_replica_auc_summary.csv`, licence check 20261022-063927_r250. Boundary rows are the 204 test rows with an S003 score in (0.1, 0.9). TabPFN/TabICL test predictions come from a refit on all 250 rows.
