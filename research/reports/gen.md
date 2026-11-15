# gen：反推数据生成器 + 复刻数据模拟（R150–R156）

作者：AI 代理 "gen"（Claude Code），2026-10-22。计算全部在 Colab（账号 pro1，会话 rgen，high-mem 8 核）。
代码：`research/lib_gen_replica.py`（复刻库，文件开头有 API 文档）、`research/r150_*.py` – `r156_*.py`。
表格：`reports/tables/r150_*` – `r156_*`；图：`reports/figures/r150_taxonomy.png`、`r151_validation.png`、`r152_calibration.png`。

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要和 "Text for the report"）。依据：`research/reports/verify_gen.md`（R920–R922）、`research/reports/verify.md` §2 A4、B1、B5–B7。

1. **flip_y**（摘要表 flip_y 行；Table G1；Text for the report）。Old: "约 0.0025–0.005（后验均值 0.0044，P(>0) = 0.92）"、"S003 的 public 0.99923 在 flip_y = 0 时几乎不可能"、"flip_y ≈ 0.004 (P(flip_y > 0) = 0.92)" → New: MLE 0.003, likelihood support 0–0.015 (≥ 1/8 of the maximum on [0, 0.0147]); sklearn's default 0.01 has likelihood ratio 0.37; flip_y = 0 is disfavoured about 5:1 but not excluded (LR 0.20; 0.03–0.84 when the posterior temperature changes by ±20%). Source: R922 `20261022-092146_r922_label_noise_private` (notes.md Conclusion), `reports/tables/r922_likelihood_named.md`. The old precision came from R156's 4-point prior, which has no mass above 0.01.
2. **S003 private AUC**（摘要 §3 最后一条；Text for the report）。Old: "S003 的 private 预期约 0.9977（只用 public 时的 90% 区间是 0.9944–0.9991）。private 比 public 低大约 0.0015"、"an expected private AUC of about 0.9977" → New: about 0.997 under a flat prior on flip_y (median 0.9973, mean 0.9968, 90% interval 0.992–0.9996); private about 0.002 below public. Source: `reports/tables/r922_posterior_by_prior.csv` (row `uniform_0_0.1`). The same table reproduces 0.9978 under R156's grid prior (row `gen_grid4`), so the R156 computation was right and only the prior differs. This also answers the first open question at the end of this file.
3. **Gap to Bayes**（摘要 §3 (ii)；Text for the report）。Old: "离 Bayes 只差 3e-7"、"within 3·10⁻⁷ of the Bayes AUC" → New: about 2.6·10⁻⁷ below the Bayes AUC on replicas fitted to our data. Source: R921 `20261022-094951_r921_bayes_regret_replicas`, `reports/tables/r921_summary.csv` (parametric group: expected-AUC regret 2.57e-7 ± 0.02e-7 over 1000 replicas, positive in 100%). R153's 3.1e-7 (100 replicas) agrees within its SE; Table G2 is unchanged because it reports R153.
4. **class_sep**（摘要表；Text for the report；Table G1）。Old: "0.71（95% 0.61–0.81）" → New: 0.70 (about 0.63–0.78). Source: R920 `20261022-100551_r920_generator_alternatives` (summary.json: exact-likelihood MLE 0.703, LR 95% [0.639, 0.760]; replica coverage 21/24, so the honest width is about ±0.075), `reports/tables/r920_class_sep_likelihood.csv`. The old interval was conservative, not wrong.
5. **(fix round 2, 2026-10-22) Fit of the 8 dependencies**（摘要表 n_redundant 行；Text for the report 第 1 段）。Old: "R² ≥ 1−4e-14"、"R² ≥ 1 − 4·10⁻¹⁴" → New: "1 − R² ≈ 4e-14"、"1 − R² ≈ 4·10⁻¹⁴". Reason: the archived worst case is 1 − R² = 4.07e-14, marginally above the stated bound. Source: R440 claim G8 (`reports/tables/r440_core_numbers.csv`, run `20261022-184020_r440_core_numbers`; `research/reports/r2_audit.md` §5 item 6); `20261022-064606_r150_taxonomy` summary.json `dependency_r2_min` 0.9999999999999593.

## 中文摘要（给队友）

**1. 数据就是 sklearn 的 `make_classification`，外加逐列仿射变换。** 我只用 20,000 行特征（不看标签），找到了全部结构
（R150，`20261022-064606_r150_taxonomy`）：

| 参数 | 估计值 | 证据 |
|---|---|---|
| n_informative | **12**：6, 15, 30, 46, 52, 61, 64, 80, 100, 112, 142, 144 | 20 个"有结构"的列秩为 12；在全部 125,970 个 12 列子集里，按类内相关矩阵的 log det 选出的基明显最好（领先 1.09 nats） |
| n_redundant | **8**：8, 17, 26, 49, 57, 67, 77, 98 | 8 个精确线性组合，1 − R² ≈ 4e-14，最大残差 4.8e-6（就是 6 位小数的舍入误差）；完整公式见 `r150_dependencies.csv` |
| n_repeated | 0 | 没有 \|corr\| ≈ 1 的列对 |
| 噪声列 | 130 个，相互独立的高斯列 | 噪声 std ~ U(0.5, 3)（KS p = 0.26），均值 ~ U(−5, 5)（KS p = 0.87），两者无关 → 先 `shift=0, scale=1`，再逐列 x·a + b |
| n_clusters_per_class | 1 | BIC：总体最优 K = 2，每个类内部最优 K = 1 |
| weights | None（平衡） | 10,016 : 9,984 |
| hypercube | True | 8 个信息列的类均值差**恰好为 0**（\|z\| ≤ 2.08），另 4 个（46, 80, 100, 112）\|z\| ≥ 47 → 两个中心只在 4 个坐标上不同 |
| class_sep | **0.70**（约 0.63–0.78；R920） | 4 个活跃坐标的未缩放均值差 / 2；R152 在 176 个已知真值的复刻上校准过（估计量无偏，sd 0.05）；R920 的精确似然给出 MLE 0.703 |
| flip_y | **MLE 0.003**，似然支持 0–0.015（sklearn 默认值 0.01 的似然比 0.37） | 训练集里没有标签与高置信后验矛盾（R150、R922）；S003 的 public 0.99923 让 flip_y = 0 的可能性降低约 5 倍，但不能排除（似然比 0.20，后验温度变化 ±20% 时 0.03–0.84；R922） |

另外：内部参数 A_k、B 的分布也和 sklearn 一致（隐含 B ∈ [−1.08, 1.11]，KS p = 0.73）。R152 证明这套**无标签识别**方法在复刻数据上 100% 找对信息列和冗余列。
**13 列高方差块已经张满全部 12 维信息空间**，所以块外那 7 个结构列（6, 17, 52, 57, 61, 142, 144）不带任何新信息。

**2. 复刻库 `research/lib_gen_replica.py`（给全队用）。** 两种复刻：(a) `sample_parametric`：用真实数据拟合的两类高斯 + 真实的冗余映射和噪声分布（列位置、尺度都和真实数据一样）；(b) `sample_make_classification`：直接调 sklearn（与 sklearn 输出逐位相同），每个种子重新抽 A_k、B 和尺度。每个复刻都有 250 + 19,750 行、已知测试标签、**真实 Bayes 后验**、一个 395 行的"public"子集。
R151（`20261022-072608_r151_replica_validation`）：参数复刻在 std 曲线、相关矩阵特征值、块大小/秩、BIC、边界行比例（0.96% vs 真实 0.95%）上都和真实数据一致；R003 的模型排序在 97% 的复刻里完全一样，CV AUC 也接近（L1 0.855 vs 0.849，LDA 0.887 vs 0.875，QDA 0.992 vs 0.995，L2-logreg 0.903 vs 0.895，GMM 0.99997 vs 1.0）。

**3. 用复刻回答真实数据回答不了的问题**（R153，参数复刻 100 个 `20261022-073058_r153_param`；sklearn 复刻 100 个 `20261022-075419_r153_mc_cs0.71_h4`）：

- **(i) Bayes 上限**：没有标签噪声时 private AUC = 0.99995；有噪声时约等于 1 − flip_y/2（0.0025 → 0.9987，0.005 → 0.9974，0.01 → 0.9949）。**决定 private 分数的是标签噪声，不是模型。**
- **(ii) 我们的方法**：gmm2_hv（= S003）比 Bayes 低约 3e-7（R921 在 1000 个按真实数据拟合的复刻上测得 2.6e-7），已经是最优。tol 收紧、PCA 基、全部结构列、真实信息列这几个变体和它的差距都在 1e-8 以内。gmm2_all 和 gmm4_hv 每次都略差（约 −3.5e-5），QDA 0.993，L1 0.86。CV（5×4）对所有直推式变体都是 0.99997，**根本分不出好坏**。
- **(iii) Public LB 噪声**：flip_y = 0 时 public AUC 的 sd 只有 0.00009（52% 的抽样正好等于 1.0）；flip_y = 0.0025–0.01 时 sd 是 0.002–0.004，**比直推式变体之间的真实差距（≤ 3.5e-5）大 50–100 倍**。K=2 变体在 public 上 60–100% 打平；gmm2_hv 和 gmm2_all 在 public 上 10–42% 打平、12–47% 排反。**所以 public LB 不能用来挑直推式变体。**
- **(iv) 加入块外的信息列**：真实数据（以及 100% 的参数复刻）里没有任何作用（差 ≤ 1.8e-7）。但在 sklearn 复刻里，有 5% 的情况高方差块漏掉了一个信息方向，这时 gmm2_hv 最多损失 0.007，而"全部结构列 + PCA 到秩"能恢复到 Bayes。**结论：结构列规则更稳健；对我们的数据不会改变结果。**
- **S003 的 private 预期**（R156 `20261022-081314_r156_flip_posterior`；R922 `20261022-092146_r922_label_noise_private`）：R156 在 4 点网格（0 / 0.0025 / 0.005 / 0.01，0.01 以上没有先验质量）上得到 0.9977，这个精度来自先验。R922 改用 [0, 0.1] 上的均匀先验，同样结合训练标签和 public 0.99923：**S003 的 private 约 0.997**（中位数 0.9973，90% 区间 0.992–0.9996）。private 比 public 低约 0.002 是正常的，不代表模型有问题。

**4. 候选提交**（R155，官方 5×20 CV）：`20261022-080453_r155_gmm2_inf_tight`（12 个信息列，EM tol 1e-8）和 `20261022-080505_r155_gmm2_hv_tight`。两者 CV 都是 1.0000，期望 private AUC 和 S003 相同（差 < 1e-8），**不值得占一个提交名额**。最多当作第二个最终选择，和 S003 互为备份。

**对团队的建议**：
1. 最终提交保留 S003 类模型（gmm2_hv 或同等变体）。已经没有可以挖的提升空间了：剩下的差距都来自标签噪声。
2. 不要用 public LB 比较直推式变体；第二个最终选择最好选方法上互补、但同样接近 Bayes 的模型（例如结构列 GMM）。
3. 报告里可以直接引用"生成器已识别 + Bayes 上限"这条证据链。

**负面 / 中性结果**：EM 收紧 tol、PCA 基、块外列、真实信息列都没有改进（R153、R155）；K=4 和全部 150 列的 GMM 稍差；R155 summary 里的 "boundary" 三个指标定义有误（用的是平滑后的分数，不是后验），正确的比较见 R150 的 posterior_agreement；第一次 R152 运行失败（loky 无法 pickle scipy 的 `expit` ufunc），已标记 INVALID 并重跑。

**合规**：只用了比赛数据（train + test 特征，标签只来自训练集）。我**没有**尝试暴力搜索 `make_classification` 的 random_state 来复现隐藏的测试标签：那相当于从比赛数据以外获取答案，违反规则精神。在 R156 里，S003 的 public 分数被当作关于 flip_y 的一个观测值来用，没有用来选模型。

## Text for the report

**Reverse-engineering the generator.** Using only the 20,000 unlabeled feature rows, we found that the data follow scikit-learn's `make_classification` with a per-column affine rescaling (runs R150, R152; Table G1). Twenty columns are linearly structured with rank 12. Eight of them are exact linear combinations of the other twelve (1 − R² ≈ 4·10⁻¹⁴; the residuals are at the level of the 6-decimal rounding). The remaining 130 columns are independent Gaussian noise whose standard deviations and means are consistent with U(0.5, 3) and U(−5, 5). The 12 informative columns were identified label-free: among all 125,970 candidate 12-column bases, we chose the one that maximizes the log-determinant of the within-class correlation matrix. Each class is a single Gaussian (BIC). Eight informative coordinates have exactly zero class-mean difference, which is the signature of `hypercube=True`, and the two class centroids differ in the remaining four. The resulting estimate is class_sep ≈ 0.70 (about 0.63–0.78; exact likelihood, R920). On 176 replicas with known truth, this label-free procedure recovered the informative and redundant sets in 100% of cases and gave an unbiased class_sep estimate. In the fitted model, the classes differ mainly in covariance (R154). The best linear rule reaches a population AUC of only 0.947, whereas the quadratic Bayes rule reaches 0.99995. This explains why linear models stall at 0.85–0.90 while QDA and mixture models do not.

**Replica simulations.** We built a replica generator (`research/lib_gen_replica.py`) in two variants. The parametric variant resamples from the fitted per-class Gaussians, the exact redundancy map and the noise distribution. The second variant calls `make_classification` itself, with bit-identical internals. Each replica is a full competition with known test labels and the true Bayes posterior. Parametric replicas reproduce the real data's summary statistics and the model ranking of R003 (identical order in 97% of 30 replicas; R151). On 100 parametric replicas (R153; Table G2), the Bayes-optimal private AUC is 0.99995 without label noise and about 1 − flip_y/2 with it. Our transductive two-component mixture on the high-variance block (S003) is about 2.6·10⁻⁷ below the Bayes AUC on replicas fitted to our data (R921, 1000 replicas). Its variants (converged EM, PCA basis, all structured columns, true informative columns) differ from it by less than 10⁻⁸, while a 150-dimensional mixture or K = 4 loses about 3.5·10⁻⁵. The public leaderboard has 395 rows. With label noise, its standard deviation is 0.002–0.004, two orders of magnitude larger than the true differences between transductive variants, so it cannot rank them. Combining the training labels with S003's public score gives a maximum-likelihood flip_y of 0.003, with likelihood support from 0 to 0.015. The scikit-learn default of 0.01 remains plausible, and flip_y = 0 is disfavoured about 5:1 but not excluded (R922). Under a flat prior on flip_y, the expected private AUC of S003 is about 0.997 (median 0.9973, 90% interval 0.992–0.9996), about 0.002 below its public score. The remaining gap to 1 is irreducible label noise, not model error.

Table G1 – Generator parameters recovered from the unlabeled features (R150, R152; class_sep R920; flip_y from the labels, R922).

| Parameter | Estimate | Evidence |
|---|---|---|
| n_informative / n_redundant / n_repeated | 12 / 8 / 0 | rank 12 of 20 structured columns; 8 exact dependencies |
| noise columns | 130, std ~ U(0.5, 3), mean ~ U(−5, 5) | KS p = 0.26 / 0.87; max pairwise \|r\| = 0.029 |
| n_clusters_per_class, weights | 1, balanced | BIC K = 2 overall, K = 1 within class; 10,016 vs 9,984 |
| hypercube, class_sep | True, 0.70 (≈ 0.63–0.78; R920) | 8 informative columns with zero mean difference (\|z\| ≤ 2.1), 4 with \|z\| ≥ 47 |
| flip_y | MLE 0.003; likelihood support 0–0.015 | 0 confident train contradictions; public score of S003 (R922) |

Table G2 – Private-test AUC on 100 parametric replicas (19,355 rows; R153). Mean (sd); regret = Bayes − model, paired.

| Procedure | flip_y = 0 | flip_y = 0.005 | regret (flip 0) | public-LB sd (flip 0.005) |
|---|---|---|---|---|
| Bayes (true posterior) | 0.99995 (0.00001) | 0.99742 (0.00040) | 0 | 0.0029 |
| GMM K=2, block (S003) | 0.99995 (0.00001) | 0.99740 (0.00040) | 3·10⁻⁷ | 0.0028 |
| GMM K=2, all structured columns | 0.99995 (0.00001) | 0.99740 (0.00039) | 3·10⁻⁷ | 0.0028 |
| GMM K=4, block | 0.99992 (0.00005) | 0.99737 (0.00040) | 3.5·10⁻⁵ | 0.0029 |
| GMM K=2, all 150 columns | 0.99992 (0.00002) | 0.99737 (0.00040) | 3.4·10⁻⁵ | 0.0028 |
| QDA (reg 0.1), block, 250 labels | 0.99314 (0.00091) | 0.99052 (0.00106) | 0.0068 | 0.0038 |
| L1 logistic, all columns | 0.8602 (0.0180) | 0.8574 (0.0180) | 0.140 | – |

## Runs, jobs and files

| Run | Kind | What |
|---|---|---|
| 20261022-064606_r150_taxonomy | analysis | feature taxonomy, 8 dependencies, make_classification test, label noise, posterior agreement |
| 20261022-065046_r152_calibration | analysis | FAILED (loky pickling of `expit`), INVALID, superseded |
| 20261022-065325_r152_calibration | analysis | identification accuracy and class_sep calibration on 176 replicas |
| 20261022-065605_r154_class_structure | analysis | per-class Gaussians, population AUCs (Bayes, best linear, mean-only, covariance-only) |
| 20261022-072608_r151_replica_validation | analysis | replicas vs real: summary statistics and R003 ranking |
| 20261022-073058_r153_param | analysis | 100 parametric replicas × 4 flip levels: Bayes AUC, procedures, LB noise, swaps, outside-block columns, ABC |
| 20261022-075419_r153_mc_cs0.71_h4 | analysis | the same on 100 make_classification replicas |
| 20261022-080453_r155_gmm2_inf_tight | experiment | official CV 1.0000, test_pred available (equivalent to S003) |
| 20261022-080505_r155_gmm2_hv_tight | experiment | official CV 1.0000, test_pred available (equivalent to S003) |
| 20261022-081314_r156_flip_posterior | analysis | posterior over flip_y; expected private AUC of S003 |

Colab jobs (pro1 / rgen): J20261022-064744, -065227 (failed), -065506, -065746 (fetched later after a hung session), -072746, -073009, -075559, -080630, -080723.

## Methods and libraries used (for references.md)

- scikit-learn 1.9.1 (Pedregosa et al., 2011): `make_classification` (design from Guyon, 2003, NIPS 2003 feature-selection challenge), `GaussianMixture` (EM: Dempster, Laird & Rubin, 1977), LDA/QDA, logistic regression; NumPy, SciPy (KS test, D'Agostino normality test), pandas, joblib.
- BIC (Schwarz, 1978); Mardia's multivariate skewness and kurtosis (Mardia, 1970); parametric bootstrap (Efron & Tibshirani, 1993); approximate Bayesian computation (rejection ABC: Pritchard et al., 1999; Beaumont et al., 2002); Hanley & McNeil (1982) for AUC noise context.

## Open questions

- The flip_y posterior uses a coarse four-point grid with a uniform prior. Values between 0.005 and 0.01, or flip_y > 0.01, were not simulated; a finer grid and more replicas would tighten the S003 private predictive.
- The parametric replica fixes the fitted Gaussian parameters (plug-in). Parameter uncertainty is not propagated, although with 20,000 rows it is small.
- The real data's within-class Mardia skewness is lower than in the Gaussian replicas (1.10 vs 1.13–1.40). This is probably a truncation artefact of the statistic and was not investigated further.
