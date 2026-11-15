# mix: boundary-row ranking of the transductive GMM (r100–r106)

Author: AI (Claude Code), label `mix`, Colab pro1 / session `rmix` (CPU High-RAM, 8 cores). Date: 2026-10-22 (JST).
Code: `research/lib_mix.py`, `research/r100_*.py` … `research/r106_*.py`. Tables: `reports/tables/r100_*` … `r106_*`.

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要和 "Text for the report"）。依据：`research/reports/verify_gen.md` §3（R922）、`research/reports/verify.md` §2 B6、B7、B9。mix 关于模型的结论不变；更正后"边界行的改进在 private 上看不出来"这一点反而更强。

1. **Private-LB spread**（摘要第 1 段；"Where the remaining public-LB error comes from"）。Old: "private LB 的 SE 约 1e-4"、"the private-LB standard error (~10⁻⁴)" → New: sd about 1e-5 with clean labels and about 3–6e-4 at plausible noise rates. Source: R922 `20261022-092146_r922_label_noise_private`, `reports/tables/r922_private_by_flip.csv` (sd of S003's private AUC: 1.1e-5 at flip_y 0; 3.1e-4 at the MLE 0.003; 3.9e-4 at 0.005; 5.5e-4 at 0.01; 6.6e-4 at 0.015).
2. **Orders of magnitude**（Text for the report）。Old: "three orders of magnitude below" → New: "three to four orders of magnitude below the private-LB sd at plausible noise rates, and about two orders below it even with clean labels". Note: the synthesis wrote only "three to four orders"; that holds at plausible noise (1.7e-7 against 3–6e-4), but with clean labels the ratio is about 1:65 (1.7e-7 against 1.1e-5), so both cases are stated.
3. **Wrong-label rate**（摘要第 5 点；Text for the report）。Old: "错标率的后验均值是 0.21%（0.05–0.4%）"、"about 0.2 % (0.05–0.4 %)" → New: about 0.31% (90% interval 0.05–0.76%) under a flat prior on flip_y. Source: `reports/tables/r922_posterior_by_prior.csv`, row `uniform_0_0.1` (posterior mean flip_y 0.0063, 90% [0.001, 0.01525]; wrong-label rate = flip_y / 2). The synthesis and verify_gen print the upper end as 0.77%; the archived q95 0.01525 / 2 gives 0.76%, which is used here. The old interval was narrower because R106's grid is denser at small flip_y and its ±4e-4 window inflates flip_y = 0 (verify_gen §3).
4. **S003 private AUC**（摘要第 5 点；Text for the report）。Old: "S003 的期望 private AUC ≈ 0.998"、"about 0.998" → New: about 0.997 (90% interval 0.992–0.9996). Source: same table and row (predictive median 0.9973, mean 0.9968, q05 0.9923, q95 0.9996).

## 中文摘要（给队友）

**结论先说：没有一个候选能在 private LB 上可检测地超过 S003。** S003 在复刻数据上离贝叶斯上限只差 1.7e-7 AUC，而 private LB 的 sd 在干净标签时约 1e-5，在合理的噪声水平下约 3–6e-4（R922）。所以 S003 已经到顶，剩下的误差来自标签噪声，不来自边界样本的排序。建议最终提交保留 S003，别再为“边界样本排序”花提交次数。

做了什么、看到了什么：

1. **无标签特征发现**（R100，`20261022-063714_r100_labelfree_feature_discovery`）。在 20,000 行上，只有 20 个特征和别的特征相关（|r| > 0.05，n = 20,000 时 SE ≈ 0.007），其余 130 个与任何特征的 |r| 都不超过 0.029，是独立噪声。这 20 个特征只张成 12 维空间，有 8 个精确线性关系（8 个零特征值）。块外那 7 个相关特征（6, 17, 52, 57, 61, 142, 144）都是块内特征的精确线性组合（R² = 1.0000）。用块 GMM 的后验做条件似然比检验（df = 14，BIC 门槛），没有找到任何新的类相关特征，迭代第 0 步就停了。所以 13 个高方差特征的块**已经张成全部信息子空间**。两个分量都非常接近高斯（Mahalanobis KS p = 0.73 / 0.36，偏度和超峰度都 < 0.1），BIC 选 K = 2。结构上很像 `make_classification(n_informative=12, n_redundant=8, n_clusters_per_class=1)` 再加逐特征缩放（这是推测）。监督的折内选择（每折按标签取边际 LRT 的 top-13）与块平均只重合 77%：它总会选上冗余的 17，又漏掉只在联合时才有信息的 15、26、30。训练集里唯一“标错一侧”的第 9 行后验是 0.79，属于边界样本，不是标错的标签。
2. **按 BIC 选模型**（R101 `20261022-064201_r101_bic_model_choice`，R104 `20261022-072159_r104_reg_sweep`）。K = 2 full 最好：K = 3 的 BIC 高 851，tied、diag、spherical 都高 11.8 万以上。在同一组 142 维变量上比较联合模型，“块混合 × 130 个独立高斯”胜过 142 维混合（R003 的 gmm2_all 在无标签层面就被否定了）。K = 2 的 30 次随机重启都收敛到同一个最优解（log-lik 极差 1e-9），所以重启集成没有作用。**正则化不能忽略**：分量协方差最小特征值约 1.1e-4，S003 用的 reg_covar 1e-6 约占它的 1%。reg → 0 时排序在 reg ≤ 1e-8 处收敛，这时 S003 的 204 个边界行 Kendall 为 0.9959，期望 AUC 不变。
3. **官方 CV 候选**（R102 共 13 个，R105 共 3 个，全部有 test_pred）。CV AUC 都是 1.0000，只有 tied 0.731、diag 0.851，监督选特征 0.99997。CV log-loss 主要反映校准，不反映排序：rate 映射会把分数压缩到 [0.004, 0.98]，所以 log-loss 是 0.0171；直接输出后验时是 0.0077。
4. **参数自助法 / 复刻**（R103 `20261022-073702_r103_replica_bootstrap`）。用拟合好的 K = 2 生成器各做 40 个复刻，每个 20,000 行、标签已知，在每个复刻上把每个流程完整重跑一遍。干净标签（G0）下：贝叶斯 test AUC 为 0.999951，S003 流程差 1.7e-7。各流程与 S003 的配对差（期望 AUC）：reg → 0 MLE +2.3e-9 ± 0.8e-9（78% 的复刻更好，是唯一方向一致的改进，但小到没有意义）；半监督 EM +1.6e-10；polish、发现、种子集成约 0；bagging −2.4e-8；自训练 QDA −2.1e-8；每类 2 分量 −3.7e-6；K = 4 −3.4e-5；142 维 GMM −3.4e-5；监督选特征 −3.5e-4；伪标签二次 logistic −4.5e-4；监督 QDA −6.7e-3；tied −0.23。另一个现象：S003 与贝叶斯排序在边界行上的 Kendall 只有 0.934，但这几乎不损失 AUC，因为约 186 个边界行接近抛硬币。
5. **标签噪声**（R103 公开榜模拟；R106 `20261022-080900_r106_label_noise_rate`）。干净标签下，公开榜 ≤ 0.99923 的概率只有 0.001。若按 sklearn 默认 flip_y = 0.01（即 0.5% 标签错误），这个概率是 0.83。把“公开榜 0.99923”和“训练集里没有高置信的错标行”两条一起看，在 flip_y 的均匀先验下，错标率约 0.31%（90% 区间 0.05–0.76%；R922）。由此推出 S003 的 private AUC 约 **0.997**（90% 区间 0.992–0.9996；干净标签时是 0.99995，flip 0.01 时是 0.9949），所有接近贝叶斯的候选都一样。
6. **负面结果**（都已归档）：特征发现没有新特征；半监督 EM、种子集成、bagging、自训练 QDA 都没有收益；伪标签二次 logistic、监督选特征、tied、diag、K = 4、全部 150 个特征的 GMM 都更差。

最多 3 个候选：**没有一个能在 private LB 上超过 S003。** `20261022-074857_r105_gmm2_block_mle`（reg → 0 的 MLE）是 S003 在理论上更干净的等价版本（+2e-9），可以作为报告里的“最终模型形式”，但单独占一个提交名额不值得。

基础设施问题：`colab/remote.py` 的 `prepare()` 没有 `return`，`run` 在同步之后会报 `'NoneType' object has no attribute 'get'`。我没有改 colab/，绕法是先 `remote.py up`（同步），再 `run --no-sync`，所以 run 的 meta.json 里 `code_bundle` 写的是 "unsynced"。每个作业实际用的代码包（`logs/code_bundles/<hash>.tar.gz`）如下：R100 b229bfbf58b5900a，R101 045a290b2df3f98b，R102 b70b4a55376f9278，R104 baf2b3063f48ac6e，R103 c43a5c05df2c1fa4，R105 6c9f6f1cef4b86f1，R106 158773d48d458382（都是作业前最后一次 `up` 生成的包；我的文件已和包内内容核对过）。日志流好几次卡住（作业在 VM 上已经跑完），用 `remote.py fetch` 取回。us-east 的会话 `m-hm-kkb-use1d1` 反复 exec 超时，`down` 后重建为 `use1d2`。还有一次 r104 作业失败（exit 2）：上传中断后，VM 上没有新文件。

## Text for the report (English)

**Is there anything left to gain on the boundary rows?** The transductive Gaussian mixture (S003) is saturated in cross-validation (AUC 1.0000 in all 100 folds), so we judged refinements by evidence that does not rely on CV AUC: label-free likelihood (BIC) on the 20,000 train+test rows, and a parametric bootstrap. Label-free analysis shows that only 20 of the 150 features are correlated with anything. They span a 12-dimensional space with 8 exact linear redundancies, and the 13 high-variance features already span all of it. A conditional likelihood-ratio test, with soft labels from the mixture, finds no further class-dependent feature. Each class is one Gaussian (Mahalanobis-distance KS p = 0.73 / 0.36), and BIC selects K = 2 with full covariances. Tied and diagonal covariances are rejected by more than 10⁵ BIC units, and so is a 142-dimensional mixture over all features. Supervised in-fold feature selection recovers only 77 % of the informative block, because some block features are informative only jointly. We fitted the K = 2 model as a generator and drew replica datasets with known labels (40 × 20,000 rows). On every replica we re-ran each procedure end to end. The Bayes AUC of the generator is 0.99995, and the S003 procedure comes within 1.7 × 10⁻⁷ of it. Every refinement differed from S003 by less than 10⁻⁸: removing the covariance regulariser (+2.3 × 10⁻⁹ ± 0.8 × 10⁻⁹), semi-supervised EM with the labels as constraints, restart ensembles and polishing. Every alternative was worse: bagging, self-training QDA, extra components, all-feature mixtures (−3.4 × 10⁻⁵), supervised selection (−3.5 × 10⁻⁴), a quadratic logistic model on pseudo-labels (−4.5 × 10⁻⁴) and supervised QDA (−6.7 × 10⁻³).

**Where the remaining public-LB error comes from.** Under clean labels, a public score as low as S003's 0.99923 would occur in only 0.1 % of simulated 395-row public splits. With label noise of the kind `make_classification` adds by default (flip_y = 0.01, i.e. 0.5 % wrong labels), it occurs in 83 %. We combined the public score with the absence of confidently mislabelled training rows. Under a flat prior on flip_y, the resulting posterior wrong-label rate is about 0.31 % (90% interval 0.05–0.76 %), and the implied private AUC is about 0.997 (90% interval 0.992–0.9996; R922) for S003 and for every near-Bayes variant alike. The headroom that better boundary ranking could recover (~10⁻⁷) is three to four orders of magnitude below the private-LB standard deviation at plausible noise rates (3–6 × 10⁻⁴), and about two orders below it even with clean labels (1 × 10⁻⁵; R922). We therefore kept S003 and stopped spending submissions on boundary-row variants. This negative result is the strongest evidence that the final model is at the Bayes limit.

| Candidate (run_id) | CV AUC | CV log-loss | Kendall τ vs S003 on its 204 boundary rows | Replica Δ AUC vs S003 (mean ± SE, 40 reps) | Verdict |
|---|---|---|---|---|---|
| S003 reproduction (`20261022-064436_r102_ref_s003`) | 1.0000 | 0.0171 | 1.000 | 0 (Bayes gap 1.7e-7) | reference |
| Unregularised MLE (`20261022-074857_r105_gmm2_block_mle`) | 1.0000 | 0.0171 | 0.996 | +2.3e-9 ± 0.8e-9 | equivalent, cleaner |
| Semi-supervised EM, ε = 0.01 (`20261022-064500_r102_semisup_eps01`) | 1.0000 | 0.0077* | 0.9996 | +1.6e-10 ± 2.4e-10 | equivalent |
| 10-restart ensemble (`20261022-064447_r102_gmm2_block_seed10`) | 1.0000 | 0.0171 | 1.000 | ≈ 0 | equivalent |
| Bagged mixture (`20261022-064454_r102_gmm2_block_bag20`) | 1.0000 | 0.0174 | 0.982 | −2.4e-8 ± 0.4e-8 | no gain |
| Self-training QDA (`20261022-064735_r102_selftrain_qda99`) | 1.0000 | 0.0074* | 0.980 | −2.1e-8 ± 0.5e-8 | no gain |
| Semi-sup. EM, 2 comp./class (`20261022-064506_r102_semisup_2pc`) | 1.0000 | 0.0075* | 0.905 | −3.7e-6 ± 3.1e-6 | worse |
| GMM on all 150 features (R003 `gmm2_all`; replica: 142-dim PCA) | 1.0000 | – | 0.535† | −3.4e-5 ± 0.1e-5 (8 reps) | worse |
| Supervised top-13 selection + GMM (`20261022-064749_r102_supsel13_gmm2`) | 0.99997 | 0.0254 | 0.697 | −3.5e-4 ± 1.0e-4 | worse |
| Quadratic logit on pseudo-labels (`20261022-064738_r102_quadlogit_pseudo`) | 1.0000 | 0.0510 | 0.361 | −4.5e-4 ± 0.2e-4 | worse |
| Tied covariance (`20261022-064745_r102_gmm2_block_tied`) | 0.7307 | 0.632 | 0.019 | −0.229 | rejected (BIC) |

\* lower log-loss only because these output the posterior directly instead of rate-compressed scores (calibration does not affect AUC). † Spearman on S003's boundary rows, computed from the R003 test predictions. Replica Δ = expected test AUC under the generator's Bayes posterior; test-AUC differences give the same ordering.

## Methods and libraries used (for `registry/references.md`)

- Gaussian mixture / EM: Dempster, Laird & Rubin (1977); scikit-learn `GaussianMixture` (Pedregosa et al., 2011); own NumPy EM for polishing and label-constrained (semi-supervised) EM (e.g. Nigam et al., 2000; Zhu, 2005 survey).
- BIC: Schwarz (1978). Likelihood-ratio test, χ² approximation: Wilks (1938).
- Parametric bootstrap: Efron & Tibshirani (1993).
- Self-training / pseudo-labelling: Yarowsky (1995); Lee (2013).
- AUC SE (for the private-LB scale): Hanley & McNeil (1982). Kendall τ / Spearman ρ: SciPy (Virtanen et al., 2020).
- Generator structure hypothesis: scikit-learn `make_classification` (Guyon, 2003 design) — used only as an interpretation, no external data.
- Libraries: numpy, scipy, scikit-learn 1.9.1, pandas, joblib (versions in `requirements-lock.txt`). No new packages.

## Compliance notes

- Every model uses unlabeled train+test **features** (accepted under D007). Labels enter only through the training rows of each CV fold (component-to-class mapping, or label constraints in semi-supervised EM). No external data, no hand labelling.
- The replica generator maps its components to classes with the 250 training labels. It is an evaluation tool, not a submitted model.
- Nothing was submitted to Kaggle.
