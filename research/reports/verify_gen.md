# verify_gen：生成器识别、Bayes 上限、标签噪声与 private 预期的核查（R920–R922）

作者：AI 核查代理 "gen"（Claude Code），2026-10-22（JST）。计算全部在 Colab（账号 pro1，会话 vgen，High-RAM 8 核；会话已关闭）。
被核查对象：`research/reports/gen.md`（R150–R156）、`research/lib_gen_replica.py`、mix 的复刻数字（`research/reports/mix.md`，R103/R106）、`registry/beliefs.md` 的 B004。
我的代码**不调用** `lib_gen_replica` 或 `lib_mix`，全部独立重写：`research/r920_generator_alternatives.py`、`r921_bayes_regret_replicas.py`、`r922_label_noise_private.py`。

## 中文摘要（给队友）

**一句话**：生成器结构和"S003 离 Bayes 只差约 2.5e-7"都经得起核查；**但关于标签噪声和 private 分数的结论说得太乐观、太确定了**。B004（private ≥ 0.999 且与 public 差 ≤ 0.001，置信度 70%）不现实，按我们的证据，实际概率只有约 15%。

**1. 生成器识别：成立，而且比原报告给的证据更强。**（R920 `20261022-100551_r920_generator_alternatives`）
- 8 条冗余关系是**精确**的：残差标准差正好等于 6 位小数舍入误差的理论值（比值 0.992–1.004）；第 13–20 个特征值就在舍入噪声的水平上（约 2e-14），第 12 个是 0.19。所以 n_informative = 12、n_redundant = 8、n_repeated = 0 是**确定性**结论，不存在"别的拆分同样好"。
- 但要注意："精确线性组合"本身**不能**说明哪 12 列是信息列：20 列里任取 12 列（125,970 种）**全部**满秩。哪几列是信息列，要靠分布假设来判断。我用独立的方法重新推导（A_k 的行是 iid U(−1,1) ⇒ 类内 Gram 矩阵近似 Wishart，逐列尺度做 profile），**没有用标签**，结果与 gen 的集合完全一致：领先第二名 13.0 nats（似然比约 4×10⁵），加上 hypercube 约束后领先 17.7 nats；20/20 次 bootstrap 结果相同；在 24 个已知真值的复刻上 24/24 都找对。我也读了代码：`lib_gen_replica.identify_structure(X)` 不接收标签（lib_gen_replica.py:239–381），标签只用来确定哪个分量叫"类 1"。
- 每类 2 个簇？否。K=4 对 K=2 的似然比统计量：真实数据 480；在"每类 1 簇"的复刻里是 290–1254（真实值在中间）；在"每类 2 簇"的复刻里是 121,000–176,000。差了 250 倍以上，**彻底排除**。
- hypercube=False？8 个"均值差为 0"的坐标加上 4 个活跃坐标效应大小几乎一样，在 hypercube=True 下很自然，在 False 下几乎不可能：局部密度比约 **360 : 1**。
- class_sep：用 4 个活跃坐标上 8 个与尺度无关的比值做精确似然，MLE 0.703，95% 区间 0.64–0.76（复刻校准显示这个区间略窄，覆盖率 21/24；诚实的区间约 0.63–0.78）。**0.6（似然比 0.008）、0.8（0.004）、0.9（≈0）都明显更差**，"0.6–0.9 都一样好"不成立。gen 的 0.61–0.81 偏保守，但没有错。
- 噪声列里没有任何类别信号（KS p = 0.31 / 0.58）；行已打乱；干净类别正好 10,000 : 10,000（z = −0.54）。
- **flip_y 根本不能从特征识别**（它只改标签，不改 X），只能靠 250 个训练标签和 public 分数判断，见第 3 点。

**2. "S003 比 Bayes 低约 2e-7"：差距是真的，不是 MC 噪声；但没有任何实际意义。**（R921 `20261022-094951_r921_bayes_regret_replicas`，1000 个参数复刻 + 1000 个 make_classification 复刻）
- 干净标签下，private 上 Bayes − S003 = **2.51e-7 ± 0.22e-7**（SE，t = 11）；去掉标签抽样噪声的"期望 AUC"差距是 **2.57e-7 ± 0.018e-7**，1000 个复刻里 100% 为正。相当于约 9,400 万对样本里只有约 24 对排反。gen 的 3.1e-7（100 个复刻，SE 0.75e-7）和 mix 的 1.7e-7（40 个复刻）都在各自约 1 个 SE 以内，彼此一致。
- 有标签噪声（flip_y = 0.01）时，配对差距是 −0.3e-5 ± 0.5e-5，与 0 无法区分；private AUC 本身的 sd 是 5.7e-4，比这个差距大 2000 倍。
- 提交文件按 8 位小数写出，产生大量并列（8,918 行取最小值、6,416 行取最大值，只有 2,897 个不同的分数），这只让期望 AUC 损失 5e-11，可以忽略。
- 在一般的 make_classification 复刻上，平均差距是 3.1e-5，主要来自 5.5% 的情形：高方差块漏掉了一个信息方向（这时损失 5.4e-4）。真实数据的块已经张满 12 维（R150 和本次都确认），所以不受影响。

**3. 标签噪声与 private 预期：原结论过于乐观、区间太窄。**（R922 `20261022-092146_r922_label_noise_private`）
做法：在 VM 上**逐位复现**了 S003 的提交分数（哈希与提交文件一致），只用真实的 19,750 行测试特征和标签-free 的后验，模拟"未知的 395 行 public 子集 + 标签噪声"。
- **训练标签**：250 个里**没有一个**与高置信后验矛盾。唯一不一致的是第 9 行（y = 0，后验 0.79，观测标签的概率 0.21），属于边界样本。观测标签概率低于 0.2 的行数为 0。
- **public 0.99923 的含义**：约等于 30 对排反。public 里没有错误标签时，得到这个分数的概率是 1.2e-4；恰有 1 个错误标签时是 2.4e-3（约 20 倍）。所以"至少有一个错标签"更可能，但**"flip_y = 0 几乎不可能"说过头了**：把训练标签和 public 合在一起，flip_y = 0 相对最优值的似然比是 **0.20**；后验的温度变化 ±20% 时，这个比值在 0.03–0.84 之间，所以 0 不能被排除。
- **flip_y 的似然**：峰值在 0.003，似然 ≥ 最大值 1/8 的范围是 **0–0.015**；sklearn 默认值 0.01 的似然比是 0.37（完全可能），0.02 是 0.034。gen 的 4 点网格在 0.01 处截断，mix 的网格在小 f 处更密，两者都让后验偏向小噪声。
- **S003 的 private 预测**（flip_y 取 [0, 0.1] 上的均匀先验）：均值 **0.9968**，中位数 0.9973，**90% 区间 [0.9923, 0.9996]**。flip_y 的后验均值是 0.0063（错标率 0.31%，90% 区间 0.05%–0.77%），不是"约 0.2%"。如果用 gen 的 4 点网格先验，能复现 gen 的 0.9978，说明 gen 的计算本身没错，差别全在先验。各个 flip_y 下的 private 分布：f = 0 → 0.99995；0.0025 → 0.9987 ± 0.0003；0.005 → 0.9975 ± 0.0004；0.01 → 0.9949 ± 0.0006；0.02 → 0.9900 ± 0.0008。
- **B004 不现实**：private ≥ 0.999 要求 flip_y ≲ 0.002。三种合理先验（均匀、gen 的网格、"作者很可能用了 sklearn 默认值"）下概率都只有 **0.14–0.17**；只有在偏向极小噪声的对数均匀先验下才到 0.51。|public − private| ≤ 0.001 的概率是 0.33。public 和 private 的预期差距约 0.002，而不是 0。

**建议**：
1. B004 按规则不能改。请主会话另登记一条新 belief，例如"S003 private 的中位数 0.997，90% 区间 [0.992, 0.9996]，P(≥ 0.999) ≈ 15%"。
2. gen.md 里给报告用的英文段落要改：不要写 "P(flip_y > 0) = 0.92"、"flip_y ≈ 0.004"、"expected private ≈ 0.9977" 这种精确数字，改成"flip_y ≈ 0.003（似然支持 0–0.015），private 约 0.997（90% 0.992–0.9996）"。mix.md 里"private LB 的 SE 约 1e-4"也不对：有噪声时 sd 是 3–6e-4，干净标签时是 1e-5。
3. 模型选择不受影响：S003 类模型就是 Bayes 极限，private 分数由标签噪声决定。这些结论都不需要再占用提交次数。

---

## Details (English)

### Runs and jobs

| Run | Job | What |
|---|---|---|
| `20261022-100551_r920_generator_alternatives` | J20261022-100732-pro1-vgen | alternatives to the generator: exactness, basis likelihood, clusters/class LR, hypercube, class_sep, noise columns, order/balance |
| `20261022-094951_r921_bayes_regret_replicas` | J20261022-094632-pro1-vgen | Bayes − S003 regret on 1000 parametric + 1000 make_classification replicas (own generators) |
| `20261022-092146_r922_label_noise_private` | J20261022-092327-pro1-vgen | S003 bit-exact reproduction, train-label check, public-score likelihood, private distribution per flip_y, posteriors under 5 priors, B004 |

Tables: `reports/tables/r920_*`, `r921_*`, `r922_*`; figures: `reports/figures/r920_generator.png`, `reports/figures/r922_label_noise.png`. Each run's notes.md has a Conclusion.

### 1. Generator identification (R920)

| Alternative / claim | Statistic (real data) | Reference distribution | Verdict |
|---|---|---|---|
| n_informative/n_redundant split ≠ 12/8 | λ₁₂ = 0.193, λ₁₃…λ₂₀ ≈ 2.5e-14 (0.87 × the 6-decimal rounding floor) | any other split changes the rank deterministically | 12/8/0 confirmed (deterministic) |
| dependencies exact | R² ≥ 1 − 4e-14; residual sd / sd predicted by rounding alone = 0.992–1.004 for all 8 (`r920_dependency_residuals.csv`) | rounding error only | confirmed: residuals are pure rounding noise |
| "exact combination" identifies the roles | all 125,970 12-subsets of the 20 columns are full rank | – | roles are **not** identified by exactness; they need the distributional argument below |
| informative basis (label-free) | Wishart profile log-lik (A_k rows iid U(−1,1), per-column scales profiled): claimed set best, margin 13.0 nats (runner-up swaps 15↔26), 17.7 nats under the hypercube constraint; 20/20 row bootstraps | 24/24 known-truth replicas recovered, margin median 11.7, min 5.1 | confirmed (independent criterion; gen's log det within-class correlation is the pooled special case) |
| n_clusters_per_class = 2 | LR(K4 vs K2) = 480 | H0 parametric [350, 536], H0 make_classification [290, 1254], H1 two clusters/class [121,298, 176,157] (12 replicas each) | 2 clusters/class rejected by a factor > 250 in the statistic |
| hypercube = False | 8 zero-difference z's: Σz² = 9.11 (χ²₈ cdf 0.67); 4 active standardised differences max/min = 1.21 | sklearn's hypercube=False centroid process, ABC-matched effect size, 2M draws each: P(joint ≤ obs) 0.357 (True) vs 4.7e-5 (False); local density ratio ≈ 360 | hypercube = True favoured ~360:1 |
| class_sep 0.6–0.9 | exact likelihood from w_ki = Δx_i²/var_k(x_i) = 4cs²/V_ki (V ~ Σ₁₂U², 8 independent values): MLE 0.703, 95% LR [0.639, 0.760]; LR vs MLE: 0.6 → 0.008, 0.65 → 0.27, 0.75 → 0.27, 0.8 → 0.004, 0.85 → 1e-6, 0.9 → 0 | 24 replicas at 0.71: MLE mean 0.702, sd 0.038, CI coverage 21/24 | class_sep ≈ 0.70 (honest ±0.075); 0.6/0.8/0.9 not equally good; gen's 0.61–0.81 is conservative but correct |
| class info outside the block | 130 noise columns: z(mean) KS p 0.31, z(log var ratio) KS p 0.58, max \|z\| 3.11 (Bonferroni p 0.49); std ~U(0.5, 3) KS p 0.26, mean ~U(−5, 5) KS p 0.87 | N(0,1) | none |
| shuffle | Spearman(id, posterior) test −0.001, train 0.079 (SE 0.063); runs test z −1.1 | iid rows | rows shuffled |
| weights = None | Σ posterior = 9,996.2 ± 7.0 → z = −0.54 vs exactly 10,000; train Σy = 113 vs Σπ = 113.6 | exact 10,000/10,000 clusters | balanced |
| flip_y | X is unchanged by flip_y | – | not identifiable from features; label evidence in §3 |

Small discrepancy with gen's R152: gen reports "zero-difference set recovered in 95%". Over all 20 structured columns I find the true zero set in 24/24 replicas, but in 19/24 one or more redundant columns also have \|z\| < 4. A redundant column's mean difference is 2cs·Σ±B_ij, which is sometimes near 0. On the real data exactly 8 columns have \|z\| < 4, and all of them are in the likelihood-best basis. This is consistent and does not affect the claim.

### 2. Bayes AUC and the S003 regret (R921)

Own parametric generator: a converged K = 2 GMM in the 12-dim PCA basis of the 20 structured columns. The structured columns are rebuilt exactly (max reconstruction error 3.4e-7), the noise columns are iid, and values are rounded to 6 decimals. The high-variance rule selected the real 13-column block in 99.7% of replicas. I also wrote my own make_classification (class_sep 0.71, hamming 4, affine map). The S003 procedure was refitted on every replica. "Expected AUC" = E[concordant]/E[n1·n0] with labels ~ Bern(true posterior), which removes label-sampling noise.

| Group | n | Bayes clean private AUC | Regret, observed (mean ± SE) | Regret, expected AUC (mean ± SE) | Regret at flip 0.01 |
|---|---|---|---|---|---|
| parametric | 1000 | 0.999952 (sd 1.2e-5) | 2.51e-7 ± 0.22e-7 (t 11; 36% < 0) | 2.57e-7 ± 0.018e-7 (100% > 0; 90% range 1.8–3.6e-7) | −0.3e-5 ± 0.5e-5 |
| make_classification, all | 1000 | 0.999976 | 3.1e-5 ± 0.8e-5 | 3.1e-5 ± 0.8e-5 | 3.5e-5 ± 1.0e-5 |
| … block spans, no noise columns | 186 | 0.999981 | 8.8e-7 ± 5.2e-7 | 8.1e-7 ± 5.0e-7 | – |
| … block spans, with noise columns | 759 | 0.999975 | 1.5e-6 ± 1.0e-6 | 1.6e-6 ± 1.0e-6 | – |
| … block misses a direction | 55 | 0.999971 | 5.4e-4 ± 1.2e-4 | 5.4e-4 ± 1.2e-4 | – |

Verdict: the gap is real and far outside MC noise (SE 2e-9 for the expected-AUC version). It is equivalent to ~24 of 9.4·10⁷ private pairs. The 8-decimal rounding of the csv adds 5e-11. Under any plausible label noise the gap is invisible (private sd 5.7e-4 at flip 0.01). Gen's 3.1e-7 is 0.7 of its own SE above my estimate and mix's 1.7e-7 about 0.8 SE below it, so all three agree.

### 3. Label noise, public score and the private distribution (R922)

Setup: the S003 test scores were recreated on the VM with ssc TransductiveGMM("highvar", 2) and the official 5×20 fold mean, formatted "%.8f". The hash of the 19,750 strings is c70057e06406a2d6, **identical** to `submissions/S003_*.csv`. The ranking Kaggle scored is heavily tied: 8,918 rows at the minimum, 6,416 at the maximum, 2,897 distinct values. As the clean-class posterior π I used a converged label-free K=2 GMM (PCA basis, tol 1e-10). It agrees with S003's own mixture to within 0.014 (one row differs by more than 0.01). Labels: c ~ Bern(π); with probability flip_y the label is redrawn uniformly, so P(wrong) = flip_y/2. The public set is a random 395 of the 19,750 test rows (6M simulated draws; k = 0 has 742 hits, 3.7% relative MC error).

**Train labels.** There are 0 rows with P(observed label) < 0.2 and 1 row < 0.5 (row 9: y = 0, π = 0.788). L_train(f) ∝ Π[(1−f)π(y) + f/2] ≈ exp(−125 f).

**Public score.** P(public rounds to 0.99923 | k wrong public labels):

| k | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| exact 5-decimal window | 1.24e-4 | 2.44e-3 | 4.1e-4 | 3.8e-5 | 3e-6 |
| ±1.5e-4 (R153/R156 window) | 1.77e-3 | 3.04e-2 | 5.4e-3 | 4.8e-4 | 3e-5 |
| ±4e-4 (R106 window) | 1.48e-2 | 9.4e-2 | 1.5e-2 | 1.3e-3 | 9e-5 |

P(public ≤ 0.99923 | clean) = 0.00075 (mix: 0.001) and P(public = 1 | clean) = 0.51 (gen: 0.52), both reproduced.

**Combined likelihood** (exact window): the MLE is flip_y = 0.003; the 1/8-support is [0, 0.0147] and the 1/32-support is [0, 0.020]. Relative likelihood: f = 0 → 0.20, 0.001 → 0.71, 0.0025 → 0.99, 0.005 → 0.88, 0.01 → 0.37, 0.015 → 0.12, 0.02 → 0.034, 0.03 → 0.002. Sensitivity: the f = 0 value depends on the posterior's tail calibration (temperature 0.8 / 1.25: LR₀ = 0.84 / 0.03; R106's wide window: 0.52). The upper end of the support is stable (0.012–0.015 in all variants).

**Private AUC of S003 given flip_y** (19,355 rows, 3000 draws per f):

| flip_y | 0 | 0.001 | 0.002 | 0.0025 | 0.003 | 0.005 | 0.0075 | 0.01 | 0.015 | 0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.99995 | 0.99945 | 0.99895 | 0.99869 | 0.99845 | 0.99746 | 0.99619 | 0.99494 | 0.99244 | 0.98995 |
| sd | 1.1e-5 | 1.7e-4 | 2.4e-4 | 2.8e-4 | 3.1e-4 | 3.9e-4 | 4.8e-4 | 5.5e-4 | 6.6e-4 | 7.8e-4 |
| P(≥ 0.999) | 1 | 0.99 | 0.44 | 0.14 | 0.03 | 0 | 0 | 0 | 0 | 0 |

The mean is 1 − f/2 to within 1e-5. The S003-mixture posterior gives the same values to within 3e-5.

**Posterior predictive** (`r922_posterior_by_prior.csv`):

| Prior on flip_y | post. mean f (90%) | private mean | private 90% | P(B004) = P(private ≥ 0.999) | P(\|pub − priv\| ≤ 0.001) |
|---|---|---|---|---|---|
| uniform [0, 0.1] | 0.0063 (0.001–0.015) | 0.9968 | [0.9923, 0.9996] | 0.16 | 0.33 |
| uniform [0, 0.03] | 0.0063 | 0.9968 | [0.9923, 0.9996] | 0.16 | 0.33 |
| gen's 4-point grid {0, 0.0025, 0.005, 0.01} | 0.0044 | 0.9978 | [0.9947, 0.99995] | 0.14 | 0.47 |
| default-aware (0.2 at 0, 0.5 at sklearn's 0.01, 0.3 flat) | 0.0081 | 0.9959 | [0.9940, 0.99995] | 0.17 | 0.19 |
| log-uniform [1e-4, 0.1] (favours tiny f) | 0.0030 | 0.9985 | [0.9952, 0.99995] | 0.51 | 0.69 |

Gen's numbers (mean 0.0044, private ≈ 0.9977) are reproduced exactly under gen's prior, so the computation is right. The precision comes from the prior: a 4-point grid truncated at 0.01 with 25% mass at 0. Mix's posterior wrong-label rate of 0.21% (0.05–0.4%) comes from a grid that is denser at small f and a ±4e-4 window that inflates f = 0. Under a flat prior the rate is 0.31% (0.05–0.77%).

### Problems found (ranked)

1. **B004 is miscalibrated** (registry/beliefs.md). It states 70% for private ≥ 0.999 with \|public − private\| ≤ 0.001. Because the public score is 0.99923, the conjunction is equivalent to private ≥ 0.999, which needs flip_y ≲ 0.002. Its probability is 0.14–0.17 under the reasonable priors and 0.51 only under a log-uniform prior.
2. **gen.md overstates the label-noise conclusions**: "flip_y ≈ 0.0025–0.005, P(>0) = 0.92", "public 0.99923 nearly impossible at flip_y = 0" and "S003 private ≈ 0.9977 (90% 0.9944–0.9991)". The public-only likelihood ratio against f = 0 is ~8, and the combined ratio is ~5 (0.03–0.84 under calibration changes). f up to 0.015 is well supported, including sklearn's default 0.01 (LR 0.37). The honest private interval is [0.992, 0.9996]. The "Text for the report" paragraph and Table G1 should be reworded.
3. **mix.md**: "private LB SE ~1e-4" is wrong. It is 1.1e-5 under clean labels and 2.5–6e-4 at plausible noise rates. The interval 0.05–0.4% for the wrong-label rate is too narrow. Neither changes mix's model conclusions.
4. Minor: gen's class_sep interval (0.61–0.81) is wider than the likelihood supports, so it is conservative. Gen's zero-set recovery rate is defined inside the identified basis. Neither is an error.

### Not verified / limits

- The label model assumes sklearn flip semantics and a random 395-row public split. A stratified or differently sized public split was not checked; ±5 rows has no material effect.
- The k = 0 public likelihood depends on the tail calibration of the Gaussian posterior. I bracketed it with temperatures 0.8 and 1.25; the true model is Gaussian (R920), so a large miscalibration is unlikely.
- I did not check other agents' candidates, only S003. Every near-Bayes candidate has the same private distribution to within ~1e-5.
- Generator alternatives outside make_classification, such as non-uniform A_k entries, were not modelled. The basis criterion's Wishart approximation of the uniform-row Gram matrix was validated only by the 24/24 replica recoveries.
