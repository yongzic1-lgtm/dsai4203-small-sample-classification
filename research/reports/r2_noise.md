# r2_noise：标签噪声与 public/private 模型（第 2 轮研究，R420–R426）

- 作者：AI 研究代理 "r2_noise"（Claude Code，Claude Opus 5.5），2026-10-22 18:20–20:00 JST。
- 计算：全部在 Colab（账号 pro1，会话 r2noise，High-RAM 8 核，6 个作业，会话已关闭）。本机只做了 < 30 s 的冒烟测试，档案重定向到 scratchpad。
- 新脚本：`research/r420_replica_flip_grid.py`、`r421_public_split_joint.py`、`r422_linear_ceiling_r921.py`、`r424_pack_r420.py`（纯文件打包，不是 run）、`r425_flip_grid_reweight.py`、`r426_abc_attribution.py`。
- 复用的已核查代码（没有修改）：`lib_gen_replica.py`、`r931_cand_lib.py`、`lib_mix.expected_auc`。`r922`、`r921`、`r933` 是脚本，import 时就会运行，所以把要用的函数原样复制进新脚本。复制有没有出错，用逐位复现来检验（见 §1.1、§2.1、§3.1）。
- 没有上传 Kaggle，没有改 `registry/*.md`、`registry/submissions.csv`，没有删文件。

## 中文摘要（给队友）

**一句话**：三个问题的答案都是"原结论基本不变，数字更准了"。S003 的 private 预期是 0.9968（90% 区间 0.992–0.9996），与 B008 一致；S005、S006 的 public 预测与 B006、B007 一致，只是 B006 的点估计略偏高；分层 split 不改变任何结论；线性上限 0.947 和边界行数约 186 都经得起 R921 生成器的重算。

**(a) 把 flip_y 网格扩展到 0.02（实际到 0.03），用 R922 的平坦先验似然重新加权**（R420、R425、R426）
- 在同样的 222 个复刻上加了 20 个噪声水平。原来 4 个水平上的 284,160 个 AUC 与 r933 的存档**逐位相同**。
- 每个水平上 S003 的 private 都约等于 1 − f/2：f = 0.015 时 0.9924，f = 0.02 时 0.9899。r201、r209 相对 S003 的差距在 0–0.03 之间**基本不随 f 变化**（r201 约 −2.5e-4，r209 约 −6.5e-5）。
- **S003 的 private 分布**：
  - R922 平坦先验下，均值 0.9968，中位数 0.9974，90% 区间 [0.9922, 0.9996]，P(≥ 0.999) = 0.16。这与 R922 在真实特征上的结果、B008 都一致。
  - r936 用 R156 的 4 点网格时是 0.9978。差别来自权重：4 点网格在 0.01 以上没有质量，而平坦先验下 f > 0.01 的后验概率是 17%。
- **第二个最终选择的收益**：r209 是 +2.0e-5，r201 是 +7.0e-6（r936 时分别是 +1.45e-5 和 +4.1e-6）。两者都远小于 private 的波动，**D010 不需要改**。
- **S005 / S006 的 public 预测**（以 S003 的 0.99923 为条件）：

| | S005 = r201 | S006 = r209 |
|---|---|---|
| 预登记（B006 / B007） | 0.9988，80% 0.9980–0.9997，P(高于 S003) ≈ 0.20 | 0.9991，80% 0.9987–0.9996，P(高于) 0.25–0.28，P(相同) 0.04 |
| 复刻 ABC（R425，R922 加权） | **0.99868**，80% 0.99782–0.99941，P(高于) 0.19 | **0.99910**，80% 0.99882–0.99941，P(高于) 0.27，P(相同) 0.034 |
| 真实特征联合模拟（R421，精确窗口） | **0.99852**，80% 0.99767–0.99933，P(高于) 0.14 | **0.99912**，80% 0.99884–0.99941，P(高于) 0.29，P(相同) 0.045 |
| 落在预登记 80% 区间内的概率 | 0.85（复刻）/ 0.79（真实特征） | 0.92 / 0.94 |

  - B007 准确。
  - B006 的点估计偏高 1–3e-4，但区间仍有约 80% 的覆盖率。所以登记 outcome 时，拿 0.9985–0.9987 作参照更合适。
  - r936 当时的 0.99884 只用了 125 个被接受的 public 子集（MC 误差约 8e-5）。R426 证明，从 0.99884 降到 0.99868 是蒙特卡罗噪声造成的，与网格扩展无关：改变权重只让预测移动 < 1e-5。

**(b) 分层 split（public 正例数固定为 197 或 198）**（R421）：**不改变任何结论**。
- 每个 k 下得到 0.99923 的概率变化都在 MC 误差以内。
- flip_y 的 MLE 仍是 0.003，1/8 支持区间仍是 [0, 0.0148]，LR(f = 0) 仍是 0.19–0.20。
- S003 的 private 区间变化 ≤ 1e-5。
- S005 / S006 的条件 public 预测变化 < 4e-5。
- verify_gen 列出的"分层 split 没有核查"这一项可以关闭。

**(c) R154 的线性上限和边界行数，用 R921 的生成器重算**（R422）：**成立**。
- 线性上限：两种拟合给出同一个最佳线性投影 AUC 0.946843。参数自助法下的抽样 sd 是 0.0014，所以写"0.947 ± 0.0015"。Bayes AUC 0.99995，Bayes 错误率 0.31–0.33%。
- 边界行数（0.1 < p < 0.9）：期望值 182–186 行，单个测试集的 sd 是 14。真实数据用收敛后验算是 189 行，位于复刻分布的第 68 百分位。
- 报告里出现的 203（r321）和 204（r932）不是另一个模型的结果，而是另一种定义：它们用的是 S003 的分数，这个分数被压缩在 [0.0061, 0.9864] 之间，不是校准过的后验。
- 用 250 行训练的线性模型在复刻上只有 0.894（收缩 LDA）和 0.912（L2 逻辑回归），比上限低 0.035–0.053。

**建议**（只供主会话参考，我没有改 registry）：
1. B006、B007 不改。上传后登记 outcome 时，参照本报告 §1.5 的表。
2. `reports/extensions/e2_learning_curve.md` 第 14 行和第 61 行，"203 rows with a posterior between 0.1 and 0.9" 改为"about 190 rows (1 %) have a posterior between 0.1 and 0.9 (189 with the converged posterior, R922/R422; 203 by S003's own score, r321)"。
3. `research/r424_r420_pooled.npz`（3.0 MB）会随每个代码包上传。等 r425/r426 冻结后可以删除，`r935_r933_pooled.npz` 也一样。我没有删。

---

## Details (English)

### Runs and jobs

| Run | Job (wall s) | What |
|---|---|---|
| `20261022-182358_r420_replica_flip_grid_s136-209` | J20261022-182536-pro1-r2noise (567) | r933 tn0 replicas (seeds 136–209), 24 flip_y levels 0–0.03, 512 public subsets per replica |
| `20261022-185515_r420_replica_flip_grid_s210-357` | J20261022-185653-pro1-r2noise (1299) | same, seeds 210–357 |
| `20261022-183558_r421_public_split_joint` | J20261022-183617-pro1-r2noise (522) | R922 likelihood with random vs stratified public split; joint S003/S005/S006 public simulation on real features |
| `20261022-191747_r425_flip_grid_reweight` | J20261022-191928-pro1-r2noise (679) | pooled analysis of R420: bit-exact check, per-level table, R922 re-weighting, ABC public predictions |
| `20261022-193028_r422_linear_ceiling_r921` | J20261022-193209-pro1-r2noise (678) | R154's ceiling, Bayes error and boundary rows with R921's generator; 200-replica bootstrap |
| `20261022-194211_r426_abc_attribution` | J20261022-194353-pro1-r2noise (700) | why the r201 public prediction moved from r936's 0.99884 to 0.99868 |

Tables: `reports/tables/r421_*`, `r422_*`, `r425_*`, `r426_*`. Figures: `reports/figures/r421_split_joint.png`, `reports/figures/r425_flip_grid.png`. Each run's `notes.md` has a Conclusion. The wall times include upload and merge; the computation itself took 50 s (R422) to 1,077 s (R420 s210-357).

### 1. (a) Extended label-noise grid on the replicas, re-weighted with R922

#### 1.1 Design and reproduction check

R420 is r933's worker in the decision-relevant mode `--train-noise 0 --round-decimals 8` ("tn0": train labels clean, test labels flipped with sklearn's flip_y semantics, scores formatted `%.8f` like the submission CSV). It runs on the same seeds as the archived tn0 runs (136–357, 222 parametric replicas). The levels are 0, 0.001, …, 0.02, plus 0.0025, 0.025 and 0.03 (24 levels). The label noise is nested: one uniform number per row and seed, so a row flipped at f is also flipped at every f' > f. Each replica has 512 public subsets; the first 16 are r933's.

**Check.** At the 4 shared levels and the first 16 subsets, 142,080 public and 142,080 private AUCs equal the archived arrays of `20261022-165947_r933_…_s136-246_tn0_r8_cv` and `20261022-165009_r933_…_s247-357_tn0_r8_cv` exactly (max |diff| 0; `r425_check.csv`). The restriction of the ABC to r936's design reproduces `r936_public_abc.csv` exactly: 125 accepted subsets from 68 replicas, r201 0.998836, r209 0.999129 (`r425_r936_repro.csv`).

#### 1.2 Per level (`r425_levels.csv`, csv scores, 222 replicas × 16 subsets)

| flip_y | S003 private mean (sd) | 1 − f/2 | r201 − S003 (MC SE) | r209 − S003 (MC SE) | accept ±1.5e-4 |
|---|---|---|---|---|---|
| 0 | 0.99995 (1.1e-5) | 1 | −2.45e-4 (0.7e-5) | −6.5e-5 (0.1e-5) | 0.16 % |
| 0.0025 | 0.99870 (2.9e-4) | 0.99875 | −2.56e-4 (1.0e-5) | −6.9e-5 (0.5e-5) | 1.00 % |
| 0.005 | 0.99744 (4.1e-4) | 0.99750 | −2.58e-4 (1.3e-5) | −6.9e-5 (0.7e-5) | 1.25 % |
| 0.01 | 0.99489 (5.5e-4) | 0.99500 | −2.61e-4 (1.6e-5) | −7.0e-5 (1.0e-5) | 1.00 % |
| 0.015 | 0.99239 (7.0e-4) | 0.99250 | −2.42e-4 (1.8e-5) | −5.7e-5 (1.2e-5) | 0.56 % |
| 0.02 | 0.98991 (7.9e-4) | 0.99000 | −2.47e-4 (2.1e-5) | −6.7e-5 (1.4e-5) | 0.28 % |
| 0.03 | 0.98490 (9.2e-4) | 0.98500 | −2.56e-4 (2.6e-5) | −5.9e-5 (1.7e-5) | 0.07 % |

S003's private AUC equals 1 − f/2 within 1.2e-4 at every level. The paired differences do not depend on f; cand's "gaps are flat in flip_y" now holds up to 0.03. The untied Bayes log-odds against S003's tied csv scores differ by −1.5e-5 to −3.0e-5 at f ≥ 0.014. That is at most 1.6 MC SE, so I treat it as noise and did not pursue it.

#### 1.3 S003's private predictive (`r425_private.csv`)

Level weights are R922's combined likelihood (train labels × S003's public 0.99923, exact window, converged posterior) under a flat prior on [0, 0.1], binned to the nearest level. Only 0.09 % of the posterior mass lies beyond 0.03.

| Weights | mass f ≤ 0.005 / 0.006–0.01 / > 0.01 | mean | median | 90 % | P(≥ 0.999) | P(\|0.99923 − priv\| ≤ 0.001) |
|---|---|---|---|---|---|---|
| r936 (R156 4-point grid) | 0.85 / 0.16 / 0 | 0.99775 | 0.99799 | [0.99468, 0.99995] | 0.14 | 0.46 |
| **R922 flat prior (primary)** | 0.54 / 0.30 / 0.17 | **0.99683** | **0.99738** | **[0.99218, 0.99958]** | **0.16** | 0.33 |
| R421 flat prior, stratified split | 0.53 / 0.30 / 0.17 | 0.99681 | 0.99737 | [0.99216, 0.99958] | 0.16 | 0.33 |
| R922 real features (for comparison) | – | 0.9968 | 0.9973 | [0.9923, 0.9996] | 0.16 | 0.33 |
| R421 real features, random split | – | 0.99683 | 0.99734 | [0.99230, 0.99959] | 0.16 | 0.32 |

Conditioning on S003's replica public inside the ABC gives the same answer: 0.99683 ± 0.00003, P(≥ 0.999) 0.17 (`r425_public_pred.csv`). The replicas thus reproduce R922's real-feature answer by an independent route, and B008 is right as written: median about 0.997, 90 % [0.992, 0.9996], P(≥ 0.999) about 0.15, and about 0.0024 below the public score. cand's "≈ 0.998, confirmed" (verify_cand) was produced by the truncated grid, as verify.md C1 already says.

#### 1.4 Second final selection

E[max(S003, X)] − E[S003] (private, csv scores):

| Weights | r209 | r201 | difference |
|---|---|---|---|
| r936 (R156) | +1.45e-5 | +4.1e-6 | 1.0e-5 |
| R922 flat | +2.0e-5 | +7.0e-6 | 1.3e-5 |

The gains grow slightly with the noise level, because the spread of the pairwise difference grows. They remain below 1/100 of the per-level private sd (4–8e-4). D010's reasoning, a first-order rule risk against a 1e-5 AUC difference, is unchanged.

#### 1.5 Public AUC of S005 (r201) and S006 (r209) given S003's public 0.99923

**Replica ABC (R425).** A replica public subset is accepted when S003's csv-score public AUC is within ±w of 0.99923. The prediction is 0.99923 + (public of X − public of S003) on the same subset and labels. Three weightings are reported:
- R922's posterior per level, with equal weight within a level (primary);
- pure ABC, where flat prior × R922's train likelihood is the prior and the replica acceptance supplies the public likelihood (implied posterior mean f 0.0063, the same as R922's);
- R421's stratified-split posterior.

All three agree within 4e-5. MC SEs come from a cluster bootstrap over replicas.

**Real-feature joint simulation (R421).** The actual submitted strings of S003, S005 and S006 are used; their value hashes c70057e06406a2d6, b1bc5699fd161b05 and 284a27641b3cada1 were recreated on the VM. Clean labels are drawn from R922's converged posterior. 12 M random public subsets are simulated with k = 0…10 wrong public labels, and a draw is accepted when S003's public rounds exactly to 0.99923. The flat prior × train likelihood weights k.

| Source | S005 r201: mean (MC SE) | 10–90 % | P(> S003) | P(= S003) | P(in B006 80 %) | S006 r209: mean (MC SE) | 10–90 % | P(> S003) | P(= S003) | P(in B007 80 %) |
|---|---|---|---|---|---|---|---|---|---|---|
| B006 / B007 (registered) | 0.9988 | 0.9980–0.9997 | ~0.20 | – | 0.80 | 0.9991 | 0.9987–0.9996 | 0.25–0.28 | ~0.04 | 0.80 |
| r936 (125 subsets, R156) | 0.99884 (8e-5) | sd 6.7e-4 | 0.20 | 0 | – | 0.99913 (4e-5) | sd 3.3e-4 | 0.28 | 0.037 | – |
| **R425 replicas, ±1.5e-4** | **0.99868 (2.4e-5)** | 0.99782–0.99941 | 0.19 | 0.013 | 0.85 | **0.99910 (0.7e-5)** | 0.99882–0.99941 | 0.27 | 0.034 | 0.92 |
| R425 replicas, ±5e-5 | 0.99865 (3.4e-5) | 0.99774–0.99938 | 0.18 | 0.011 | 0.83 | 0.99909 (1.1e-5) | 0.99877–0.99939 | 0.28 | 0.032 | 0.91 |
| **R421 real features, exact** | **0.99852** | 0.99767–0.99933 | 0.14 | 0.012 | 0.79 | **0.99912** | 0.99884–0.99941 | 0.29 | 0.045 | 0.94 |
| R421 real, stratified n1* = 198 | 0.99855 | 0.99769–0.99933 | 0.13 | 0.009 | 0.80 | 0.99912 | 0.99887–0.99939 | 0.29 | 0.049 | 0.93 |

Under R421's "default-aware" prior the two means are 0.99850 and 0.99912. The predictions do not depend on the prior.

**Why r201 moved from r936's 0.99884 to 0.99868 (R426).** The conditional public difference r201 − S003 is −4.3e-4 at f = 0 and −5.1e-4 to −6.1e-4 at every level from 0.001 to 0.03 (cluster SE 2.5–8e-5). The design comparison:

| Design | r201 prediction (MC SE) |
|---|---|
| r936 (4 levels, 16 subsets, 125 accepted) | 0.99882 (7.3e-5) |
| 4 levels, all 512 subsets | 0.99869 (2.7e-5) |
| R156 weights | 0.99869 |
| R922 weights truncated at 0.01 | 0.99869 |
| full grid | 0.99868 (2.4e-5) |

The change is Monte-Carlo noise in r936's small accepted sample; the grid extension moves the prediction by < 1e-5. For r209, every 512-subset design gives 0.99910 (r936 design: 0.99911).

**Verdict on the beliefs.**
- **B007 is accurate** in every component: point, interval, P(higher) and P(equal).
- **B006's point 0.9988 is 1.2e-4 (replicas) to 2.8e-4 (real features) too high.** Its 80 % interval still has 0.79–0.85 probability under both models, and P(higher than S003) is 0.14–0.19 against the registered ~0.20. The "below 0.9960 = file problem" rule is still a sensible alarm, with probability 0.006–0.011 under the models.
- The two models bracket the truth from different sides. The replicas are somewhat harder for supervised QDA than the real data (r936_cv_calib), and the real-feature model draws labels from a posterior that ranks like S003 (r932's circularity caveat). Both bias r201 downward, so its true public expectation may be nearer 0.9987–0.9988 than 0.9985.

### 2. (b) Stratified 395-row public split (R421)

#### 2.1 Model

If Kaggle stratified the public split on the observed test labels, the public set would have a fixed n1* positives. sklearn's proportional allocation gives n1* = round(395 · N1 / 19,750), where N1 is the number of observed test positives. With Σπ_test = 9,882.6 and sd about 7–10, n1* = 198 with probability 0.82 (at f = 0.003) and 197 otherwise.

Conditioning R922's random-subset simulation on observed public n1 = n1* reproduces a stratified design exactly, up to a factor that varies by < 1 % with N1. R421 therefore keeps the full histogram over (k wrong labels, observed n1, 2D) from 12 M draws per k. The random-split probabilities reproduce R922: 1.17e-4 / 2.42e-3 / 4.1e-4 for k = 0 / 1 / 2 against R922's 1.24e-4 / 2.44e-3 / 4.1e-4, within 1.3 combined MC SE.

#### 2.2 Results

P(public rounds to 0.99923 | k) (`r421_window_probs.csv`):

| k | random | stratified n1* = 197 | stratified n1* = 198 |
|---|---|---|---|
| 0 | 1.17e-4 (1,406 hits) | 1.38e-4 | 1.26e-4 (61 hits) |
| 1 | 2.42e-3 | 2.63e-3 | 2.55e-3 |
| 2 | 4.1e-4 | 4.2e-4 | 4.6e-4 |
| 3 | 3.6e-5 | 2.5e-5 | 5.1e-5 |

Likelihood support and S003's private predictive under the flat prior (`r421_support_and_private.csv`):

| Split | MLE | 1/8 support | 1/32 support | LR(0) | LR(0.01) | LR(0.02) | post. mean f | S003 private mean, 90 % | P(≥ 0.999) |
|---|---|---|---|---|---|---|---|---|---|
| random (R922 model) | 0.003 | [0, 0.0148] | [0, 0.0203] | 0.189 | 0.374 | 0.034 | 0.0063 | 0.99683, [0.99230, 0.99959] | 0.162 |
| stratified 197 | 0.003 | [0, 0.0148] | [0, 0.0203] | 0.204 | 0.370 | 0.034 | 0.0063 | 0.99685, [0.99236, 0.99961] | 0.164 |
| stratified 198 | 0.003 | [0, 0.0148] | [0, 0.0203] | 0.191 | 0.377 | 0.035 | 0.0063 | 0.99682, [0.99227, 0.99960] | 0.160 |
| stratified, mixed | 0.003 | [0, 0.0148] | [0, 0.0203] | 0.193 | 0.376 | 0.035 | 0.0063 | 0.99683, [0.99230, 0.99960] | 0.162 |

**Answer: no.**
- The window for 0.99923 is the single value 2D = 60 for every n1 near 197.5.
- The distribution of D given k hardly depends on n1. A wrong label costs about the same number of pairs whether the public set has 190 or 198 positives.
- The private AUC distribution per f is the same under both splits within 5e-5 (`r421_private_by_flip.csv`).
- The k = 0 stratified value rests on 61 hits (about 13 % MC error). This does not matter: R922's posterior-temperature sensitivity moves LR(0) from 0.03 to 0.84.

### 3. (c) Linear-projection ceiling and boundary rows with R921's generator (R422)

#### 3.1 Population quantities (`r422_population.csv`, 400k rows per class)

| Quantity | R921 fit (converged, PCA basis) | R154 fit recomputed | R154 archived |
|---|---|---|---|
| best linear AUC, formula Φ(√(Δμ'(Σ0+Σ1)⁻¹Δμ)) | 0.946843 | 0.946843 | 0.946843 |
| best linear AUC, MC (= LDA direction) | 0.94665 | 0.94663 | 0.94663 |
| mean-only / covariance-only AUC | 0.94696 / 0.99979 | 0.94691 / 0.99980 | 0.94704 / 0.99980 |
| quadratic Bayes AUC | 0.999949 | 0.999949 | 0.999949 |
| pooled Mahalanobis | 2.28394 | 2.28393 | 2.28393 |
| Bayes error (MC) | 0.319 % | 0.327 % | 0.327 % |
| expected test rows with 0.1 < p < 0.9 | 182.5 | 186.1 | 186.1 |
| … 0.05–0.95 / 0.01–0.99 | 272.8 / 531.8 | 276.9 / 534.7 | – |

R154's Monte-Carlo values are reproduced exactly (same seed). The two fits describe the same Gaussian pair, so the ceiling does not depend on the fitting details.

#### 3.2 Parametric bootstrap with R921's generator (200 replicas; `r422_replica_summary.csv`)

| Quantity | mean ± sd (5–95 %) | real data |
|---|---|---|
| refitted best-linear ceiling | 0.94696 ± 0.0014 (0.9445–0.9493) | 0.9468 |
| best direction applied to a 19,750-row test set | 0.9468 ± 0.0015 | – |
| expected Bayes error on the test rows | 0.311 % ± 0.021 % (60.6 Bayes errors) | 0.326 % (R922) |
| TRUE count 0.1 < p < 0.9 | 184.4 ± 14.1 (161–208) | – |
| refitted-posterior count 0.1 < p < 0.9 (refit − true: −0.6 ± 3.8) | 183.9 ± 14.0 | **189** (68th percentile) |
| refitted count 0.01 < p < 0.99 | 529 ± 23 | 554 (86th percentile) |
| shrinkage LDA on 250 train rows (high-variance block) | 0.894 ± 0.006 | – |
| L2 logistic regression (C = 1), same | 0.912 ± 0.007 | – |

#### 3.3 Boundary-row definitions on the real test rows (`r422_real_boundary_counts.csv`)

| Definition | 0.1–0.9 | 0.05–0.95 | 0.01–0.99 |
|---|---|---|---|
| converged posterior (R922's fit) | 189 | 286 | 554 |
| S003 mixture, single fit on 250 labels (r321) | 203 | 312 | 10,092 |
| S003's submitted fold-mean score (r932) | 204 | 319 | 10,098 |

S003's score is R × smoothed component rates. Its range is [0.0061, 0.9864], and 6,416 rows tie at the maximum. It is a monotone, compressed version of the posterior, not a calibrated probability. Counts of S003 scores in a probability band are therefore a different quantity. The (0.1, 0.9) band is only slightly affected (203–204 against 189); the (0.01, 0.99) band is meaningless for S003.

**Verdict:**
- R154's ceiling of 0.947 is confirmed, with honest uncertainty ±0.0015.
- The Bayes error is 0.31–0.33 %.
- The "~186 boundary rows" is a population expectation (182–186). Any single test set varies by ±14; the real one has 189.
- 250-row linear models reach 0.89–0.91 on the replicas, which is consistent with gen.md's statement that linear models stall at 0.85–0.90.

### 4. Problems found / corrections (ranked)

1. **B006's point prediction is slightly optimistic.**
   - Registered: 0.9988. Model-based: 0.99868 (replicas, ±2.4e-5) and 0.99852 (real features).
   - The registered 80 % interval still has 0.79–0.85 probability, so this is not a miscalibration. Use 0.9985–0.9987 as the reference when the outcome is recorded.
   - The source of the 0.9988 was r936's ABC, which had only 125 accepted public subsets (MC SE 8e-5). R426 shows that the drop is MC noise.
2. **r936 (and verify_cand) weighted the noise levels with R156's truncated grid.** On the extended grid with R922's likelihood:
   - S003's private expectation is 0.99683, not 0.99775.
   - The second-selection gains are r209 +2.0e-5 and r201 +7.0e-6, not +1.45e-5 and +4.1e-6. D010's Context quotes the r936 values; the decision is not affected.
   - The paired differences, P(beat S003) and the ranking are unchanged.
3. **Three boundary-row counts circulate:**
   - 186: R154, a population expectation;
   - 189: converged posterior on the real rows;
   - 203 / 204: S003's own score, r321 / r932.

   `reports/extensions/e2_learning_curve.md:14` and `:61` call the 203 "a posterior between 0.1 and 0.9"; it is S003's mixture score. Suggested text is in the Chinese summary, item 2.
4. **verify_gen's open item "a stratified public split was not checked" is closed.** It changes nothing (§2).

### 5. Not verified / limits

- The replica ABC inherits the parametric replica. r936_cv_calib found it somewhat harder for supervised QDA than the real data. The real-feature model draws labels from a posterior that ranks like S003. Both may overstate r201's public loss (§1.5).
- Level binning maps each fine f to the nearest of 24 levels. The bins are ±0.0005 up to 0.02 and ±0.0025 above. The resulting error in S003's mean is < 1e-5, measured by the agreement with R922's 0.0005-step grid.
- Only the tn0 design (train labels clean) was extended. This is justified by R922: none of the 250 train labels contradicts a confident posterior. A stratified split was modelled only on the observed label, with n = 395. Other public sizes and other stratification variables were not modelled.
- Not pursued: on csv scores, untied Bayes log-odds trail S003 by up to 3e-5 at f ≥ 0.014 (≤ 1.6 MC SE; §1.2).
- `research/r424_r420_pooled.npz` (3.0 MB, packed copy of the r420 arrays and of the R922/R421 likelihood tables, with sha256 of every source in its manifest) travels in every code bundle. It can be deleted once r425/r426 are frozen; I did not delete it.

### 6. Suggested registry text (for the main session; not written by me)

- `registry/beliefs.md`, B006 / B007 Outcome, after the upload: compare the public score with R425/R421. Expected values are r201 0.9985–0.9987 (80 % about 0.9977–0.9994) and r209 0.9991 (80 % 0.9988–0.9994).
- `registry/journal.md` (new entry): "r2_noise（R420–R426）：flip_y 网格扩展到 0.03，用 R922 平坦先验重新加权后，S003 private 预期 0.9968（90% 0.992–0.9996），与 B008 一致；第二选择收益 r209 +2.0e-5、r201 +7.0e-6，D010 不变；S005/S006 条件 public 预测 0.9985–0.9987 / 0.9991；分层 split 无影响；R154 线性上限 0.947±0.0015、边界行约 186（真实 189）成立。"
