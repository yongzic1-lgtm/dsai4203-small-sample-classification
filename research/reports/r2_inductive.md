# r2_inductive：严格归纳的最终选择（D010 第 3 种情形）与 D009 证据 (2) 的复核

- 作者：AI 研究代理 "r2_inductive"（Claude Code，Claude Opus 5.5），2026-10-22 18:00–19:50 JST。
- 计算：全部在 Colab（账号 pro2，会话 r2ind，High-RAM 8 核）。会话 r2ind2 因账号会话数已满被拒绝分配，所以所有作业都在 r2ind 上顺序运行；结束后 r2ind 已关闭（r2ind2 从未建立）。
- 没有上传 Kaggle，没有改 registry/*.md、registry/*.csv（除脚本自动写入的行），没有删除文件。
- 新代码：`research/r400_inductive_lib.py`（库）、`r401_inductive_official.py`、`r402_inductive_replicas.py`、`r403_pack_r402.py`（纯打包）、`r404_inductive_pooled.py`。

---

## 中文摘要（给队友）

**一句话**：如果讲师禁止使用测试集特征，两个最终选择应为 **扩展网格 QDA（`20261022-095943_r930_qda_block_nll_ext`）+ r201（`20261022-071234_r201_qda_block`，即 S005）**。前者在 222 个复刻上比 r201 高 **1.94e-4**（MC SE 8.8e-6；91–93% 的复刻-划分里更高），是所有严格归纳候选中最好的；r201 作为搭档几乎不增加期望值（+4.8e-6），但在训练标签有错时能兜底。这个组合的 private 期望只比 S003 低约 **6e-5**。要执行这个方案，扩展网格 QDA 需要上传一次（预期 public 0.9991，80% 区间 0.9986–0.9995）。D009 的证据 (2) 在 222 个复刻上成立，而且数字和 r254 几乎一样。

**(a) 严格归纳候选的比较**（r933 的复刻设计：只给测试标签加噪声，按提交 CSV 的 8 位小数打分，16 次 public/private 划分；`reports/tables/r404_private.csv`）
- 与 r201 的配对差（R156 权重；括号是 MC SE；P = 在 private 上高于 r201 的概率）：
  - 扩展网格 QDA（r930）：**+1.94e-4**（8.8e-6），P 0.93
  - r210：+8.8e-5（7.8e-6），P 0.79
  - r201 + BQDA-trainonly 概率平均（新）：+6e-6（8e-6），P 0.55，等于 r201
  - r255_bqda_eb_trainonly：−9.6e-5（2.0e-5），P 0.37
  - 宽 kappa 网格的 BQDA-trainonly（新）：−8.7e-5（1.8e-5），P 0.36
  - 训练折内的 QDA-LW（r208）：−9.8e-3，完全不行
- 换成 R922 的平坦先验权重（flip_y 扩展到 0.03）：顺序不变，每个差的变化都在 1e-5 以内。各噪声水平（0–0.03）上的差几乎不变。
- private 期望值（R156 / 平坦先验）：S003 0.99775 / 0.99680；扩展网格 QDA 0.99769 / 0.99674；r210 0.99758 / 0.99663；r201 0.99749 / 0.99655。
- **两个都选归纳模型时**（Kaggle 取较高的一个）：最好的组合是扩展网格 QDA + 概率平均，比单选扩展网格 QDA 多 8.5e-6；扩展网格 QDA + r201 多 4.8e-6；+ r210 多 3.8e-6。差别只有 private 波动（约 3.3e-4）的 1/100 左右，所以选 r201 作搭档：它已经预登记（S005），而且对错标的训练行更稳（r936 的 tn1：扩展网格 QDA 单独 −4.7e-4，与 r201 配对后 −1.45e-4，均相对 S003）。
- D010 第 3 条里举的例子"r201 + r210"比推荐组合低约 1e-4；"r255_bqda_eb_trainonly"比 r201 还差，不推荐。
- **EB 的 kappa0 网格边界**：默认网格下 EB 在真实数据 100/100 折、复刻 22,200/22,200 折都选 kappa0 = 1.0（网格最大值）。网格放宽到 300 后，真实数据选 10（90 折）或 3（10 折），复刻上选 3 或 10 的占 99.9%。CV AUC 0.99919 vs 0.99903（配对 +0.00016，split SE 约 1e-4，在噪声内），复刻上的 private 差 < 1e-5。所以边界确实存在，但不影响结论。

**(b) D009 证据 (2)：边界行的排序**（222 个复刻，贝叶斯后验在 (0.1, 0.9) 的测试行，每个复刻约 185 行，干净标签；`reports/tables/r404_boundary.csv`）
- AUC：贝叶斯上限 0.806，**GMM（S003）0.804**，自训练 r209 0.718，扩展网格 QDA 0.698，贝叶斯 QDA（r255_bqda_eb_hv）0.678，r210 0.670，r201 0.664，BQDA-trainonly 0.652，QDA-LW 0.593。
- GMM 减贝叶斯 QDA = **0.126**（MC SE 0.002），**222/222 个复刻**里 GMM 都更好；对 r209、r201、r210 也都是 222/222。
- 按 r254 自己的定义（S003 分数在 (0.1, 0.9) 的行）：0.814 vs 0.686，而 r254 在 6 个复刻上得到 0.825 vs 0.702。方向和大小都对得上，可以把"6 个复刻"换成"222 个复刻"。

**(c) 核查**
- 与 r933 共用的 6 个流程在相同 seed、相同噪声水平上，private 和 public AUC 都与 r933 逐位相同（56/56 项，最大差 0）。
- 真实数据上，r201、r210、扩展网格 QDA、r209、S003、r208 的 test 预测与档案逐位相同；两个贝叶斯 QDA 最多差 7.8e-16，8 位小数的 CSV 一个值都不变（差异来自 BLAS 线程数）。

**建议（给主会话）**
1. D010 第 3 条（讲师禁止）具体化为：扩展网格 QDA（r930）+ r201。为了不在最后一刻才上传，建议提前上传一次 r930_qda_block_nll_ext，预期 public 0.9991（80% 0.9986–0.9995；条件于 S003 的 0.99923）。
2. D009 证据 (2) 的替换文字见英文 §6。
3. 宽 kappa BQDA 和概率平均的两个新 run 不需要上传。

---

## English

### 1. Question and design

**(a)** If the instructor forbids any use of the unlabeled test features (D010, switch 3), S002, S003 and every transductive candidate are out, and both final selections must be strictly inductive: each fold model sees only the 200 training rows of its outer fold (features and labels). Which two?

**(b)** D009's evidence (2) says that on replicas the GMM ranks the boundary rows much better than a Bayesian QDA (0.825 vs 0.702, r254). That rested on 6 replicas and on boundary rows chosen by the GMM's own scores. Re-check it on ≥ 200 replicas with a neutral definition.

Design: the r933 replica design that r936 used for the final selection ("tn0").
- 222 parametric replicas of the real data (`research/lib_gen_replica.py`), seeds 136–357. These are the same replicas, label flips and public subsets as r936's tn0 group, so every shared procedure must reproduce r933's numbers exactly (§4).
- Training labels clean, test labels flipped with sklearn flip_y semantics (nested across levels).
- Every procedure re-run end to end on each replica with the official 5×20 splits of the replica's labels; test score = mean of the 100 fold models.
- Scores formatted like the submission CSV (`"%.8f"`); 16 public/private splits per replica.
- Two weightings of the noise levels: R156 (0 / 0.0025 / 0.005 / 0.01 with 0.077 / 0.407 / 0.361 / 0.155) and the flat-prior posterior of R922, binned to the simulated levels 0 / 0.0025 / 0.005 / 0.0075 / 0.01 / 0.015 / 0.02 / 0.03 with weights 0.089 / 0.270 / 0.221 / 0.174 / 0.144 / 0.073 / 0.025 / 0.004 (`reports/tables/r404_weights.csv`).

Candidates (all strictly inductive unless marked):

| Label | Procedure | Archived official run | Real CV AUC |
|---|---|---|---|
| r201 | in-fold block → scaler → QDA, nested reg, inner AUC | `20261022-071234_r201_qda_block` (S005) | 0.99990 (4 folds < 1) |
| r210 | same, inner log-loss | `20261022-082047_r210_qda_block_nll` | 0.99998 (1) |
| ext | same, inner log-loss, grid extended to 1e-5 | `20261022-095943_r930_qda_block_nll_ext` | 0.99998 (1) |
| bqda_to | in-fold block → EB Bayesian QDA | `20261022-070202_r255_bqda_eb_trainonly` | 0.99903 |
| bqda_to_wk | same, EB kappa0 grid widened to 300 (new) | `20261022-185248_r401_bqda_eb_trainonly_widekappa` | 0.99919 (24) |
| lw | in-fold block → QDA-LW (eigen, shrinkage auto) | `20261022-075228_r208_qda_block_lw` | 0.98631 |
| ens | 0.5 × (r201 + bqda_to) probabilities (new) | `20261022-185255_r401_ens_r201_bqda_trainonly` | 0.99981 (9) |
| S003, r255_hv, r209 | transductive references (use test features) | `20261022-050708_r003_gmm2_hv`, `…070200_r255_bqda_eb_hv`, `…080632_r209_selftrain_qda_lw` | 1.0 |

Paired CV checks of the two new runs (`scripts/compare_runs.py`): bqda_to_wk − bqda_to +0.00016 (split SE ≈ 0.0001, corrected p 0.76); ens − r201 −0.0001 (split SE ≈ 0.0001, corrected p 0.77). Both are within split noise.

### 2. Results (a): strictly inductive candidates

**Table A1.** Private AUC on 222 replicas, CSV scores (`reports/tables/r404_private.csv`). "Diff" = paired difference vs r201, mean over replicas × draws, with the Monte-Carlo SE over replicas. "P(beat)" = share of replica-draws in which the candidate's private AUC is higher than r201's (ties count as not beating).

| Candidate | E[private] R156 | E[private] flat | Diff vs r201, R156 (MC SE) | Diff vs r201, flat (MC SE) | P(beat r201), R156 | Diff vs S003, R156 |
|---|---|---|---|---|---|---|
| ext (r930) | 0.997686 | 0.996739 | **+1.94e-4** (8.8e-6) | +1.92e-4 (9.3e-6) | 0.93 | −6.3e-5 |
| r210 | 0.997581 | 0.996634 | +8.8e-5 (7.8e-6) | +8.8e-5 (8.1e-6) | 0.79 | −1.68e-4 |
| ens | 0.997498 | 0.996550 | +6.1e-6 (8.1e-6) | +3.4e-6 (8.5e-6) | 0.55 | −2.51e-4 |
| r201 | 0.997492 | 0.996547 | 0 | 0 | – | −2.57e-4 |
| bqda_to_wk | 0.997405 | 0.996455 | −8.7e-5 (1.8e-5) | −9.2e-5 (1.8e-5) | 0.36 | −3.44e-4 |
| bqda_to | 0.997396 | 0.996446 | −9.6e-5 (2.0e-5) | −1.00e-4 (2.0e-5) | 0.37 | −3.53e-4 |
| lw | 0.987692 | 0.986762 | −9.8e-3 (1.6e-4) | −9.8e-3 (1.6e-4) | 0.00 | −1.01e-2 |
| *S003 (transductive)* | 0.997749 | 0.996802 | +2.57e-4 (1.1e-5) | +2.55e-4 (1.1e-5) | 0.94 | 0 |
| *Bayes posterior* | 0.997752 | 0.996802 | +2.60e-4 | +2.56e-4 | 0.95 | +3e-6 |

The variance-reduced version (labels integrated out under the noisy Bayes posterior, column `expected_diff_vs_r201`) gives the same picture: ext +1.85e-4 (6.0e-6), r210 +8.1e-5 (5.9e-6), ens −2.6e-6 (6.0e-6), bqda_to −1.12e-4 (1.8e-5). The private-AUC sd across replicas is 3.1–4.1e-4 under R156 (`private_auc_sd_draw0`), so every gap among the QDA variants is below one private sd.

**Flat in flip_y.** Differences vs r201 by level (`reports/tables/r404_by_level.csv`), flip_y 0 → 0.03: ext 1.86e-4 → 1.86e-4 (range 1.82–1.96e-4); r210 8.2e-5 → 8.2e-5 (7.7–9.2e-5); ens −2.9e-6 → −2.9e-5; bqda_to −1.13e-4 → −1.42e-4. Only the two weakest variants drift at high noise. The R156 prior and the flat prior therefore give the same ranking. That is why they agree.

**Table A2.** Both final selections strictly inductive; Kaggle scores the better one (`reports/tables/r404_pairs.csv`, R156; flat in brackets).

| Pair | E[max] − E[r201] (MC SE) | E[max] − better member alone (MC SE) |
|---|---|---|
| ext + ens | +2.03e-4 (8.3e-6) [+2.05e-4] | +8.5e-6 (1.7e-6) [+1.24e-5] |
| ext + bqda_to | +2.01e-4 (8.7e-6) [+2.01e-4] | +6.5e-6 (1.4e-6) [+9.2e-6] |
| **ext + r201** | +1.99e-4 (8.3e-6) [+2.00e-4] | +4.8e-6 (1.1e-6) [+7.9e-6] |
| ext + r210 | +1.98e-4 (8.6e-6) [+1.98e-4] | +3.8e-6 (0.8e-6) [+6.1e-6] |
| r210 + ens | +1.09e-4 (7.2e-6) [+1.13e-4] | +2.1e-5 (2.6e-6) |
| r201 + r210 (D010's example) | +1.00e-4 (7.0e-6) [+1.03e-4] | +1.2e-5 (1.7e-6) |
| r201 + bqda_to | +5.4e-5 (6.5e-6) [+5.6e-5] | +5.4e-5 (6.5e-6) |

The first member decides almost everything: any pair with ext is within 5e-6 of the best pair. The expected private AUC of ext + r201 is 0.997691 (R156) / 0.996747 (flat), i.e. 5.8e-5 / 5.5e-5 below S003 alone.

**Choice of the second member.** ext + ens is the best pair, but only by 3.7e-6 (R156) / 4.5e-6 (flat) over ext + r201, about 1/100 of the private sd. Three facts favour r201 as the partner:
1. r201 is already pre-registered as S005 and has a fresh-VM bit-exact re-run (R910).
2. ext is tuned by log-loss with reg near the rank-check floor (1e-4 in 80% of replica folds, `reports/tables/r404_reg_choice.csv`). In r936's sensitivity group with flipped training labels (tn1, raw scores, 136 replicas), ext alone was −4.72e-4 vs S003, and ext + r201 was −1.45e-4 (`reports/tables/r936_private.csv`, `r936_pairs.csv`). r201 insures against that failure mode, which the clean-train design (tn0) does not show. The real training labels show no confident contradiction (verify C10), so tn0 remains the primary design.
3. ens has no record under training-label noise, and an extra upload buys less than 5e-6.

**EB grid edge of r255_bqda_eb_trainonly.** With R255's grid (kappa0 ≤ 1) EB picks kappa0 = 1.0 in 100/100 real folds and in 22,200/22,200 replica folds. With the grid widened to 300 it picks 10 (90 folds) or 3 (10) on the real data, and 3 (48%) or 10 (52%) on the replicas; 30 occurs in 0.07%, 300 never (`reports/tables/r401_eb_kappa_frequency.csv`, `r404_eb_kappa.csv`). The real CV changes by +0.00016 (within split noise) and the replica private AUC by +9e-6. The edge is real, but nothing depends on it.

**Public-LB predictive** if ext is uploaded, conditional on S003's real public 0.99923 (ABC as in r936, ±1.5e-4 window, 125 accepted replica-draws from 68 replicas; `reports/tables/r404_public_abc.csv`): mean 0.99910, predictive sd 0.00042, 10–90% [0.99862, 0.99950], P(public > S003) 0.40, P(equal) 0.02. For r201 the same table gives 0.99884, sd 0.00066, matching r936.

### 3. Results (b): boundary rows (D009 evidence 2)

**Table B1.** Rows whose true Bayes posterior is in (0.1, 0.9), all 19,750 test rows of each replica (mean 185 rows), AUC against the clean labels; 222 replicas (`reports/tables/r404_boundary.csv`). "S003 − X" is the paired difference with its MC SE; "S003 better" is the share of replicas in which S003's boundary AUC is higher.

| Procedure | Boundary AUC (MC SE) | S003 − X (MC SE) | S003 better | Kendall τ with posterior | r254 definition (S003-boundary rows) |
|---|---|---|---|---|---|
| Bayes posterior (ceiling) | 0.806 (0.002) | −0.0016 (0.0003) | 0.36 | 1 | 0.816 |
| **S003 GMM** | **0.804** (0.002) | 0 | – | 0.93 | **0.814** |
| r209 self-training | 0.718 (0.003) | 0.086 (0.002) | 1.00 | 0.53 | 0.727 |
| ext QDA | 0.698 (0.003) | 0.107 (0.002) | 1.00 | 0.46 | 0.707 |
| r255_hv Bayesian QDA | 0.678 (0.003) | **0.126** (0.002) | 1.00 | 0.41 | **0.686** |
| r210 QDA | 0.670 (0.003) | 0.135 (0.003) | 1.00 | 0.39 | 0.677 |
| r201 QDA | 0.664 (0.003) | 0.140 (0.003) | 1.00 | 0.37 | 0.671 |
| ens | 0.664 (0.003) | 0.140 (0.003) | 1.00 | 0.37 | 0.671 |
| bqda_to | 0.652 (0.003) | 0.153 (0.003) | 1.00 | 0.34 | 0.659 |
| lw | 0.593 (0.003) | 0.211 (0.003) | 1.00 | 0.21 | 0.597 |

The expected AUC under the Bayes posterior (less noisy) gives the same values to ±0.002; S003 is better in 222/222 replicas there too. On r254's own definition the GMM-vs-BQDA gap is 0.814 − 0.686 = 0.128, against r254's 0.825 − 0.702 = 0.123 on 6 replicas. The claim holds in sign and size. These ~185 rows are about 1% of the test set, and the 0.09–0.14 boundary-AUC gaps are what produce the 0.6–3.5e-4 private-AUC gaps to S003 in Table A1.

### 4. Checks

- **r402 = r933.** For S003, r255_hv, r209, r201, r210 and ext, at flip_y 0 / 0.0025 / 0.005 / 0.01, on all 222 shared seeds and 16 draws, the CSV-score private and public AUCs equal r933's exactly (56/56 rows, max |diff| 0; `reports/tables/r404_check_vs_r933.csv`).
- **Real-data reproduction** (`reports/tables/r404_real_check.csv`). The single-thread re-implementation reproduces the archived test_pred of S003, r209, r201, r210, ext (r930) and r208 bit-identically. r255_hv and r255_trainonly differ by at most 7.8e-16, and no CSV value changes. The official ens run equals 0.5 × (r201 + bqda_to) to 1.3e-15, and the wide-grid official run equals its re-implementation to 1.1e-16 (`20261022-185320_r401_inductive_official_checks`). The same recomputation in a multi-threaded main process is not bit-identical for r201 (sha16 differs), so BLAS threading changes the last bits.
- Hypotheses (written before running, in each run's notes.md): confirmed for ext, r210, lw, ens, the EB edge and both checks. Refuted on the small side for bqda_to (predicted −2e-4 to −1e-3, observed −9.6e-5). The boundary gaps of r209, r255_hv and the supervised QDAs were larger than predicted (0.09–0.14 vs 0.01–0.10), and the Bayes ceiling on these rows is 0.806, not 0.83.

### 5. Text for the report (paste-ready)

> **If test features are not allowed.** We also compared strictly inductive candidates, whose fold models see only the 200 training rows of each fold. We used the same 222 parametric replicas and noise model as for the main candidates, with scores rounded as in the submission file. The best was QDA on the in-fold high-variance block with the regularisation chosen by inner log-loss on a grid extended down to 1e-5. Its expected private AUC was 1.9·10⁻⁴ above the AUC-tuned version (Monte-Carlo SE 9·10⁻⁶) and 6·10⁻⁵ below the transductive GMM. A Bayesian QDA trained on the 200 rows lost 1·10⁻⁴; Ledoit–Wolf QDA lost 1·10⁻². The differences did not change with the label-noise rate (0–3%) or with the prior on it. In this setting we would select the extended-grid QDA and the AUC-tuned QDA. The second adds only 5·10⁻⁶ in expectation, but it is robust when training labels are wrong, which the log-loss-tuned model is not.

> **Where the unlabeled data help.** On the same replicas we scored each method on the ~185 test rows per replica whose true posterior lies between 0.1 and 0.9. The transductive GMM ranks these rows almost as well as the true posterior (AUC 0.804 vs 0.806). Self-training reaches 0.718, the Bayesian QDA 0.678, and the tuned supervised QDAs 0.66–0.70. The GMM was better than each of them in all 222 replicas. Cross-validation cannot see this (CV AUC 0.9999–1.0 for all of them), but it is the source of the 10⁻⁴-level private-AUC differences.

### 6. Corrections and proposals (for the owners; I edited none of these files)

1. `registry/decisions.md` D010 item 3 (main session, via a new decision or a Status note): make it concrete as "**ext-grid QDA `20261022-095943_r930_qda_block_nll_ext` + r201 `20261022-071234_r201_qda_block`**; evidence `reports/tables/r404_private.csv`, `r404_pairs.csv`". The current examples "r201 + r210" (≈ 1e-4 worse) and "r255_bqda_eb_trainonly" (worse than r201) should not be used.
2. Upload ext once, pre-registered, so that switch 3 needs no last-minute upload. Suggested belief line (next free B id): "r930_qda_block_nll_ext（严格归纳，扩展网格 log-loss QDA，CV 0.99998）的 public ≈ 0.9991，80% 落在 0.9986–0.9995（以 S003 的 0.99923 为条件，预测 sd 0.00042；r404_public_abc.csv）；高于 S003 的概率约 40%。分数不改变 D010。"
3. `registry/decisions.md` D009 evidence (2): replace "（0.825 vs 贝叶斯 QDA 0.702，r254）" by "在 222 个复刻上，贝叶斯后验在 (0.1, 0.9) 的边界行（每个复刻约 185 行）上，GMM 的 AUC 为 0.804（贝叶斯上限 0.806），贝叶斯 QDA 0.678，自训练 0.718，调好的监督 QDA 0.66–0.70；在 222/222 个复刻里 GMM 都更好（r404，`reports/tables/r404_boundary.csv`）。按 r254 的定义是 0.814 vs 0.686。" This removes item C12 ("unverified") of `research/reports/verify.md`.
4. `research/reports/fm.md` "Where unlabeled data still matter" (0.825 vs 0.702, 6 replicas): add "confirmed on 222 replicas: 0.814 vs 0.686 on the same definition (r404)".
5. `research/r403_r402_pooled.npz` (5.7 MB) travels in every code bundle, like `r935_r933_pooled.npz` (verify item 27). Delete both once r404 and r936 are frozen; copies are in `logs/code_bundles/`.

### 7. Limitations

- All gaps are model-based. They hold in the parametric replica world, which verify_gen confirms as the generator. r936 found the real CV "luckier" than the replicas for supervised QDA; the new candidates were not conditioned on that CV pattern (no `--cv-stats` here).
- ext is a post-hoc variant. Its grid was extended after r210's choice was seen to sit at the grid edge, so it is a second-level choice. Its real CV equals r210's, and on the replicas it is better, but its selection history must be stated in the report.
- Under flipped training labels (not simulated here for the new candidates) ext is fragile (r936 tn1). The recommended pair covers that case with r201.
- The ABC public prediction uses 125 accepted replica-draws from 68 replicas; its MC error is about 1e-4.

### 8. Runs and jobs

| Run | Job (pro2/r2ind) | Content |
|---|---|---|
| `20261022-182709_r402_inductive_replicas_s136-246` | J20261022-182845-pro2-r2ind (1503 s) | 111 replicas + real-data pass |
| `20261022-190554_r402_inductive_replicas_s247-357` | J20261022-190730-pro2-r2ind (2091 s) | 111 replicas |
| `20261022-185248_r401_bqda_eb_trainonly_widekappa` | J20261022-185427-pro2-r2ind (95 s) | official CV, experiment |
| `20261022-185255_r401_ens_r201_bqda_trainonly` | same job | official CV, experiment |
| `20261022-185320_r401_inductive_official_checks` | same job | EB choices, member checks |
| `20261022-194238_r404_inductive_pooled` | J20261022-194417-pro2-r2ind (18 s) | pooled analysis (input `research/r403_r402_pooled.npz`, sha256 4246ec03…) |

Tables: `reports/tables/r401_candidates.csv`, `r401_eb_kappa_frequency.csv`, `r404_{private,by_level,pairs,public_abc,boundary,eb_choice,eb_kappa,reg_choice,weights,check_vs_r933,real_check}.csv` (+ `r404_private.md`, `r404_pairs.md`, `r404_boundary.md`). Figure: `reports/figures/r404_inductive_boundary.png`. All runs record `git: no-commit, dirty` (no commit exists yet; ARCHIVE_RULES §9).
