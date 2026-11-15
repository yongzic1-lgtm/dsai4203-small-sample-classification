# verify_cand：最终候选的 private 预期、两个最终选择、D009 的措辞（R930–R936）

作者：AI 核查代理 "cand"（Claude Code），2026-10-22（JST）。计算全部在 Colab 上完成（账号 pro2，会话 vcand / vcand2，High-RAM 8 核；两个会话都已关闭）。
本报告接续上午被用量上限打断的那次核查，沿用了当时的脚本和 run（r930、r932、r933 s0-7），没有重做。

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要第 2 点和英文 §2 开头）。依据：`research/reports/verify_gen.md` §3（R922）、`research/reports/verify.md` §3 C1。

1. **S003 expected private AUC**（摘要第 2 点；§2）。Old: "S003 的 private 期望 0.9977" and "private mean (mixture) 0.99775", presented without the prior → New: 0.9977 under the R156 4-point grid prior, which is truncated at 0.01; 0.9968 [0.9923, 0.9996] under a flat prior (verify_gen R922); the candidate differences are unaffected, because the paired gaps are flat across flip_y. Sources: `reports/tables/r936_private.csv` (tn0, csv8, `weighted_r156`: S003 0.99775); `reports/tables/r922_posterior_by_prior.csv` (row `uniform_0_0.1`: mean 0.99684, q05 0.99230, q95 0.99959; row `gen_grid4`: 0.99777).

## 中文摘要（给队友）

**一句话**：最终选择 **S003 + r209（自训练 QDA-LW）**。第二个选择在期望上只多 **1.5e-5** AUC，大约是 private 分数本身波动（3e-4）的 1/20。所以第二个选谁基本无关紧要，但要选，就选一个和 GMM 方法互补、复刻上对冲价值最大的。为此要**上传 1 次 r209**，预期 public 0.9991 ± 0.0003。D009 的"不可区分"在字面上没错，但没有信息量：CV 在这个分数段根本没有分辨能力，措辞要改。

**1. D009（调好的监督 QDA 与 GMM 在 CV 上"不可区分"）**
- **嵌套 CV 是诚实的**。我独立重写的代码逐位复现了 r201 / r210 的 test 预测、100/100 折的 reg 选择，以及真实数据上 8 个候选全部 100 折的 CV AUC（最大差 ≤ 1.1e-16，两台 VM 各算一次，结果相同）。高方差块在每个 inner 训练折里重新求，不碰 test 特征。
- **reg 0.001 确实是网格边界造成的**。把网格往下扩展后，inner log-loss 选 1e-4（53 折）、3e-4（36 折）、1e-3（11 折）。其中 1e-4 又是一个边界：再小就过不了 sklearn 的秩检验。CV AUC 不变（都是 0.99998，只有 1 折低于 1.0）；OOF log-loss 从 0.0087 降到 0.0076，最优点在 3e-4（0.0071）。在复刻上，更小的 reg 让真实 AUC 和 S003 的差距从 1.7e-4 缩到 6e-5，但**差距仍然存在**（100% 的复刻里监督模型都更差）。而且训练标签一旦有错，小 reg 更脆弱，见下。所以 CV 层面的结论不变，"真实差距"层面的数字会变。
- **Winner's curse**：sup 这条线共试了 63 个候选、587 个配置（我按 `r249_comparison.csv` 的 n_configs 重新加总，算上 GMM 参照行是 588）。r210 的 inner 指标是看过 r201 之后才改的，属于第二层选择。选择感知的估计（R205）是 0.9995。AUC 已经饱和，所以这种偏差本身不大，但 CV 也就看不出来。
- **"不可区分"说得对吗？字面上对，但没有信息量。** 在 222 个复刻上，官方 5×20 CV 的校正 t 检验对每个候选**每一次都不显著**（222/222）；可这些复刻里，监督 QDA、BQDA、自训练的真实 AUC **每一次都比 S003 低**（100%）。也就是说，在 ≤ 3e-4 的差距上，这个 CV 的检验功效是 0。真实数据上还有一个单向的迹象：100 折里监督 QDA 从来没有比 GMM 好过（0 胜，1–4 负）。
- **建议改写 D009**："250 个标签下，调好正则的严格监督 QDA 的 CV 也饱和在 0.9999–1.0000，所以 CV 分不出它和 GMM（配对差的 95% 区间约 ±2–5e-4；100 折中 0 胜 1–4 负）。这说明的是 CV 分辨率不够，不是两者一样好：在复刻数据上，同样的 CV 检验 222/222 次不显著，而监督 QDA 的真实 AUC 每次都更低（log-loss 调参低 1.7e-4，AUC 调参低 2.6e-4）。"

**2. 各候选在 private 榜上的联合分布**（`20261022-172146_r936_final_selection`，222 个参数复刻）
- 设定：只给 test 标签加噪声，训练标签保持干净。理由是真实训练标签没有任何高置信矛盾，GMM 的 CV 100/100 折都是 1.0。分数按提交 CSV 的 8 位小数处理；噪声水平 flip_y = 0 / 0.0025 / 0.005 / 0.01，按 R156 的后验加权。
- **S003 的 private 期望 0.9977**（这是 R156 的 4 点网格先验下的值，网格在 0.01 截断），预测 sd 0.0015，主要来自"标签噪声到底有多大"这个不确定性。各噪声水平下分别是 0.99995 / 0.9987 / 0.9974 / 0.9949。改用 flip_y 的均匀先验时是 0.9968，90% 区间 [0.9923, 0.9996]（verify_gen R922）。候选之间的差在各噪声水平上几乎不变，所以不受先验影响。
- 与 S003 的配对差（括号里是 MC SE），以及"在 private 上赢 S003 的概率"：
  - GMM 同族（r105 / r155 / r102）：−1e-6（1.4e-6），P(赢) 0.45–0.50。和 S003 实质相同，只是并列行的先后不同，所以像抛硬币。
  - 自训练 r209：**−6.9e-5（5e-6）**，P(赢) 0.21
  - 贝叶斯 QDA r255：−1.16e-4（7e-6），P(赢) 0.18
  - 监督 QDA r210：−1.68e-4（7e-6），P(赢) 0.08；r201：−2.57e-4（1.1e-5），P(赢) 0.06
- 这些差距在 4 个噪声水平上几乎不变，所以结论不依赖噪声先验。
- **相关性**：同一噪声水平下，各候选的 private AUC 高度相关：GMM 同族 0.998，r209 0.97，r255 / r210 0.95，r201 0.89–0.92，因为它们面对的是同一批错标签。它们和 S003 的差，sd 只有 2e-5（GMM 同族）到 1–2e-4（其他）。所以对冲的空间很小。
- **训练标签也有错的情况**（136 个复刻，作为敏感性分析）：r210 平均掉到 −6.3e-4；在 flip 0.01、而且有训练行被翻转的复刻里，平均是 −1.9e-3。log-loss 把 reg 推到网格边界，这样的模型很脆弱。r201、BQDA 和自训练基本不受影响。

**3. 两个最终选择，以及要不要再用上传次数**
- Kaggle 取两个选择中 private 较高的那个。"S003 + X"比只选 S003 多出的期望 AUC 如下：r209 **+1.45e-5**（±1.9e-6），r255 +1.37e-5，GMM 同族 +0.7–0.8e-5，监督 QDA +0.4–0.5e-5。r209 和 r255 的差在误差范围内。按真实数据的 CV 情况做条件后，r209 仍然最好或并列最好。
- **推荐：S003 + r209_selftrain_qda_lw**（run `20261022-080632_r209_selftrain_qda_lw`）。它是半监督 QDA，与 GMM 在边界行上的排序不同（Spearman 0.70），符合 D008"两个提交方法互补"的要求。r255 是等价的备选。
- **要上传 r209**，否则它不能被选为最终提交。在 S003 的 public 为 0.99923 的条件下，r209 的预期 public 是 **0.99913**（预测 sd 0.00033，MC SE 0.00004）；P(public > 0.99923) ≈ 0.25–0.28。建议预先承诺：public 高低都不改变选择，因为 395 行分不出这么小的差。只有 public 低于约 0.9975（偏离 4 个 sd 以上）才说明文件有问题。
- **"诚实监督 QDA"（r201）上传**：只对报告叙事有用，不应选为最终提交（对冲价值最低）。预期 public **0.99884**（预测 sd 0.00066，MC SE 0.00008），P(高于 S003) ≈ 0.16–0.20。public 和 S003 之差的 sd（6.5e-4）大于真实差距，所以它的 public 分数**不能**用来证明什么。上传次数很宽裕，可以传，但优先级低。
- **不要上传 r105 / r155 / r102**：87–95% 的情况下 public 与 S003 完全相同，对冲价值也只有 r209 的一半。

**4. 我自己之前写错、这次更正的地方**
- 上午的 r932 写过"r102 是一个免费的彩票式对冲，E[max(0, diff)] = 3.9e-5"。这是用全精度分数算的。提交 CSV 只保留 8 位小数，大量高置信行变成并列（S003 从 7,503 个不同分数变成 2,897 个，r102 从 14,414 个变成约 2,900 个）。按 CSV 分数算，r102 的对冲只有 **6.8e-6**，已在 r932 的 Conclusion 后面补了更正。

**5. 其他发现**
- 复刻上的 CV 普遍比真实 CV 差。复刻里折 AUC < 1 的折数不超过真实值的概率：S003 0.73，r255 0.26，r210 0.21，r201 0.08。说明真实训练集比较"幸运"（边界行少），或者参数复刻对监督 QDA 略难。只保留 CV 情况与真实数据一致的复刻（13 个）后，r210 仍差 −1.1e-4（±3e-5），r201 −2.7e-4，r209 约为 0（±3e-5）。排序不变。
- 两个 seed 区间的复刻 CV 折数分布不同（Mann-Whitney p ≈ 0.005）。我在另一台 VM 上重跑了 8 个 seed，结果逐位相同，所以这是抽样波动，不是机器造成的。
- fm 报告说 r255 的"log-loss 最低"，这里有口径问题：r255 的 0.0094 是逐折平均的 CV log-loss，r209 的 OOF log-loss 是 0.0071，更低。

---

## Details (English)

### Runs and Colab jobs (all exit 0, merged into the real archive)

| Run | Job (pro2) | What |
|---|---|---|
| `20261022-092058_r932_real_repro_agreement` | J20261022-092236-pro2-vcand | r931 re-implementations reproduce every archived candidate (test_pred, nested reg choices); pairwise agreement; plug-in private diff (+ correction appended 17:5x JST) |
| `20261022-0959*/1000*_r930_*` (9 runs) | J20261022-100121-pro2-vcand | r210 reg-grid edge: extended grid (log-loss / AUC tuning), fixed-reg profile 3e-3 … 1e-5, paired vs GMM |
| `20261022-094423_r933_replicas_param_s0-7` | J20261022-094559-pro2-vcand | 8 replicas, train+test label noise, raw scores |
| `20261022-160545_r933_replicas_param_s8-71` | J20261022-160720-pro2-vcand | 64 replicas, same design |
| `20261022-161642_r933_replicas_param_s72-135` | J20261022-161820-pro2-vcand2 | 64 replicas, same design |
| `20261022-165947_r933_replicas_param_s136-246_tn0_r8_cv` | J20261022-170123-pro2-vcand | 111 replicas, test-label noise only, raw + CSV-formatted scores, replica CV fold AUCs, real-data CV recomputed |
| `20261022-165009_r933_replicas_param_s247-357_tn0_r8_cv` | J20261022-165146-pro2-vcand2 | 111 replicas, same design |
| `20261022-172258_r933_replicas_param_s136-143_tn0_r8_cv` | J20261022-172435-pro2-vcand2 | cross-VM determinism re-run (bit-identical to the same seeds of s136-246; not pooled) |
| `20261022-172146_r936_final_selection` | J20261022-172324-pro2-vcand | pooled analysis (input `research/r935_r933_pooled.npz`, sha256 30256fb929f66d14…, packed by `research/r935_pack_r933.py` from the 5 runs above) |

Code: `research/r930_qda_reg_edge.py`, `r931_cand_lib.py` (array-level re-implementations; this session added validation-fold AUC collection, `cv_out=`), `r932_real_repro_agreement.py`, `r933_replica_candidates.py` (this session added `--train-noise 0`, `--round-decimals 8`, `--cv-stats`; defaults unchanged), `r934_replica_summary.py` (superseded by r936; never run), `r935_pack_r933.py`, `r936_final_selection.py`. Tables: `reports/tables/r936_{private,pairs,corr,public_abc,choice,cv_calib,cv_cond,cv_power}.csv`; figure `reports/figures/r936_private_diff.png`. Every run's notes.md has a Conclusion marked (cand).

Why two replica designs. The real training labels contain no confident contradiction: the GMM has fold AUC 1.0 in 100/100 folds, and verify_gen found no train row with P(observed label) < 0.2. R156's flip_y posterior already conditions on that. Mixing in replicas whose training labels were flipped therefore double-counts noise, and it penalises exactly the procedures that are fragile to it. The primary design (tn0) flips test labels only, which samples the train-clean conditional at every level. The tn1 design (train + test noise) is kept as a sensitivity analysis, and its train-clean subset agrees with tn0 (r210 −1.59e-4 vs −1.64e-4, r201 −2.71e-4 vs −2.53e-4, raw scores).

Why CSV formatting matters. `ssc.submission` writes `float_format="%.8f"`. On replicas this cuts S003's 7,592 distinct test scores to 2,896 and r102's 14,414 to 2,902. Expected AUC is unchanged (label noise is independent of the ranking among confident rows), but every tie-dependent quantity changes: P(beat), and the hedge value E[max(0, X − S003)].

### 1. D009: nested-CV honesty, grid edge, winner's curse, wording

| Check | Evidence | Verdict |
|---|---|---|
| r201 / r210 nested CV is honest (block re-derived per inner fold, no test features) | r931 re-implementation reproduces test_pred bit-identically and the reg choice in 100/100 folds (r932). The VM-recomputed real-data fold AUCs equal the archived fold_scores.csv for all 8 candidates (max \|diff\| ≤ 1.1e-16, on 2 VMs) | confirmed |
| r210 reg 0.001 in 100/100 folds is a grid-edge artefact | extended grid: inner log-loss picks 1e-4 (53) / 3e-4 (36) / 1e-3 (11); 1e-4 is again an edge (sklearn rank tol 1e-4 on the rank-12 block). OOF log-loss profile 0.0137 (3e-3), 0.0087 (1e-3), **0.0071 (3e-4)**, 0.0074 (1e-4), 0.0080 (3e-5), 0.0085 (1e-5) (`20261022-100052_r930_reg_edge_summary`) | confirmed |
| a lower reg changes the CV conclusion | CV AUC 0.99998 (1 fold < 1) for every reg in [1e-4, 3e-3], paired vs GMM −1.6e-5, corrected p 0.84 | no |
| a lower reg changes the true gap | replicas (tn0, CSV): r210 −1.68e-4 → extended grid −6.3e-5, still below S003 in 100% of replicas at flip 0; with train-label noise (tn1) the extended grid is as fragile as r210 (−4.7e-4) | yes, by ~1e-4, invisible to CV |
| winner's curse | sup track: 63 candidates, 587 configurations (re-summed from `reports/tables/r249_comparison.csv`: 588 including the GMM reference). r210's inner metric is a second-level choice made after r201; R205's selection-aware estimate is 0.9995 | bounded, ≤ 5e-4 on CV |
| "indistinguishable in CV" | real CV: GMM 1.0 in 100/100 folds; r210 0.999984 (0 folds better, 1 worse), r201 0.999903 (0 better, 4 worse); 95% intervals [−1.8e-4, +1.5e-4] and [−6.1e-4, +4.2e-4]. Replicas: the corrected t-test is non-significant in **222/222** replicas for every candidate, while the true clean-label gap is negative in **100%** of replicas for r255 / r209 / r210 / r201 (`r936_cv_power.csv`) | literally true, uninformative (zero power) |
| "best supervised r210 0.99998, r201 0.9999; r209 1.0; r255 1.0" | archived and recomputed CV identical | confirmed |
| "r255 … lowest log-loss" | r255 cv_logloss (fold mean) 0.0094; r209 oof_logloss 0.0071; extended-grid QDA 0.0076 | overstated (metric mismatch) |

Proposed wording (English, for D009 and the report): *"With 250 labels a tuned, strictly supervised QDA also saturates the official CV (AUC 0.9999–1.0000), so the CV cannot separate it from the transductive GMM. The 95% interval of the paired difference is about ±2–5e-4, and the supervised model is never better in any of the 100 folds (worse in 1–4). This reflects the resolution of the CV, not equal performance. On 222 parametric replicas the same test was non-significant every time, yet the supervised QDA's true AUC was lower in every replica: by 1.7e-4 with log-loss tuning and 2.6e-4 with AUC tuning. With mislabelled training rows the log-loss-tuned model loses 3–10 times more."*

### 2. Joint distribution of the private AUC (r936)

Primary design: tn0, CSV-formatted scores, 222 replicas, 16 public/private splits each. "Mixture" = R156 posterior over flip_y, a 4-point grid truncated at 0.01. Under it S003's private mean is 0.9977; a flat prior gives 0.9968 [0.9923, 0.9996] (verify_gen R922, `reports/tables/r922_posterior_by_prior.csv`). The candidate differences do not depend on this choice. MC SE uses the replica as the unit.

| Candidate | private mean (mixture) | predictive sd (mixture) | private mean at flip 0 / 0.0025 / 0.005 / 0.01 | sd at 0.005 | diff vs S003 (MC SE) | P(beat S003) | P(tie) | hedge E[max(0, X − S003)] (MC SE) |
|---|---|---|---|---|---|---|---|---|
| **S003** `r003_gmm2_hv` | 0.99775 | 0.00148 | 0.99995 / 0.99870 / 0.99744 / 0.99489 | 4.06e-4 | – | – | – | – |
| r105_gmm2_block_mle | 0.99775 | 0.00148 | same to 1e-6 | 4.06e-4 | −1.1e-6 (1.4e-6) | 0.50 | 0.005 | 7.9e-6 (0.8e-6) |
| r155_gmm2_inf_tight | 0.99775 | 0.00148 | same to 2e-6 | 4.07e-4 | −1.7e-6 (1.5e-6) | 0.46 | 0.006 | 8.0e-6 (0.8e-6) |
| r102_semisup_eps01 | 0.99775 | 0.00148 | same to 1e-6 | 4.06e-4 | −0.9e-6 (1.3e-6) | 0.45 | 0.024 | 6.8e-6 (0.7e-6) |
| r255_bqda_eb_hv | 0.99763 | 0.00148 | 0.99983 / 0.99858 / 0.99733 / 0.99477 | 4.14e-4 | −1.16e-4 (7.1e-6) | 0.18 | 0 | 1.37e-5 (1.9e-6) |
| **r209_selftrain_qda_lw** | 0.99768 | 0.00148 | 0.99989 / 0.99863 / 0.99737 / 0.99482 | 4.10e-4 | **−6.9e-5 (5.3e-6)** | **0.21** | 0 | **1.45e-5 (1.9e-6)** |
| r210_qda_block_nll | 0.99758 | 0.00148 | 0.99979 / 0.99853 / 0.99727 / 0.99472 | 4.20e-4 | −1.68e-4 (6.8e-6) | 0.08 | 0.0001 | 5.0e-6 (1.1e-6) |
| r201_qda_block | 0.99749 | 0.00149 | 0.99971 / 0.99844 / 0.99718 / 0.99463 | 4.42e-4 | −2.57e-4 (1.1e-5) | 0.06 | 0 | 4.1e-6 (0.8e-6) |
| (post hoc) extended-grid QDA | 0.99769 | 0.00148 | 0.99989 / 0.99864 / 0.99737 / 0.99482 | 4.16e-4 | −6.3e-5 (5.7e-6) | 0.21 | 0 | 1.26e-5 (1.9e-6) |
| (reference) Bayes posterior | 0.99775 | 0.00148 | 0.99995 / 0.99870 / 0.99744 / 0.99489 | 4.27e-4 | +3.0e-6 (7.6e-6) | 0.53 | 0.0004 | 5.4e-5 (4.3e-6) |

- The paired gaps are flat across noise levels. Clean-label gaps: −1.22e-4 (r255), −6.5e-5 (r209), −1.64e-4 (r210), −2.45e-4 (r201); at flip 0.01: −1.16e-4, −7.0e-5, −1.71e-4, −2.61e-4. So the ranking does not depend on the flip_y prior. That matters because verify_gen's likelihood supports flip_y up to ~0.015, beyond this grid. The hedge values grow with the noise level (at 0.01: r255 3.2e-5, r209 3.0e-5, GMM family 1.3–1.4e-5).
- Correlation with S003 within one noise level (replicas × draws): GMM family 0.997–0.998; r209 0.965–0.968; r255 0.94–0.95; r210 0.95; r201 0.89–0.92; Bayes 0.93–0.94. The sd of the difference to S003 at 0.0025 / 0.005 / 0.01 is 1.7–4.0e-5 for the GMM family, 0.78 / 1.04 / 1.43e-4 for r209, 1.03 / 1.32 / 1.87e-4 for r255 and 1.47 / 1.92 / 2.35e-4 for r201. Across the noise mixture all correlations exceed 0.99, because flip_y is shared. Only the non-GMM candidates can disagree with S003 materially, and when they do they mostly lose.
- Regret correlation (AUC minus Bayes AUC) with S003: GMM family 0.98–0.99, r209 0.71–0.73, r255 0.46–0.51, r201 0.36–0.39.
- Raw vs CSV scores: on full-precision scores r102's hedge is 3.6e-5 (2.7e-6); on CSV scores it is 6.8e-6. r932's "free lottery hedge 3.9e-5" is therefore an artefact of evaluating scores that Kaggle never sees (correction appended to r932's notes).
- Sensitivity (tn1, 136 replicas, train + test label noise, raw scores, R156 weights): r210 −6.27e-4 (9.6e-5), r201 −2.89e-4 (1.6e-5), r255 −1.51e-4, r209 −7.0e-5; when train rows are flipped, r210 falls to −8.5e-4 / −1.15e-3 / −1.86e-3 at 0.0025 / 0.005 / 0.01.
- CV conditioning (`r936_cv_cond.csv`). The real CV pattern (every transductive candidate perfect, r210 ≤ 1 and r201 ≤ 4 folds below 1.0) occurs in 13 of 222 replicas. In those, r209 −0.1e-6 (2.9e-5, hedge 5.2e-5), r255 −6.2e-5 (2.7e-5), r210 −1.09e-4 (3.2e-5), r201 −2.73e-4 (5.1e-5). Replica CV is worse than the real CV for all candidates (`r936_cv_calib.csv`: median folds < 1, replica vs real: S003 0 vs 0, r255 5 vs 0, r209 0 vs 0, r210 8 vs 1, r201 13 vs 4). The real training sample is either lucky or slightly easier than the parametric replica. The ranking survives conditioning.
- Determinism: seeds 136–143 re-run on the other VM are bit-identical (all arrays). The CV fold-count difference between seed ranges 136–246 and 247–357 (Mann-Whitney p 0.002–0.017, correlated across procedures) is sampling variation. Replica difficulty does not differ between the ranges (Bayes clean AUC p 0.91, boundary rows p 0.95).

### 3. Final selection and uploads

E[max(S003, X)] − E[S003] on the private LB (tn0, CSV, mixture; `r936_choice.csv`):

| Second selection X | gain (MC SE) | P(pair beats S003 alone) | needs an upload |
|---|---|---|---|
| **r209_selftrain_qda_lw** | **1.45e-5 (1.9e-6)** | 0.21 | yes |
| r255_bqda_eb_hv | 1.37e-5 (1.9e-6) | 0.18 | yes |
| (post hoc) extended-grid QDA | 1.26e-5 (1.9e-6) | 0.21 | yes (not recommended: post hoc, fragile to train noise) |
| r155 / r105 / r102 | 8.0e-6 / 7.9e-6 / 6.8e-6 | 0.45–0.50 | yes |
| r210 / r201 | 5.0e-6 / 4.1e-6 | 0.08 / 0.06 | yes |
| S002 (already uploaded, QDA reg 0.1) | ≈ 0 (loses ~7e-3, R153) | ≈ 0 | no |

Recommendation:
1. Final selections: **S003 + r209** (`20261022-080632_r209_selftrain_qda_lw`); r255 is an equivalent alternative (difference 8e-7, well inside the MC error). r209 is also the best or joint-best second choice under CV conditioning, and it matches D008's "methodologically complementary" rule: a semi-supervised QDA whose boundary-row ranking differs from the GMM's (Spearman 0.70 on S003's 204 boundary rows).
2. Upload r209 once; otherwise it cannot be selected. Expected public, given S003's public 0.99923 (ABC on tn0 CSV scores, window ±1.5e-4, 125 accepted replica-draws from 68 replicas, flip_y posterior 0.07 / 0.44 / 0.35 / 0.14): **0.99913, predictive sd 0.00033, MC SE 0.00004**; P(public > S003) 0.28, P(equal) 0.04. With a ±3e-4 window: 0.99911 (sd 0.00037, MC SE 0.00003). Suggested pre-registration: "public ≈ 0.9991 ± 0.0003; the choice of S003 + r209 does not depend on this score; a public below 0.9975 means a broken file".
3. Optional, for the report's journey only: r201 (`20261022-071234_r201_qda_block`, the pre-registered honest supervised QDA). Expected public **0.99884, predictive sd 0.00066, MC SE 0.00008**; P(> S003) 0.20 (±3e-4 window: 0.99869, sd 0.00076). The sd of its public difference to S003 (6.5e-4) exceeds the true gap (2.6e-4), so its public score cannot support any comparison. Do not select it as final. If a supervised QDA is uploaded, use r201 rather than r210: r210 adds a second-level selection, a grid-edge reg, and fragility to train-label noise.
4. Do not upload the GMM-family variants: their public equals S003's in 87–95% of replica-draws.

Caveats. (a) All of this is conditional on the parametric generator hypothesis (verify_gen confirms it). A failure mode outside the replica world is not quantified; only the non-GMM candidates hedge against it, which is one more reason to prefer r209 / r255 over a GMM variant. (b) The flip_y grid stops at 0.01; the gaps are flat in flip_y, and higher noise only raises the (still tiny) hedge values. (c) The ABC public predictions use the replica's public subsets, and the real public set is unknown. (d) S003's private expectation here (0.9977, predictive sd 0.0015, R156 prior) is the same computation as R156; verify_gen's flat-prior value is 0.9968 [0.9923, 0.9996]. The candidate differences above do not depend on that choice.

### Problems found (for the main session)

1. D009's "indistinguishable in CV" is true but has zero power at these effect sizes (222/222 non-significant while 100% of true gaps are negative). Replace it with the wording above.
2. r210's reg is a grid-edge choice. It costs ~1e-4 true AUC on replicas, which the CV cannot see. Tuning by log-loss also makes the model fragile to mislabelled training rows (−6.3e-4 on average with train-label noise).
3. r932 (this verifier, morning session) overstated r102's hedge value (3.9e-5 raw vs 6.8e-6 for the CSV scores Kaggle actually sees). Corrected in r932's notes.
4. "r255 has the lowest log-loss" compares a fold-mean CV log-loss (0.0094) with OOF log-losses; r209's OOF 0.0071 is lower.
5. `research/r935_r933_pooled.npz` (3.5 MB) sits in research/ so that the pooled analysis could run on Colab: remote.py syncs only code and data/raw. It travels in every later code bundle; delete it once the main session no longer needs to re-run r936 (it is archived in logs/code_bundles).
