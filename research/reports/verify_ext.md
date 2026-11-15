# verify_ext：四个扩展草稿（E1–E4）的对抗式核查（第二轮）

核查者：Claude Code（verifier "ext"），2026-10-22 16:00–16:40 JST。这是对 10:06 版本的重做。草稿从 07:51 以后没有改过，所以 10:06 版本的结论仍然针对现在的草稿；这一轮我把每个数字重新对了一遍，又补了一次计算。
计算都在 Colab 上跑（账号 pro1，会话 vext，High-RAM 8 核，结束后已 `down`）：
- 上一轮：R940 `20261022-091211_r940_e1_pairs_check`、R941 `20261022-092201_r941_e2_replica_truth`、R942 `20261022-094142_r942_e2_coverage`、R943 `20261022-100104_r943_e4_null_step`；
- 本轮新增：R944 `20261022-161150_r944_e3_replica_decomp`（job `J20261022-161330-pro1-vext`）。

**更正上一轮的一个结论**：10:06 版本说 E3 在小 k_feat 时的差距“主要来自 honest CV 的悲观，而不是 leaky 的乐观”，那是只看一份样本（GMM 代理标签）得出的。R944 用 100 份复制样本检验后发现，**平均而言这个说法不对**：平均下来小 k_feat 的差距本身很小，而且几乎全部来自泄漏带来的乐观。只有在我们这份样本里，k_feat = 2 的差距碰巧偏大（0.056，在复制样本里排第 96 百分位），这种偏大的情形才主要是 honest CV 偏低造成的。下面的 E3 部分已经按这个改写。

我没有改任何扩展文件、registry、src，只写了 `research/r940–r944_*.py` 和这份报告。

## 中文摘要（给队员）

**总体判断**：草稿里的数字绝大多数能在归档里逐一找到，E1 和 E4 的关键表格我在 Colab 上用原代码、原种子逐位复现了。没有发现数据泄漏，也没有违反比赛规则。问题有四类：
1. 四份都超过 2 页；
2. E2 外推区间远比标称的窄，这是最重要的科学问题；
3. E1 和 E4 的结论措辞比证据强；
4. E3 对小 k_feat 的解释在平均意义下成立，但没有说明我们这份样本在 k_feat = 2 时偏离了平均；另外公开榜分数已经出来了，草稿还写着 pending。

另外有 15 处数字和归档对不上，大多是四舍五入：E1 4 处，E2 3 处，E3 4 处，E4 4 处（见 Details §3）；另有几处措辞超出了证据。其中 3 处是上一轮漏掉的：E2 表 1 的 0.696 应为 0.695，0.786 应为 0.785；E4 表 2 的 ±0.019 应为 ±0.018。

**新情况**：S004（E3 最终模型）已经在 16:08 JST 提交，public 0.79136。和 leaky CV 0.867 比低了 0.075，和 honest CV 0.793 只差 0.002。所以 E3 brief 里“与 leaderboard 比较”这一项现在可以完整回答了，但草稿还写着 pending，`e3_leaderboard.csv` 也没刷新。需要作者跑 `extensions\e3_leakage\run.py --refresh-lb 20261022-063908_e3_leakage` 后改正文。

**页数**：我用 XeLaTeX 排了版，设置是 Calibri 11 pt、A4、2.54 cm 页边距，接近 Word 默认。图片占满版心宽度时，E1 2.66 页、E2 2.71 页、E3 2.22 页、E4 2.23 页；图片缩到 60% 时是 2.55 / 2.57 / 2.16 / 2.19 页。
- 正文字数（含标题，不含注释、表格、图注）是 868 / 936 / 828 / 902 词，图注另有 61 / 95 / 64 / 101 词。
- 按“每页约 1000 词正文加 1 张图或表”的粗略预算算：E1、E3 字数在内；E2 正文加图注 1031 词，压线；E4 有 3 个图表，多 1 个。
- 但实际排出来四份都超了。结论是每份要删 250–400 词，或者删掉一个图表，或者确认课程允许更小的字号。
- 四份的标题都正好是 Question / Protocol / Results / Conclusion。作者行都还是 TODO，目前这样是预期的。

**E1（验证方案）**：结论在它自己的定义下基本成立，但要写清楚三点。
- 选中的这一对模型，真实差距是多少？两个都不对。用 50 次 5×20 共 1000 个 repeat 合起来算，这 250 行上的 CV 差距是 **−0.0044 ± 0.0004**（R940）。5×50 reference 的 −0.0029 比它更吵，SE 实际约 0.0019，草稿写的 0.0014 偏小。官方种子上的 −0.0101 离均值 1.84 个 sd，草稿说“约 2 sd”略有夸大。
- 这会推翻结论吗？不会。MDD 衡量的是 sd(A−B)，也就是 split 噪声，和真实差距无关。R940 把 R300 的 8 个线性候选两两配对共 28 对：5×20 的 sd(A−B) 在 0.0018–0.0046 之间，E1 这一对 0.0031 大约在第 35 百分位。28 对里有 25 对 MDD 的整个 CI 低于 0.01。真实差 0.012–0.019 的几对，MDD 是 0.0063–0.0077。
- 但 **MDD = 2 sd 对应的检验功效只有约 50%**。换成常用的 80% 功效（2.8 sd），5×20 的 MDD 是 0.0086，CI [0.0071, 0.0107]，按 E1 自己的规则就是“不确定”，28 对里只有 4 对能过。反过来，如果问的是“排对顺序的概率”，5×10 对所有 28 对都 ≥ 0.93。所以“只有 5×20 可信”取决于用哪个标准，草稿要写明标准和它对应的功效。
- 换一份 250 行数据，这一对的差距 sd 是 0.0105，是 5×20 split 噪声的 3.4 倍。复制总体上的差距是 +0.0087，符号和这 250 行的 CV 相反（R941，基于模型，只作参考）。
- MDD 的算法是对的：50 个估计的 sd × 2，CI 用 49 自由度的 chi-square，各次执行的种子不重叠。我重算了 CI 端点，对得上。

**E2（学习曲线）**：外推方法本身常见，可以用，但区间不诚实。这是最严重的问题。
- 复制总体上，L1 基线在 500 行训练时的期望 AUC 是 **0.923**（R941）。草稿的点估计 0.906 低了 0.017，0.923 落在 90% 区间 [0.881, 0.916] 和“0.88–0.92”之外，只在 envelope 内。
- 覆盖率模拟（R942）：在 100 份新的 250 行复制样本上原样跑 E2 的流程，名义 90% 的区间只有 **21%** 覆盖真值；power law 点估计平均偏低 0.031，log-linear 平均偏高 0.017，envelope 覆盖 81%。
- 复制总体的校准：x ≥ 120 时和真实数据吻合（0.802/0.828/0.851 对 0.810/0.830/0.847），但 x = 80 时偏高 0.033（约 2.6 SE）。所以“0.923”这个具体数要当作模型给出的估计。“区间偏窄、点估计偏低”是流程本身的性质，不依赖这个具体数。
- 建议改成“约 0.91–0.92，诚实的不确定度约 ±0.04”，并删掉“back-test 支持 power law”的推论：单个样本上的 back-test 不能代表平均情况。
- “增益在每个 bootstrap 里都为正”是拟合边界（b ≥ 0）保证的，不提供信息。
- QDA 和 GMM 的结论成立。

**E3（信息泄漏）**：两条流程确实是 brief 要的那两种做法，原理部分也对。对 gap 随 k 变化的解释，平均意义下成立，但需要补充“我们这份样本偏离了平均”。
- 两条流程：leaky 用全部 250 行选一次特征，然后在 CV 里固定这些列；honest 把选择放进 Pipeline，每折重做；两者用同一组 split。k_feat = 150 时差距正好为 0，置换标签下 honest ≈ 0.50。
- R944 的结果（100 份复制样本，平均意义下），**草稿的机制基本成立**：
  - honest CV 对自己那套流程是无偏的（每个 k_feat 偏差的绝对值 ≤ 0.004）。
  - 差距几乎全部来自泄漏带来的乐观。k_feat = 2/5/7/10/20/50 时，平均差距 0.015/0.010/0.008/0.024/0.064/0.087，其中泄漏乐观 0.014/0.009/0.002/0.014/0.054/0.079。
  - 差距的峰值在 k_feat = 50（65% 的样本）或 30（25%）。
  - k_fold 方面，泄漏乐观在 2/5/10 折时几乎不变（0.054/0.054/0.053）；差距随折数变小，是因为 honest 用的训练行变多了（honest 比最终模型低 0.046/0.016/0.010）。这正好证实了草稿对 k_fold 的解释。
- **但对我们这份样本，解释不完整**：
  - 真实数据 k_feat = 2 的差距 0.056 在复制样本里排第 96 百分位（平均 0.015，sd 0.017）。在 8 份差距同样 ≥ 0.04 的复制样本里，泄漏乐观只有 +0.002，honest CV 却比它自己模型的真实表现低 0.037。这和真实数据代理标签上看到的模式一致（leaky − 代理 = +0.011，honest − 代理 = −0.045）。所以“winner's curse”平均能解释约 0.014，但解释不了这份样本的 0.056，剩下的部分是 honest CV 在这份样本上的估计噪声。
  - k_feat = 7 处差距 ≈ 0（0.001）也有运气成分：复制样本的中位数是 0.006，只有 44% 低于 0.005。小 k 处出现低谷是普遍现象（97% 的样本最低点落在 k_feat = 2/5/7），差距正好为 0 则不是。
- 公开榜（S004 = 0.79136）和 honest CV（0.793）几乎相等，比 leaky CV（0.867）低 0.075，约 3.6 个公开榜 SE；比 GMM 代理值 0.820 低 0.029，约 1.4 个 SE，仍在预先登记的 90% 区间 0.785–0.854 内。R944 的复制样本里，k_feat = 20 时 leaky 平均比最终模型高 0.048，honest 平均低 0.016，两者都和这个结果相容。
- 草稿还有几处不准：
  - “~48 个特征”是纯噪声下的期望；真实标签下有 61 个特征超过 1/√n。
  - 置换标签下 gap 在 k_feat = 20–30 是平台，不是“在 30 达到峰值”。
  - 代理标签上 k_feat = 10 最好（0.864），而 honest CV 选的是 7，草稿没提。

**E4（特征排序稳定性）**：B = 2000，满足 brief；drop 表逐项复现。
- “sharp drop”成立，但余量不大：观测到的最大跳 0.49 / 0.37 / 0.37 / 0.30，在 50 条置换曲线里每个 k 都只有 1 条超过（加 1 校正后 p ≈ 0.04）。
- 判定规则有问题：config 里预先登记的主规则是 **perm**（m* = 7/8/9/17），它也是唯一校准过的规则（噪声下误收 6–8%）。草稿的“位置不随 k 变：是”用的却是 gap 和 half 规则。half 规则在噪声下 k ≥ 10 时有 72–100% 的曲线会收进纯噪声特征。应该改成“7–8 个强特征在每个 k 都被选中；弱特征是否被收进来，取决于 k”。
- 另外两处措辞：“k ≥ 8”应写成“k = 10/13/20”，因为只测了 5/10/13/20；“6 个 block 特征从不稳定”在 perm 规则下不成立（67 在 k = 13/20、8 和 26 在 k = 20 都算稳定）。
- “稳定 ≠ 有用”有数字支持：三类模型上 block 都优于 stable 集（p = 0.018 / 0.0006 / 0.009）；6 个不稳定的 block 特征单独就让 QDA 达到 0.917；17 是 block 的精确线性组合。不过要补一句：对线性模型，stable 集确实有用（比全部 150 个特征高 +0.063，p = 0.038）。

**建议主会话做的事**（按顺序）：
1. 把 S004 的 public 0.79136 写进 E3：刷新 `e3_leaderboard.csv`，改正文的 Leaderboard 段和 Conclusion。
2. 让各扩展的负责人按 Details §3 改数字，按 §4 改措辞，然后各删 250–400 词。
3. E2 的置信度段落按 R942 重写。

---

## Details (English)

### 0. Method

- Re-read the four drafts, the briefs (`docs/ASSIGNMENT_CHECKLIST.md` §B2), extension code and configs, explorer reports `ext12.md` / `ext34.md`, and `registry/submissions.csv` / `beliefs.md`.
- Every number in each draft was compared at full precision with `reports/tables/*.csv` and `experiments/<run_id>/summary.json`, using local read-only pandas.
- Pages were measured by typesetting each draft with XeLaTeX (Calibri 11 pt, A4, 2.54 cm margins, tables as tabularx, figure at 100% and at 60% of the text width) and reading `\pagetotal` on the last page. Scratch files: `D:\tmp\claude\D--polyu-class-ml\c9dcd3b2-a841-4b45-b8a8-1aabc1690c51\scratchpad\m_1.0` and `...\m_0.6`.
- Colab runs (all `kind="analysis"`):
  - R940: all 28 pairs of the 8 linear R300 candidates on E1's own split seeds (50 × 5×20).
  - R941: parametric replica population (`lib_gen_replica.sample_parametric`), 100 draws, true learning curve and E1 pair population gap.
  - R942: E2's unmodified fitting functions on 100 replica 250-row samples (coverage).
  - R943: E4 bootstrap and null curves with E4's functions and seeds, plus null calibration of the drop rules.
  - **R944 (new)**: E3's two procedures on 100 replica 250-row samples with 20,000 labelled test rows each, decomposing the gap into leak optimism and honest-CV bias.

### 1. Format

| Draft | Prose words (incl. headings) | Captions | Tables / figures | Typeset pages (figure 100% / 60%) | Headings | Author line |
|---|---|---|---|---|---|---|
| E1 | 868 | 61 | 1 (10 rows × 8 cols, wraps heavily) / 1 | 2.66 / 2.55 | exact | `Author: TODO (Kaggle: TODO)` |
| E2 | 936 | 95 | 1 (7 × 6) / 1 | 2.71 / 2.57 | exact | `Author: TODO (Kaggle: TODO)` |
| E3 | 828 | 64 | 1 (11 × 7) / 1 | 2.22 / 2.16 | exact | `Author: TODO Full Name (Kaggle: TODO)` |
| E4 | 902 | 101 | 2 / 1 (1×4 panels) | 2.23 / 2.19 | exact | `Author: TODO Full Name (Kaggle: TODO)` |

- **Heuristic budget** (≈ 1,000 prose words plus one figure or table per page):
  - E1 (929 words incl. captions) and E3 (892) fit the word budget.
  - E2 (1,031) is at the limit.
  - E4 (1,003) has 3 visuals, one more than two pages allow.
- **Typeset**: all four overflow. Shrinking the figure to 60% saves only 0.05–0.15 pages.
- **Fixes**:
  - E1/E2 need about 400 fewer words, or the table cut to the 5–6 columns the argument uses. In E1, drop "Linear: std A" and "Fits per decision"; in E2, drop the [min, max] brackets.
  - E3/E4 need about 250 fewer words.
  - E4 should drop or merge one table. The Table 2 caption alone is 70 words.

### 2. Brief coverage

**E1**: all 5 items answered.
- Hold-out vs k = 3/5/10: Protocol "Schemes".
- Only the split seed changes, with a shuffle=False control: "Repetition" and the no-shuffle row.
- Model seed fixed: "model seeds stay fixed".
- Spread reported: Table 1 "std A" and the hold-out range 0.692–0.923.
- "Which scheme would you trust …": Conclusion, sentence 1.

**E2**: all 5 items answered.
- 50–250 rows with 10 subsets: Protocol "Subsets".
- Mean and spread: Table 1.
- Extrapolation: "AUC of about 0.91 with 500 training rows".
- Confidence: Conclusion ¶1. It is answered, but wrong in substance (§4).

**E3**:
- Filter, leaky, honest, "gap vs k and why", and both meanings of k: answered.
- "Compare … with the leaderboard score": **stale**. The draft says "pending", but S004 has public 0.79136 (registry/submissions.csv, submitted 2026-10-22T07:08:41Z). The answer needs to be written in.

**E4**: all 6 items answered.
- Several hundred resamples, no model: B = 2000.
- Frequency vs chance k/150: Table 1 and Figure 1.
- 2–3 values of k: four were run.
- Sharp drop and its dependence on k: Conclusion ¶1.
- Above-drop model vs all 150: "+0.063 honest".
- Stable vs worth keeping: Conclusion ¶2.

### 3. Traceability: mismatches (everything not listed matched the archive)

Each item gives the claim, the archived value and the correction. Items marked *new* were missed in the 10:06 round.

**E1** (e1_paired_difference.csv, e1_scheme_spread.csv, e1_qda_paired_difference.csv, r300_*.csv, summary.json reference). R940 reproduces sd 0.014857 / 0.004165 / 0.003056, mean −0.004428 and official-seed gap −0.010063.
1. Hold-out 30% MDD CI "[0.036, 0.053]": mdd_lo = 0.035497, so **0.035**.
2. "split noise is 4.5-6 times smaller": the ratios are 4.53 / 5.81 / 5.84 / 6.09 / 5.76, so **4.5–6.1**.
3. "a draw about 2 sd away": (−0.01006 + 0.00443) / 0.00306 = **1.84 sd**.
4. "the Nadeau-Bengio test … never finds the linear pair's difference significant": frac_nb_p_lt_0.05 = **0.02** for 3-fold and 5-fold (0 for 5×10 and 5×20). It should say "in at most 1 of 50 executions".
5. Reference "SE 0.0014" equals the archived 0.001421 but is understated. The reference's per-repeat sd is 0.0100, while E1's own executions average 0.0134 (= 0.00299 · √20). The SE is therefore about 0.0019, and `sign_reliable` (|diff| / SE = 2.01) is fragile.

**E2** (e2_*_table, e2_*_extrapolation, r304, r307, r308):
1. *New*: Table 1, n = 100, L1 min "0.696": cv_min = 0.695455, so **0.695**.
2. *New*: Table 1, n = 200, L1 min "0.786": cv_min = 0.785480, so **0.785**.
3. QDA envelope "[0.995, 1]": the archived upper limit is **1.0081**. The log-linear fit exceeds 1, and the draft caps it silently; it should say "1 (log-linear exceeds 1)".
4. Figure 1 legend "AUC(500) median 0.904" while the text and table use the point fit 0.906. Both are archived (boot_median 0.9035, point 0.9057), so the caption should say which one is plotted.
5. "~1 % of test rows near the decision boundary" has no run_id. It traces to `n_boundary_0.1_0.9 = 203` in r321's summary (and 204 in the R100–R102 summaries), i.e. 1.03%.

**E3** (e3_grid.csv, e3_null.csv, e3_leaderboard.csv, r321_e3_proxy.csv):
1. k_feat = 5 leaky "0.817": 0.816495, so **0.816**.
2. k_feat = 10 gap "0.026": 0.025461, so **0.025**.
3. k_feat = 13 gap "0.045": 0.044496, so **0.044**.
4. Table 1 caption "leaky 0.62-0.71": the 5-fold null leaky CV is 0.612 at k_feat = 100, so **0.61–0.71** for k_feat < 150.
5. "~48 features": this is the null expectation 150·P(|Z| > 1) = 47.6. On the real labels **61** features exceed 1/√250 (R943).
6. "the AUC gap peaks … at k_feat = 30 under the null": the 5-fold null gap is 0.2055 (k_feat = 20) vs 0.2059 (k_feat = 30), and the 3-fold null peaks at 20 (0.2010 vs 0.1994). Only 10-fold peaks clearly at 30. It should say "plateau at 20–30".
7. "public LB score is pending": it is now **0.79136** (S004).

**E4** (e4_*.csv, r320_*.csv, r321_e4_proxy.csv). R943 reproduces e4_drop.csv exactly.
1. *New*: Table 2, all 150, L2 "0.769 +/- 0.019": auc_std_repeat = 0.018496, so **±0.018**.
2. Table 2, stable, L2 "0.832 +/- 0.006": 0.005481, so **±0.005**.
3. Stability selection minus honest top-m at k = 20, "−0.007": −0.006479, so **−0.006**.
4. "≥ 98.6% of resamples" and Conclusion "from ≥ 0.986": the minimum is 0.9855 (features 80 and 77 at k = 10), so **≥ 98.5% / ≥ 0.985**.
5. "Six of the 13 label-free block features are **never** stable": refuted by `e4_stable_sets.csv`. Feature 67 is in the perm set at k = 13 and k = 20 and in the half set at k = 20; 8 and 26 are in the perm set at k = 20. The claim holds only for the top-10 / freq ≥ 0.5 definition.
6. "does not move with k once k ≥ 8": only k ∈ {5, 10, 13, 20} were run, so **k ∈ {10, 13, 20}**.
7. "far above any linear model" (six non-stable block features, QDA 0.917): R300 has block L2 C = 1 at 0.898. The margin over the best linear model is +0.019, not "far".

### 4. Scientific soundness (adversarial)

#### E1: is the true gap 0.01 or 0.003, and is the MDD right?

- **Gap.** Neither.
  - Conditional on these 250 rows, the infinite-repeat CV gap is **−0.0044 ± 0.0004**, pooled over 50 × 20 = 1000 repeats (R940 `summary.json: e1_pair.true_gap_1000rep`). The reference −0.0029 ± 0.0019 is within 1 SE of it.
  - The −0.0101 under the official seed is a 1.84-sd draw, made more likely because it was the closest-to-0.01 of 8 linear gaps (winner's curse; the draft says so).
- **Does it undermine E1?** No.
  - The MDD depends on sd(A − B), not on the gap. Across the 28 pairs (R940), the 5×20 sd(A − B) is 0.0018–0.0046, median 0.0033. 25/28 pairs keep the whole MDD CI below 0.01; the three failures all involve L1 C = 0.05.
  - Pairs whose real gap is 0.012–0.019 have a 5×20 MDD of 0.0063–0.0077 (upper CI 0.0078–0.0096). So the conclusion transfers to pairs that really differ by about 0.01.
  - However, the Conclusion's "two similar models about 0.01 apart" is not what was measured. The pair is 0.004 apart, and p_correct is a normal-shift extrapolation Φ(0.01/sd). The draft should say so.
- **Overstatements.**
  - (a) MDD = 2 sd ≈ (1.96 + 0) sd, i.e. **50% power** at α = 0.05. With 80% power (2.8 sd), 5×20 gives 0.0086 [0.0071, 0.0107]: inconclusive under E1's whole-CI rule, and only 4/28 pairs pass.
  - (b) For the decision the brief asks about (choosing = ordering), p_correct(0.01) ≥ 0.93 already for 5×10 on all 28 pairs. "Trust only 5×20" therefore depends on the criterion, which must be named.
  - (c) Data-sampling noise is 3.4 times the 5×20 split noise. Across 250-row replica samples, the population gap of this pair has sd 0.0105 and mean +0.0087, opposite in sign to the CV gap (R941; model-based, indicative). The draft's "lower bound" caution should carry this number.
- **MDD computation**: correct.
  - MDD = 2 × sd (ddof = 1) of 50 estimates. The CI is sd · √(49/χ²) with χ²₀.₉₇₅,₄₉ = 70.2 and χ²₀.₀₂₅,₄₉ = 31.6; I recomputed it (0.005105 / 0.007616 for 5×20).
  - Execution seeds 1 + 1000 s + r are disjoint (stride ≥ 20), and the reference seeds 900000+ are disjoint from them.
  - Paired vs unpaired sd: √(sA² + sB² − 2ρ sA sB) = 0.00306 with ρ = 0.627, against 0.0050 unpaired. Both match.

#### E2: is the extrapolation justified, and are its intervals honest?

- **Method.** An inverse power law on the effective training size is standard. The sensitivity fits, the envelope and the back-test are good practice. It still extrapolates 2.5× beyond the data with 3 parameters on 5 points.
- **Intervals: not honest.**
  - R942 replays the exact E2 protocol on 100 replica samples (500 bootstrap refits instead of 2000, no out-of-subset view). The nominal 90% power-law interval for AUC at 500 rows covers the replica truth (0.9230 ± 0.0006, R941) in **21%** of samples, and lies entirely below it in 72%.
  - Point bias is −0.031 (sd 0.031, RMSE 0.044). Log-linear is biased +0.017 (coverage 37%); unweighted 13%; drop-50 20%; envelope 81%.
  - QDA: power law covers 57%, envelope 81%. The QDA bias is only −0.0005, so the QDA conclusion ("< 0.002 gain") stands.
- **Why the interval is too narrow.**
  - It resamples subset scores of one 250-row sample, so parent-sample noise is missing.
  - The SE weights with `sigma_floor = 0.002` give the n = 250 point (CV-seed noise only) about 250× the weight of n = 50, which pins the curve to the realised sample.
  - The single back-test (R308) is one draw; on replicas the power law under-predicts on average. So "the back-test favours the power law" is not a valid argument for the 0.88–0.92 range. The replica truth sits between the power law and the log-linear fit.
- **Calibration caveat (stated more fully than in round 1).**
  - The replica's L1 truth vs E2's observed means at t = 40/80/120/160/200 is 0.636/0.768/0.802/0.828/0.851 vs 0.659/0.735/0.810/0.830/0.847. It is good for t ≥ 120, but +0.033 at t = 80 (2.6 SE of the observed mean).
  - The real-population value at 500 rows is therefore uncertain beyond what R941 gives. The under-coverage is a property of the procedure and does not depend on that value.
- **Gain bounds.** "+0.040 … positive in every bootstrap replicate" is guaranteed by the bounds b ∈ [0, 10⁴], c ∈ [0.01, 3] in `config.yaml: fit.bounds`, so it is uninformative.
- **Suggested wording**: "about 0.91–0.92 at 500 rows. A coverage simulation on replicas shows the bootstrap interval under-covers (21% for a nominal 90%), so the honest uncertainty is about ±0.04 (envelope 0.86–0.97)."

#### E3: are the pipelines right, and does the gap-vs-k explanation hold?

- **Leaky pipeline = the brief's procedure (1).** Confirmed in code. `leaky_columns()` fits `CorrelationTopK` on all 250 rows, and the CV pipeline is `Pipeline(scaler, FixedColumns(cols), logreg)` (`extensions/e3_leakage/run.py:131-142`).
- **Honest pipeline = procedure (2).** `Pipeline(scaler, CorrelationTopK(k), logreg)` is refitted per fold. Both are scored on identical splits (a signature check raises otherwise), and the 5-fold row has the official signature 5708ce5c3828. The k_feat = 150 gap is exactly 0, and the permuted-label honest CV is 0.49–0.51.
- **Theory.**
  - Correct. Given the all-row correlation s of a noise feature, E[r_train · r_val | s] = s² − 1/n.
  - E[r_val | s] = s independent of n_val, so the k_fold dependence cancels to first order. The null shows a nearly flat gap: 0.201 / 0.206 / 0.210 for 3 / 5 / 10 folds.
- **Small-k attribution (R944, corrects round 1).** The model, splits and grid are E3's, run on 100 replica 250-row samples with 20,000 true-label test rows each. Means over draws, 5-fold:

  | k_feat | gap | leak optimism (leaky CV − its fold models' test AUC) | honest bias (honest CV − its fold models' test AUC) | leaky − final model | honest − final model | real gap (percentile in replicas) |
  |---:|---:|---:|---:|---:|---:|---:|
  | 2 | 0.015 (sd 0.017) | 0.014 | 0.000 | +0.013 | −0.002 | 0.056 (96th) |
  | 5 | 0.010 | 0.009 | −0.001 | +0.007 | −0.003 | 0.023 (88th) |
  | 7 | 0.008 | 0.002 | −0.003 | −0.002 | −0.010 | 0.001 (23rd) |
  | 10 | 0.024 | 0.014 | −0.004 | +0.010 | −0.014 | 0.025 (56th) |
  | 13 | 0.041 | 0.030 | −0.004 | +0.026 | −0.016 | 0.044 (56th) |
  | 20 | 0.064 | 0.054 | −0.003 | +0.048 | −0.016 | 0.073 (70th) |
  | 30 | 0.078 | 0.070 | −0.000 | +0.063 | −0.015 | 0.090 (74th) |
  | 50 | 0.087 | 0.079 | +0.003 | +0.070 | −0.017 | 0.090 (62nd) |
  | 100 | 0.056 | 0.052 | +0.001 | +0.038 | −0.018 | 0.044 (22nd) |
  | 150 | 0 | −0.001 | −0.001 | −0.019 | −0.019 | 0 |

  (Source: `experiments/20261022-161150_r944_e3_replica_decomp/per_draw.csv`, `reports/tables/r944_e3_replica_decomp.csv`. The percentile is 1 − frac_ge_real.)
  - **On average the draft's mechanism holds.** Honest CV is unbiased for its own procedure. The whole gap, including the small one at k_feat ≤ 5, is leak optimism. Round 1's statement that "the small-k gap is mostly honest pessimism" is **refuted as a general claim**: on average honest − final is only −0.002 / −0.003 at k_feat = 2 / 5.
  - **But the real sample is atypical at small k.** The real gap 0.056 at k_feat = 2 is exceeded by only 4% of replicas. In the 8 replicas with a k_feat = 2 gap ≥ 0.04, the gap comes from honest-CV under-estimation: honest bias −0.037, leak optimism +0.002, leaky − final +0.002, honest − final −0.055. This is the pattern seen on the real data against the GMM proxy (leaky − proxy +0.011, honest − proxy −0.045). So the draft's "winner's curse among real features" accounts for about 0.014 on average, but not for this sample's 0.056. The rest is validation noise of the honest estimate, whose sd across samples is 0.03.
  - **k_feat = 7.** The dip at small k is generic: the minimum over k_feat ≤ 13 falls at 2 / 5 / 7 in 30 / 30 / 37 of 100 draws. The exact coincidence (gap 0.001, fold overlap 1.00) is partly luck: the replica median gap at 7 is 0.006, and 44% of draws are below 0.005. The draft's per-sample explanation (the in-fold filter re-selects the same 7) is correct for this sample and should be flagged as sample-specific.
  - **k_fold** (k_feat = 20). Leak optimism is flat at 0.054 / 0.054 / 0.053 for 2 / 5 / 10 folds. Honest − final shrinks from −0.046 to −0.016 to −0.010, and fold overlap rises from 0.63 to 0.78 to 0.84. This confirms the draft's first-order cancellation and its second-order explanation.
  - **Peak.** The gap peaks at k_feat = 50 in 65% of draws and at 30 in 25%, so "peaks at 30–50" is right.
  - **Leaky vs the LB target at k_feat = 20.** Replica +0.048 (honest −0.016), compared with the real +0.047 vs the proxy and +0.075 vs the public LB.
- **Leaderboard (new).** S004 public = 0.79136 against leaky 0.867, honest 0.793 and proxy 0.820. On ~395 rows the SE is ≈ 0.021, so:
  - leaky − public = +0.075 ≈ 3.6 SE;
  - honest − public = +0.002;
  - proxy − public = +0.029 ≈ 1.4 SE, which is within its own 90% public-subset range 0.785–0.854 but near the bottom.
  - B005 records both predictions as confirmed.
  - The draft's "leaky CV overstates the test AUC by 0.047" (vs the proxy) should be reported next to "0.075 vs the public LB". The two differ by about 1.4 public SEs and are compatible.
- **Consequence paragraph.** It holds that leaky CV picks k_feat = 50 (proxy 0.825). The draft omits that honest CV picks k_feat = 7 (proxy 0.852) while the proxy optimum is k_feat = 10 (0.864).

#### E4: bootstrap count, drop detection, and "stable is not useful"

- **Bootstrap count.** B = 2000 satisfies "several hundred". The frequencies and the drop table reproduce exactly (R943).
- **Sharp-drop detection.**
  - The observed largest steps are 0.494 / 0.370 / 0.369 / 0.298. Each is exceeded by 1 of 50 permuted-label curves (p = 2/51 ≈ 0.04 with the +1 correction; 0.02 without). The null largest step has mean 0.13–0.15, q95 0.25–0.36 and max 0.39–0.51, and sits at position 1–2.
  - So "sharp" is supported at about the 5% level, with the thinnest margin at k = 20 (0.298 vs q95 0.250).
- **Rule choice.**
  - The pre-registered primary rule is `perm` (`config.yaml: drop.primary_rule: perm`), giving m* = 7/8/9/17. It is the only calibrated rule: under leave-one-out null curves it admits ≥ 1 noise feature in 6–8%.
  - The `half` rule admits ≥ 1 noise feature in 28 / 72 / 90 / 100% of null curves (mean 5.2 at k = 20). `binom` admits 27–40 noise features on average, because a lucky noise feature is reproduced by every bootstrap resample; the draft explains this correctly.
  - The headline "Sharp drop, independent of k. Yes … its position does not move with k" rests on the two rules the draft chose after the pre-registered one moved. It should be stated as "the strong group of 7 (+49) is above the drop for every k ≥ 10; whether the weak features (|corr| 0.13–0.18) clear the calibrated noise envelope depends on k (9 more at k = 20)".
- **"Stable is not useful"**: supported.
  - Block > stable for L2 / QDA / GMM: −0.030 / −0.035 / −0.017, p = 0.018 / 0.0006 / 0.009 (r320_paired.csv).
  - The six non-stable block features alone give QDA 0.917 and GMM 0.932 (±0.031, min fold 0.54).
  - Their label-free component SMD is ≈ 0 (15: −0.025, 30: 0.010, 64: 0.003), and the log variance ratios are −1.00 / +0.63 / +0.83 for 8 / 64 / 67.
  - Feature 17 is an exact linear combination of the block (1 − R² = 2.8e-14; 6, 52, 57, 61, 142 and 144 are at 3e-14 to 7e-14) and adds QDA +0.0004 (p = 0.44).
- **Nuances to state.**
  - For a linear model, stable features *are* useful: +0.063 vs all 150 (p = 0.038, borderline).
  - For L2, the six non-stable block features alone score 0.765, below all 150 features (0.769). "Worth keeping" depends on the model family, which the draft does say.
  - The honest model on the 17 features above the pre-registered drop at k = 20 beats all 150 by only +0.028 (p = 0.30; e4_paired.csv row 34). The draft does not report this, and it is the brief's literal test at the primary rule.

### 5. Leakage / rules

- No problems. No external data are used (rule 4).
- Label-free steps (block, GMM, R²) use train+test features only (D007).
- GMM proxy labels on test are used only for analysis, never to train a submitted model. The E3 submission S004 is a plain supervised model.

### 6. Files

- Code: `research/r940_e1_pairs_check.py`, `r941_e2_replica_truth.py`, `r942_e2_coverage.py`, `r943_e4_null_step.py`, `r944_e3_replica_decomp.py`; this report.
- Runs, with Conclusion sections in notes.md: R940–R943 as listed at the top, and `experiments/20261022-161150_r944_e3_replica_decomp` (job J20261022-161330-pro1-vext, 14 min including VM setup).
- Colab session vext stopped (`remote.py down -a pro1 -s vext`) at the end.
- Tables: `reports/tables/r940_e1_pairs.csv`, `r941_e2_replica_truth.csv`, `r942_e2_coverage.csv`, `r943_e4_drop_vs_null.csv`, `r944_e3_replica_decomp.csv`.
