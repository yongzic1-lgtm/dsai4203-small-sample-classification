# sup：只用标签的监督学习 + sklearn 半监督基线（R200–R249）

- 作者：AI 代理（Claude Code），label `sup`；计算：Colab 账号 pro2、会话 `rsup`（High-RAM CPU，8 核），2026-10-22 JST。
- 代码：`research/lib_sup.py`、`research/r200_*.py` … `r210_*.py`、`research/r249_sup_comparison.py`。
- 汇总表/图：`reports/tables/r249_comparison.{csv,md}`、`reports/figures/r249_comparison.png`、`reports/figures/r249_label_budget.png`（R206 数据重画；`r206_label_budget.png` 是第一版）、`reports/tables/r206_label_budget.{csv,md}`、各脚本的 `reports/tables/r20x_candidates.csv`。
- 每个 run 的 `notes.md` 都写了 Conclusion（带 `(sup)` 标记）。

---

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要和 "Text for the report"）。依据：`research/reports/verify_cand.md` §1（r930、r932、r936）、`research/reports/verify_leak.md` §4.2–4.3、`research/reports/verify.md` §2 C3–C6。

1. **"Indistinguishable"**（摘要"意味着什么"第 1 点；Text for the report 第 1 段）。Old: "只看 CV AUC，GMM 和最好的监督模型**分不出来**"、"That is statistically indistinguishable from the transductive GMM (paired difference 0.0000, Nadeau-Bengio corrected p = 0.84)" → New (D009 wording): the CV is saturated (0.9999–1.0000) and cannot separate the two (paired 95% interval about ±2–5e-4; the supervised model is better in 0 of 100 folds and worse in 1–4). This is the resolution of the CV, not equal performance: on 222 replicas the same test was non-significant 222/222 times, while the supervised QDA's true AUC was lower in every replica (by 1.7e-4 with log-loss tuning, 2.6e-4 with AUC tuning). Sources: `reports/tables/r930_paired_vs_gmm.md` (95% intervals [−1.8e-4, +1.5e-4] for r210's reg 0.001 and [−6.1e-4, +4.1e-4] for r201's fold scores; folds worse 1 / 4, better 0), `reports/tables/r936_cv_power.csv` (222 replicas, `share_cv_not_significant` = 1, `share_true_gap_negative` = 1), `reports/tables/r936_private.csv` (tn0, csv8, R156 weights: r210 −1.68e-4, r201 −2.57e-4).
2. **Log-loss tuning**（摘要"方法上的教训"第 1 条；Text for the report 第 2 段末）。Old: "把 inner 指标换成 log-loss 就解决了（R210：reg 每折都选 0.001…）"、"A proper scoring rule such as log-loss avoids this." → New: inner log-loss chose the grid floor 0.001 in every fold; an extended grid picks 1e-4 / 3e-4 (r930), CV AUC unchanged; the true gap to the GMM on replicas shrinks by about 1e-4; log-loss tuning is fragile to mislabelled training rows (−6.3e-4). Sources: `20261022-095943_r930_qda_block_nll_ext` (reg 1e-4 in 53 folds, 3e-4 in 36, 1e-3 in 11; CV 0.999984), `20261022-100052_r930_reg_edge_summary` (OOF log-loss optimum 0.0071 at 3e-4), `reports/tables/r936_private.csv` (tn0 csv8 weighted: r210 −1.68e-4 vs extended grid −6.3e-5; tn1 raw weighted: r210 −6.27e-4).
3. **Selection correction**（摘要"意味着什么"第 1 点；Text for the report 第 1 段）。Old: "0.9999 ± 0.0002, or 1.0000 ± 0.0001" as the supervised level, with no selection correction next to it → New: adds "0.9995 after model selection (r205)". Source: `20261022-072747_r205_select_supervised` summary.json (auc_mean 0.99955, sd over repeats 0.0006, 46-configuration menu nested; the track tried 587 configurations).
4. **r209 `*_r003` runs**（摘要"无标签数据的价值"自训练一条；Text for the report 第 2 段）。Old: the "well-tuned base (reg 0.003)" comparison was cited without a caveat → New: marked as not nested-honest, because reg 0.003 is R201's modal choice over all outer folds (`research/r209_semisup_followup.py:46,49`; already disclosed in 合规说明 below). Never cite `r209_sup_qda_tt_r003` / `r209_selftrain_qda_r003` as nested-honest estimates.
5. **(fix round 2, 2026-10-22) "No method exceeded 0.88 on all 150 features"**（摘要"意味着什么"第 3 点；Text for the report 第 1 段末）。Old: "在全部 150 个特征上，每种方法都不超过 0.88"、"and no method exceeded 0.88 on all 150 features" → New: "在全部 150 个特征上，本线的模型都不超过 0.88（fm 的 TabICL v2 不做特征选择也有 0.993）"、"and no model in this track exceeded 0.88 on all 150 features (the pretrained TabICL v2 reaches 0.993 on all features without selection; fm)". Source: R440 claim P31 (`reports/tables/r440_core_numbers.csv`, run `20261022-184020_r440_core_numbers`; `research/reports/r2_audit.md` §5 item 1). Sup-track maximum on all features 0.8753 (`20261022-075225_r208_qda_all_lw`); `20261022-074251_r251_tabicl_all` auc_mean 0.99289.
6. **(fix round 2, 2026-10-22) Label budget of strictly supervised QDA**（摘要"意味着什么"第 4 点；Text for the report 第 2 段）。Old: "严格监督约 **80** 个"、"or about 80 when the feature block must also be found from the labelled rows" → New: "严格监督约 **80–120** 个（reg 固定 0.003 时 80；S005 的嵌套流程 120；R441 重现）"、"or about 80–120 when the feature block must also be found from the labelled rows (80 with reg fixed at 0.003, a value taken from R201; 120 for the nested S005 procedure; R206, re-run in R441)". Source: R441 `20261022-182022_r441_label_budget_rerun` and R440 claims L12, L14 (`reports/tables/r440_label_budget_first_hit.csv`: first budget with mean AUC ≥ 0.99 is 80 for `eq206_qda_infold_r003`, 120 for `r201_qda_block`; `reports/tables/r441_label_budget.csv`: r201 at 80 labels mean 0.9842, median 0.9984; at 40 labels 371 of 400 fits undefined); `research/reports/r2_audit.md` §2.3, §5 item 2.
7. **(fix round 2, 2026-10-22) GPC-ARD**（摘要"意味着什么"第 2 点）。Old: "GPC-ARD 0.978" → New: "GPC-ARD 0.976". Source: R440 claim P7 (`research/reports/r2_audit.md` §5 item 3). 0.976 is this track's `20261022-072331_r202_gpc_ard_block` (auc_mean 0.97574), as in the table above and in the English text; 0.978 is fm's `20261022-070207_r255_gpc_ard_hv` (0.97776, train+test block).

## 中文摘要（给队友）

### 做了什么

- **63 个正式候选 + 1 个失败 run**。全部用官方 CV（5 折 × 20 次重复，seed 2026，split 签名 `5708ce5c3828`），每个都存了 `test_pred.npy`。所有调参都用**嵌套 CV**：在 200 行训练折内部做 `RepeatedStratifiedKFold(5, 2)`，用 GridSearchCV 选参，所以外层 CV 是诚实的。整条线一共试了 **587 个配置**（各 run 的 `n_configs` 加总，见 `r249_comparison.md` 的表头）。另有 2 个分析 run：R206 标签预算曲线，R249 对比表和图。
- 数据使用分三档，每个 run 都在 `uses_unlabeled_test_features` 里标明：
  1. **严格监督**：只用 250 行。高方差块在每个训练折里用那 200 行重新求（R001 的规则：std > 1.5 × 中位数 std），标准化也在折内做，完全不碰 test 特征。
  2. **label-free 选块**：13 列的块和它的标准化用 train+test 特征确定（R003 的 `SubspaceModel`），但模型只用标签。
  3. **sklearn 半监督**：SelfTraining、LabelSpreading、LabelPropagation 在全部 20,000 行上运行。只有当前训练折的行带标签，其他行（包括验证折）一律作为 −1 传入。

  对照基准是 R003 的 TransductiveGMM（在 R249 里用同一组 split 重算）。

### 关键数字（run_id 前缀都是 20261022-）

| 模型 | 用无标签数据？ | 配置数 | CV AUC ± sd(repeat) | OOF log-loss | 与最佳严格监督的差 (split SE) / p_corr |
|---|---|---|---|---|---|
| L1 logistic，150 个特征（`065631_r200_l1_all`） | 否 | 10 | 0.8693 ± 0.0131 | 0.478 | −0.131 (0.003) / <0.001 |
| L2 logistic，13 列块，C 调到 1e4（`074147_r207_l2_block_tt_wide`） | 只用于选块 | 11 | 0.9383 ± 0.0068 | 0.336 | −0.062 (0.002) / <0.001 |
| LightGBM，块，嵌套（`074613_r203_lgbm_block_nested`） | 否 | 8 | 0.9136 ± 0.0113 | 0.376 | −0.086 (0.003) / <0.001 |
| GPC-ARD，块（`072331_r202_gpc_ard_block`） | 否 | 1 (ML-II) | 0.9757 ± 0.0047 | 0.258 | −0.024 (0.001) / 0.006 |
| Gaussian NB，块（`071541_r201_gnb_block`） | 否 | 1 | 0.8668 ± 0.0050 | 0.626 | −0.133 (0.001) / <0.001 |
| QDA reg 0.1，13 列块（= R003 / S002，`080551_r209_sup_qda_tt_r01`） | 只用于选块 | 1 | 0.9958 ± 0.0011 | 0.088 | −0.0042 (0.0003) / 0.12 |
| **QDA，折内块，嵌套 reg，inner 指标 AUC**（`071234_r201_qda_block`） | **否** | 11 | **0.9999 ± 0.0002** | 0.0146 | −0.0001 (0.0001) / 0.77 |
| **QDA，折内块，嵌套 reg，inner 指标 log-loss**（`082047_r210_qda_block_nll`） | **否** | 11 | **1.0000 ± 0.0001** | 0.0087 | 参照 |
| 选择感知估计：46 个配置的菜单整体嵌套（`072747_r205_select_supervised`） | 否 | 46 | 0.9995 ± 0.0006 | 0.028 | −0.0004 (0.0001) / 0.56 |
| LabelSpreading，kNN 图，宽网格（`080634_r209_labelspread_widegrid`） | 是 | 9 | 0.9566 ± 0.0070 | 0.446 | −0.043 (0.002) / 0.002 |
| 监督 QDA-LW，13 列块（`082044_r210_sup_qda_tt_lw`） | 只用于选块 | 1 | 0.9962 ± 0.0010 | 0.082 | −0.0038 (0.0002) / 0.14 |
| 自训练 QDA-LW，20,000 行（`080632_r209_selftrain_qda_lw`） | 是 | 1 | 1.0000 ± 0.0000 | **0.0071** | 0.0000 / 0.84 |
| TransductiveGMM K=2（R003 `050708_r003_gmm2_hv`，重算） | 是 | 1 | 1.0000 ± 0.0000 | 0.0171 | 0.0000 / 0.84 |

### 意味着什么

1. **250 个标签就足以让 CV AUC 饱和。** 只要 QDA 的 reg_param 调小（嵌套 CV 每折都选 0.001–0.01），严格监督（不碰任何 test 特征）就能做到 0.9999（r201）和 1.0000（r210）；把"从多个配置里挑最好"的选择效应算进去是 0.9995（r205）。R003 的 0.9958 来自 reg 0.1 过度正则化。CV 因此饱和，**分不出** GMM 和最好的监督模型：diff 0.0000，p_corr 0.84；对 r201 是 0.0001，p 0.71；配对差的 95% 区间约 ±2–5e-4，100 折里监督模型 0 胜、1–4 负（r930）。这是 CV 分辨率不够，不是两者一样好：在 222 个复刻上，同样的检验 222/222 次不显著，而监督 QDA 的真实 AUC 每次都更低（log-loss 调参平均低 1.7e-4，AUC 调参低 2.6e-4；r936）。所以 R003/D007 里"直推式比监督式好 0.004"的说法只对未调参的 QDA 成立，报告里要改。
2. **信号在相关结构里**，即协方差的非对角元。LDA（只看均值）0.86–0.88，GNB（均值 + 各特征方差）0.87，QDA（全协方差）0.9999。其他模型族都学不到这种结构：线性模型的上限约 0.94（R207），二阶多项式 logistic 0.93，SVM 0.95，kNN 0.90，GPC-ARD 0.976，树模型最多 0.91。
3. **特征选择。** label-free 的方差规则远好于有监督的相关系数筛选：QDA 用 top-k |corr| 是 0.9926（用完整 reg 网格后），用方差块是 0.9999；线性模型 0.86 对 0.93。在全部 150 个特征上，本线的模型都不超过 0.88（fm 的 TabICL v2 不做特征选择也有 0.993）。200 行的折内方差规则会多带 3–10 个边缘列（平均 19.1 列），QDA 用 13 列还是 19 列没有差别（R208、R201 qda_block_tt）。
4. **无标签数据的价值**不体现在 250 个标签时的 AUC 上，体现在下面几处：
   - **标签少的时候**（R206，`r249_label_budget.png`）：要达到 AUC ≥ 0.99，GMM 只要 **4** 个标签，自训练 QDA-LW 约 **20** 个，监督 QDA（13 列块，reg 0.003）约 **40** 个，严格监督约 **80–120** 个（reg 固定 0.003 时 80；S005 的嵌套流程 120；R441 重现）。在 20 个标签时，GMM 是 1.000，自训练 0.995，最好的监督模型只有 0.851。
   - **校准**：自训练 QDA-LW 的 log-loss 是 0.0071，监督模型最好 0.0087，GMM 0.0171。
   - **测试集边界行**（GMM 后验在 0.1–0.9 的 204 行）：和 GMM 排序的一致性，自训练是 Spearman 0.70，监督 QDA 只有 0.45–0.49。
   - **自训练只对没调好的基模型有用**：QDA-LW 从 0.9962 升到 1.0000（log-loss 0.082 → 0.0071）。基模型调好之后（reg 0.003）就没有提升（1.0000 → 1.0000，log-loss 0.0109 → 0.0120）。注意：reg 0.003 取自 R201 在全部外层折上的嵌套选择，所以这两个 `r209_*_r003` 的 1.0000 不是嵌套诚实的估计，不能这样引用（见下面的合规说明）。
   - **图方法（LabelSpreading / LabelPropagation）不行**：最好 0.957，在 150 维上只有 0.62。半监督学习要先有对的表示。
5. **方法上的教训**（都可以写进报告）：
   - AUC 一饱和，嵌套 CV 里按 AUC 选参就会大量打平。GridSearchCV 打平时取第一个，也就是正则最强的那个，结果系统性过度正则化。把 inner 指标换成 log-loss 能打破平局（R210：AUC 0.9999 → 1.0000，log-loss 0.0146 → 0.0087），但没有完全解决：R210 每折都选 0.001，而 0.001 是网格下边界。网格往下扩展后，inner log-loss 选 1e-4 / 3e-4（r930，100 折中 53 / 36 折），CV AUC 不变；复刻上与 GMM 的真实差距缩小约 1e-4（−1.68e-4 → −6.3e-5，r936）。另外，log-loss 调参对错标的训练行更脆弱（训练标签也有噪声时 r210 平均 −6.3e-4，r936 tn1）。
   - Winner's curse：把整个"在 46 个配置里挑最好"的过程当作一个估计器做嵌套（R205），得到 0.9995，最好单个模型是 0.9999 / 1.0000，所以乐观偏差不超过约 0.0005。偏差这么小，是因为 QDA 族在 100/100 折里都被选中。
   - 好几个网格碰到了边界（每个 run 的 Conclusion 里都写了），已经用 R207、R208、R209、R210 复查过。R201 的 qda_blockextra / qda_corrk 和 R204 的自训练分数偏低，是"网格在知道小 reg 最好之前就定死了"造成的。

### "诚实监督"提交候选

- **首选 `20261022-071234_r201_qda_block`。** 设计是预先登记的；严格监督，不用任何 test 特征；只调 1 个超参数（reg）；CV 0.9999 ± 0.0002，log-loss 0.0146。它的 test 预测和 GMM 硬标签的 pseudo-AUC 是 0.9997。作为参照，S002（QDA reg 0.1）的同一指标是 0.9940，public 是 0.99277，可见这个指标跟得住 public。所以预计 public 约 0.998–0.999（public 榜在 AUC ≈ 0.999 时 SE ≈ 0.0015）。
- **备选 `20261022-082047_r210_qda_block_nll`**：1.0000，log-loss 0.0087，pseudo-AUC 0.9998。但"用 log-loss 作 inner 指标"是看了 R201/R209 之后才定的，属于第二层选择，报告里要写明。
- 在报告叙事里，它的作用是回答"只用 250 个标签能做到多少"。它不能替代 GMM：两者 CV 一样，但在边界行上排序不同，GMM 在标签少时的优势和 public 0.99923 都是它的论据。

### 负面结果（都已归档）

- 线性模型在任何特征集、任何惩罚下都不超过 0.94。全部 150 特征上的 LDA 0.796，L2 0.787。
- 用相关系数筛选特征不如方差规则（线性、LDA、QDA 都比较过）。
- 树模型和 boosting 最多 0.91（嵌套 LightGBM），调参只多 0.006。
- RBF-SVM、kNN、GPC（包括 ARD）都明显不如 QDA。二阶多项式 logistic 的函数类是对的，但只有 200 行，学不出来（0.93）。
- QDA-LW 不调参会过度收缩（0.986）。sklearn 1.9 的 svd-QDA 要求每类行数多于特征数，所以 `r201_qda_all` 失败（已归档为 failed）；QDA-LW 在 150 个特征上是 0.875。
- 以 LR 为基模型的自训练反而更差（0.878，比监督 LR 的 0.898 低）：基模型族不对，自训练也救不回来。
- LabelSpreading / LabelPropagation 在块上 0.91–0.96，在全部特征上 0.62。

### 预注册的假设被推翻了哪些（可用于反思 Q4）

- 预测"线性 ≤ 0.93"，实际 0.938。
- 预测"严格监督 QDA 0.994–0.997"，实际 0.9999。
- 预测"GNB 0.92–0.98"，实际 0.87。
- 预测"多项式 LR 0.97–0.995"，实际 0.93。
- 预测"GPC-ARD 0.985–0.996"，实际 0.976。
- 预测"自训练 QDA ≥ 监督 QDA"：调好参时并不成立。
- 预测"GMM 比最佳监督高 0.004 且 p < 0.05"：不成立，p = 0.84。

### 计算 / 流程上的问题（需要维护 `colab/` 的人处理）

- **`colab/remote.py` 对非默认账号（pro2）有 bug。** `_colab_prefix()` 生成的是 `C="$HOME/.local/bin/colab"; A="$HOME/.colab-accounts/pro2"; mkdir -p "$A"; HOME="$A" "$C"`，再经 `wsl -d Ubuntu-24.04 -- bash -lc <cmd>` 执行。`wsl ... --` 会先用 WSL 的默认 shell 解析一遍命令行，这时 `$A`、`$C` 还没定义，被展开成空串，于是报错 `mkdir: cannot create directory ''`，exit 127。验证方法：`wsl -d Ubuntu-24.04 -- bash -lc 'A=hello; echo "[$A]"'` 输出 `[]`，换成 `wsl -d Ubuntu-24.04 -e bash -lc ...` 输出 `[hello]`。
  - 我的临时做法：在调用 remote.py 的 PowerShell 进程里设置 `$env:C="/home/cyzcomputer/.local/bin/colab"`、`$env:A="/home/cyzcomputer/.colab-accounts/pro2"`、`$env:WSLENV="C:A"`。只影响那个进程，`colab/` 下的文件没改。
  - 建议修复：用 `wsl -e`（改之前要测 pro1 的引号行为），或者在 prefix 里把 `$` 转义。
- **VM 卡死。** 第一个 high-mem VM 的 websocket 卡死了，我手动 `down` 后重建。之后几个作业的日志流也卡过 10 分钟左右，remote.py 自己重连后结果都完整取回了。
- **两次中止，都没有产生合并的 run。** 一次是 2 核标准 VM 上的 `J20261022-064229-pro2-rsup`（为了换 high-mem）；另一次是卡死的那个 VM 在启动作业之前就被停掉了。
- 我在本机 smoke test 时不小心建了 `D:\d\tmp\...\sup_fake`（Git Bash 路径被 Windows Python 误解析），已删除；`D:\d` 里原来的内容没动。
- 我没有改 `registry/*.md`（按规则不允许）。建议主会话把下面的参考文献加进 `registry/references.md`，并在 `ideas.md` / `journal.md` 里记下这条线的结论。

### 合规说明

- 没有用外部数据。半监督模型只用 test 特征、不用 test 标签。验证折的标签从不参与拟合。
- 有几个 run 的设定是看了前面的结果才定的（各自 Conclusion 里都写了），**不能当作诚实估计**：
  - R206 的 `*_r003` 曲线、`r209_sup_qda_tt_r003`、`r209_selftrain_qda_r003`：reg 0.003 取自 R201 的嵌套选择。
  - `r208_qda_block13_infold`：ratio 1.65 是对照 train+test 特征的块挑出来的（label-free，但看过 test 特征）。
  - `r210_*`：inner 指标改用 log-loss 是第二层选择。

---

## Text for the report (English)

**Supervised-only baselines and the value of unlabeled data.** To separate what the 250 labels can do on their own from what the 19,750 unlabeled test rows add, we evaluated 63 further candidates (587 hyperparameter configurations in total) with the official repeated 5-fold CV. Every tuned hyperparameter was chosen by nested CV inside the 200-row training fold. "Strictly supervised" pipelines never touch test features: the high-variance block is re-derived from the training fold itself. Once its covariance regularisation is tuned (nested `reg_param` 0.001-0.01), a strictly supervised QDA on this block reaches CV AUC 0.9999 ± 0.0002, or 1.0000 ± 0.0001 when the inner CV scores log-loss (0.9995 after model selection, r205). The CV is therefore saturated and cannot separate it from the transductive GMM: the paired difference is 0.0000 (Nadeau-Bengio corrected p = 0.84), its 95% interval is about ±2–5 × 10⁻⁴, and the supervised model is never better in any of the 100 folds (worse in 1–4). This reflects the resolution of the CV, not equal performance. On 222 parametric replicas the same test was non-significant every time, yet the supervised QDA's true AUC was lower in every replica: by 1.7 × 10⁻⁴ on average with log-loss tuning and 2.6 × 10⁻⁴ with AUC tuning (r936). Our earlier supervised baseline (QDA, reg 0.1, 0.9958) was simply over-regularised. A selection-aware estimate, nesting the choice among a 46-configuration menu of model families, gives 0.9995, so the optimism from reporting the best of many configurations is at most about 0.0005 here. The ablations locate the signal in the class-specific correlation structure. Shrinkage LDA (means only) reaches 0.86-0.88 and Gaussian naive Bayes (means and per-feature variances) 0.87, against 0.9999 for full-covariance QDA. Every model family that cannot represent class-specific correlations efficiently from 200 rows stays clearly below QDA: penalised logistic regression (best 0.938), degree-2 polynomial logistic regression (0.930), RBF-SVM (0.951), kNN (0.902), Gaussian-process classification with ARD (0.976) and regularised tree ensembles (≤ 0.914). Label-free variance-based column selection beat in-fold correlation filtering for logistic regression, LDA and QDA (the families where both were compared), and no model in this track exceeded 0.88 on all 150 features (the pretrained TabICL v2 reaches 0.993 on all features without selection; fm).

With 250 labels the CV AUC is saturated, so the value of the unlabeled rows has to be measured differently. Re-running the official splits with subsampled label budgets shows it clearly (Fig. r249_label_budget). The transductive GMM reaches AUC 1.000 with only 4 labels and self-training QDA reaches 0.99 with about 20. Supervised QDA needs about 40 labels, or about 80–120 when the feature block must also be found from the labelled rows (80 with reg fixed at 0.003, a value taken from R201; 120 for the nested S005 procedure; R206, re-run in R441); at 20 labels the best supervised model scores 0.85. At the full budget the unlabeled rows still improve calibration: self-training QDA (Ledoit-Wolf) has OOF log-loss 0.0071, against 0.0087 for the best supervised model and 0.0171 for the GMM. They also improve agreement with the GMM on the 204 ambiguous test rows (Spearman 0.70 vs 0.45-0.49). Scikit-learn's generic semi-supervised baselines are no substitute for the right model family: label spreading and label propagation on a kNN graph reach at most 0.957 (0.62 on all 150 features), and self-training helps only a poorly tuned base (QDA-LW 0.9962 → 1.0000) but not a well-tuned one (a QDA with reg fixed at 0.003; that value came from R201's nested choices over all outer folds, so this comparison is not a nested-honest estimate). One methodological lesson: once inner-fold AUCs saturate, AUC-based nested tuning produces many ties, and GridSearchCV resolves them towards the most regularised setting. Tuning by log-loss removes the ties, but it is not a full fix. On our grid it chose the smallest value offered (reg 0.001) in every fold, and an extended grid picks 1e-4 or 3e-4 with unchanged CV AUC (r930). On replicas the smaller value shrinks the true gap to the GMM by about 1 × 10⁻⁴, and log-loss tuning makes the model fragile to mislabelled training rows (−6.3 × 10⁻⁴ on average when training labels are also noisy; r936).

| Model (official 5×20 CV) | Unlabeled rows used | Configs | CV AUC ± sd | OOF log-loss | Δ vs best supervised (split SE), p_corr |
|---|---|---|---|---|---|
| L1 logistic, all 150 features | no | 10 | 0.869 ± 0.013 | 0.478 | −0.131 (0.003), <0.001 |
| L2 logistic, 13-column block | block choice only | 11 | 0.938 ± 0.007 | 0.336 | −0.062 (0.002), <0.001 |
| LightGBM, block (nested) | no | 8 | 0.914 ± 0.011 | 0.376 | −0.086 (0.003), <0.001 |
| GP classifier (ARD), block | no | 1 | 0.976 ± 0.005 | 0.258 | −0.024 (0.001), 0.006 |
| Gaussian naive Bayes, block | no | 1 | 0.867 ± 0.005 | 0.626 | −0.133 (0.001), <0.001 |
| QDA reg 0.1 (earlier baseline, S002) | block choice only | 1 | 0.9958 ± 0.0011 | 0.088 | −0.0042 (0.0003), 0.12 |
| QDA, in-fold block, nested reg (AUC) | no | 11 | 0.9999 ± 0.0002 | 0.0146 | −0.0001 (0.0001), 0.77 |
| QDA, in-fold block, nested reg (log-loss) | no | 11 | 1.0000 ± 0.0001 | 0.0087 | reference |
| Nested selection over 46 supervised configs | no | 46 | 0.9995 ± 0.0006 | 0.028 | −0.0004 (0.0001), 0.56 |
| Label spreading (kNN graph), nested | yes (20,000 rows) | 9 | 0.957 ± 0.007 | 0.446 | −0.043 (0.002), 0.002 |
| Self-training QDA (Ledoit-Wolf) | yes (20,000 rows) | 1 | 1.0000 ± 0.0000 | 0.0071 | 0.0000 (0.0000), 0.84 |
| Transductive GMM, K = 2 (final family) | yes (20,000 rows) | 1 | 1.0000 ± 0.0000 | 0.0171 | 0.0000 (0.0000), 0.84 |

*Caption.* sd = standard deviation across the 20 repeat means. Δ = paired mean AUC difference on identical folds; split SE = sd of the 20 per-repeat differences / √20; p_corr = Nadeau-Bengio corrected resampled t-test. "Block" = the label-free high-variance feature block. Runs: `reports/tables/r249_comparison.md` (analysis run 20261022-082329_r249_sup_comparison).

---

## Appendix

### Colab jobs (pro2 / rsup, all exit 0, merged into the real archive)

| job | script | runs |
|---|---|---|
| J20261022-065811-pro2-rsup | r200_sup_linear.py | 11 |
| J20261022-071416-pro2-rsup | r201_sup_quadratic.py | 15 (1 failed by construction: r201_qda_all) |
| J20261022-072103-pro2-rsup | r204_semisup_sklearn.py | 7 |
| J20261022-072500-pro2-rsup | r202_sup_gpc.py | 4 |
| J20261022-072928-pro2-rsup | r205_sup_nested_selection.py | 1 |
| J20261022-074230-pro2-rsup | r207_sup_linear_widegrid.py | 3 |
| J20261022-074602-pro2-rsup | r203_sup_trees.py | 9 |
| J20261022-074852-pro2-rsup | r206_label_budget.py | 1 analysis |
| J20261022-075121-pro2-rsup | r208_sup_qda_followup.py | 5 |
| J20261022-080727-pro2-rsup | r209_semisup_followup.py | 6 |
| J20261022-082147-pro2-rsup | r249_sup_comparison.py | 1 analysis (superseded) |
| J20261022-082225-pro2-rsup | r210_sup_tiebreak_lw.py | 3 |
| J20261022-082504-pro2-rsup | r249_sup_comparison.py | 1 analysis (final) |

### Libraries and methods to cite (add to `registry/references.md`)

- scikit-learn 1.9.1 (Pedregosa et al., 2011): LogisticRegression, with liblinear (Fan et al., 2008) and saga (Defazio et al., 2014); Linear/QuadraticDiscriminantAnalysis (Hastie, Tibshirani & Friedman, 2009, ch. 4); Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004); GaussianNB; SVC (LIBSVM, Chang & Lin, 2011; Platt scaling, Platt, 1999); KNeighborsClassifier; GaussianProcessClassifier with the Laplace approximation (Rasmussen & Williams, 2006); RandomForest (Breiman, 2001); ExtraTrees (Geurts et al., 2006); SelfTrainingClassifier (self-training, Yarowsky, 1995); LabelSpreading (Zhou et al., 2004); LabelPropagation (Zhu & Ghahramani, 2002); GridSearchCV; PolynomialFeatures.
- LightGBM 4.7 (Ke et al., 2017); XGBoost 3.2 (Chen & Guestrin, 2016).
- Nested CV and selection bias (Varma & Simon, 2006; Cawley & Talbot, 2010); corrected resampled t-test (Nadeau & Bengio, 2003); Gaussian mixture / EM (Dempster, Laird & Rubin, 1977), for the TransductiveGMM reference.
- numpy, pandas, scipy, matplotlib, joblib (already listed).

### Open points

- `reg_param = 0.001` (the grid bottom) was chosen in 100/100 folds when tuning by inner log-loss. An even smaller reg might lower log-loss further; AUC is already saturated.
- The 3-10 borderline columns found by the in-fold rule (63, 84, 105, 89, 31, 122, …) do not help QDA or linear models. Whether they matter for the GMM's boundary rows is untested.
- Pseudo-AUC against GMM labels is a proxy that favours models similar to the GMM. A real check would need one public-LB submission of `r201_qda_block` (that is for the main session to decide).
