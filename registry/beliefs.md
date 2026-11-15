# Beliefs（预注册的预测 vs 结果）

规则（`docs/ARCHIVE_RULES.md` §7）：**结果出来之前**写预测，写明置信度和检验方式；结果出来后只填 Outcome 和 Lesson，不改原来的预测。每次提交前至少一条。每个人都要有自己的几条：反思 Q4 问的是"你一开始相信、后来被 leaderboard 证伪的一件事"。

列说明：
- **Belief / prediction**：要具体、可检验（数字或方向）。
- **Confidence**：主观概率（%）。
- **How it will be checked**：用哪个 run / 提交 / 比较来检验。
- **Outcome**：实际结果（数字），以及是否证实（confirmed / refuted / inconclusive）。
- **Lesson**：学到了什么，下次怎么做。

示例写法（不是记录）：`B0xx | 2026-10-24 | ab | S002 的 public AUC 会落在 CV 均值 ±0.04 之内 | 80% | record_lb 后对比 expected_public | 0.71 vs 0.79，refuted | public 只有约 395 行，单个分数不能拿来下结论`

| ID | Date | Author | Belief / prediction | Confidence | How it will be checked | Outcome | Lesson |
|---|---|---|---|---|---|---|---|
| B001 | 2026-10-22 | TODO (AI-assisted) | S001（L1 logistic，全部特征）的 public AUC ≈ 0.85（CV 0.847） | 70% 落在 ±0.04 内 | S001 上传后 record_lb | 0.83589，confirmed（−0.014 = −0.7 SE） | 线性基线的 CV 估计是准的；public 榜 SE 约 0.02 |
| B002 | 2026-10-22 | TODO (AI-assisted) | S002（监督 QDA，13 个高方差特征）的 public AUC ≈ 0.99 | 70% | S002 | 0.99277，confirmed（+0.6 SE） | 协方差结构才是信号所在；监督模型只用 250 行也几乎够用 |
| B003 | 2026-10-22 | TODO (AI-assisted) | S003（直推式 GMM）的 public AUC ≈ 0.999，不会正好是 1.0，因为约 1% 的测试行在边界上 | 60% | S003 | 0.99923，confirmed | CV 已经饱和在 1.0，离 1.0 的那一点差距只能从边界样本上找 |
| B004 | 2026-10-22 | TODO (AI-assisted) | S003 的 private AUC ≥ 0.999，且和 public 的差 ≤ 0.001 | 70% | 截止后 record_lb --private | pending | |
| B005 | 2026-10-22 | TODO (AI-assisted) | S004（E3 最终模型：相关系数 top-20 + L2 逻辑回归 C=0.1，诚实 CV 0.793，泄漏 CV 0.867）的 public AUC ≈ 0.82，落在 0.785–0.854（90%）；会比泄漏 CV 低，接近诚实 CV 和 19,750 行代理值 0.819 | 90% 落在区间内；80% 比泄漏 CV 0.867 低 0.02 以上 | S004 上传后 record_lb，填进 E3 的 LB 对比 | 0.79136：落在区间内（靠近下端，比预期低 0.029 = −1.2 SE）；比泄漏 CV 低 0.075，和诚实 CV 只差 0.002；两条都 confirmed | 诚实 CV 才是 LB 的好预测，泄漏 CV 高估了约 0.075；预期值 0.82 取自 19,750 行代理值，偏乐观了，下次直接用诚实 CV 做预期 |
| B006 | 2026-10-22 | TODO (AI-assisted) | S005 = r201_qda_block（`20261022-071234_r201_qda_block`，严格归纳 QDA，不用任何测试集特征，CV 0.99990）的 public AUC ≈ 0.9988，80% 落在 0.9980–0.9997（以 S003 的 public 0.99923 为条件，预测 sd 0.00066；不加条件的 Hanley–McNeil SE 0.0017）；高于 S003 的概率约 20%；低于 0.9960 说明文件有问题。这个分数不影响最终选择（D010） | 80% 落在区间内 | 上传后 record_lb；对照 `reports/tables/r936_public_abc.md` | 0.99953：落在区间内（靠上端，比点估计高 +0.0007 = +0.6 SE），confirmed；而且比 S003 的 0.99923 还高——事先估计这种情况的概率只有约 14–20%（R425） | public 只有 395 行，差 0.0003 只相当于约 12 对排序；比 S003 高不说明严格监督 QDA 更好（复刻上 private 期望低 2.4e-4）。按 D010，这个分数不改变最终选择 |
| B007 | 2026-10-22 | TODO (AI-assisted) | S006 = r209_selftrain_qda_lw（`20261022-080632_r209_selftrain_qda_lw`，自训练 QDA-LW，用测试集特征，CV 1.0）的 public AUC ≈ 0.9991，80% 落在 0.9987–0.9996（以 S003 的 public 为条件，预测 sd 0.00033；Hanley–McNeil SE 0.0015）；高于 S003 的概率约 25–28%，与它完全相同的概率约 4%；低于 0.9975 说明文件有问题。这个分数不影响最终选择（D010） | 80% 落在区间内 | 上传后 record_lb；对照 `r936_public_abc.md` | 0.99956：落在区间内（+0.0005 = +0.4 SE），confirmed；比 S003 高（事先概率约 27%） | 同上：public 的差距在噪声以内（public 差的 sd 3–6e-4）；最终选择仍按 D010 |
| B008 | 2026-10-22 | TODO (AI-assisted) | 校准后重写 B004（B004 原文不改）：S003 的 private AUC 中位数约 0.997，90% 落在 0.992–0.9996；P(private ≥ 0.999) 约 0.15；private 比 public 低约 0.002 | 90% 落在区间内 | 截止后 record_lb --private | pending | |
