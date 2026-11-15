# 作业要求清单（DSAI4203 Small-Sample Classification Challenge）

按交付物分组。打勾的项后面写证据位置；没打勾的项写谁来做：**T** = 只有队员能做；**A→T** = AI 可以起草，队员核对、改写，并且能解释（规则 6）。

状态更新于 **2026-11-01**。比赛规则的转述在 `docs/COMPETITION_RULES.md`（D014）。逐日计划见 `docs/TIMELINE.md`。10-22 的原版清单在 `backups/snapshot_20261022-213717.zip` 里。

## 关键日期

- [x] **Kaggle 截止**：11:59 pm, Sunday 29 Nov 2026（规则 8）= **2026-11-29 23:59 HKT = 11-30 00:59 JST**（D006 在 Kaggle 页面核实；2026-11-01 的 Overview 页按 JST 显示截止日为 11-30）。迟交的提交 Kaggle 不收，事后也不能补。
- [ ] 课程最终截止（Canvas 上交 zip）：按 2026-11-29 23:59 HKT 计划；以 Canvas 显示的为准——**T**（Canvas 需要队员用 NetID 登录）
- [ ] Team registration（Canvas，截止 2026-10-16，已过）：我们是否登记了——**T**，Canvas 核对
- [ ] **Canvas checkpoint 日期：TODO**——**T**，Canvas。checkpoint 里写明每个成员的 **Kaggle display name**（规则 1）；**队伍成员在 checkpoint 截止时固定**（规则 2）
- [ ] **Individual reflection PDF 截止时间：TODO**——**T**，Canvas

## 评分权重（Marking）

| Criterion | Weight | 性质 |
|---|---|---|
| Core task + final solution + validation protocol（LB standing 按 upper / middle / lower third 分档计入） | 30% | 团队 |
| Extensions | 30% | **个人**（各自的扩展单独打分） |
| Report & analysis | 30% | 团队 |
| Video | 10% | 团队 |

规则 7：leaderboard 名次只占分数的一小部分，细节在 Canvas。

---

## A. Core task（30%）

- [x] 最终候选已提交到 Kaggle：S003（直推式 GMM，public 0.99923）、S005（严格归纳 QDA，public 0.99953）、S006（自训练 QDA-LW，public 0.99956）（`registry/submissions.csv`）
- [ ] 在 Kaggle Submissions 页勾选 2 个最终提交（按 D010/D013，默认 S003 + S005）；`record_lb.py --sub S### --final yes`；在 `decisions.md` 写最终选择条目——**T**，11-23 至 11-25。不勾的话 Kaggle 会自动选（Foundational 18c）
- [x] **validation protocol** 已写：`docs/VALIDATION_PROTOCOL.md`（D002：stratified 5-fold × 20，seed 2026）。E1 证据：run `20261022-064054_e1_validation_protocol`；5 × 20 是唯一 MDD 的 95% 置信区间整个低于 0.01 的方案（`reports/extensions/e1_validation_protocol.md` 表 1）
- [x] 冻结 D002：D015（2026-11-01，proposed）确认 D002 不变，并记下 §5 的选择规则没有执行、由 D010 取代；`docs/VALIDATION_PROTOCOL.md` 的状态行、§2 的 `TODO(E1)`、§5、§6 已补。D015 的团队确认见下一条
- [x] **没有泄漏**：所有预处理和特征选择都在 pipeline 内，每折重新拟合（`docs/VALIDATION_PROTOCOL.md` §2）。标签置换检验带阳性对照（R900：`20261022-091506_r900_perm_null_a`、`20261022-092943_r900_perm_null_b`）；id、行序、重复行都没有信号（R901）（`research/reports/verify_leak.md`）。测试集特征只进入无监督步骤（D007；规则依据见 D014）
- [x] **最终选择规则事先写好**：D010（在看到 S005/S006 的分数之前写下）+ D013（第 3 种情况）；public 分数不改变规则（D010 第 5 条）
- [ ] D010、D013、D015 的团队确认（现在是 proposed）——**T**
- [ ] 私下、笼统地问讲师（截至 11-01 还没有问；D015 第 4 条）：(a) 能否在测试集的无标签特征上做无监督拟合（S003；D007）；(b) 能否用测试行上由算法产生的伪标签做自训练（S006）；(c) 能否用预训练的表格基础模型（I014）。答复记为新决定：(a)(b) 都允许 → S003 + S006；只允许 (a) → 默认 S003 + S005；都不允许 → D013 的第 3 种情况——**T**
- [ ] 最终候选的 "clean git commit" 条件（`docs/VALIDATION_PROTOCOL.md` §5.1）：D015 第 3 条提议随 §5 作废，代码来源改由代码包和逐位复现证明（R910；S003 的 config 路线 10-22，S005 的 README 路线 11-01）。队员确认 D015，并决定第一次 commit 之后要不要在 Colab 上重跑最终候选——**T**（commit 只能由队员做）
- [x] 任务本质：二分类，250 行训练（113 个正例）、19,750 行测试、150 个数值特征，ROC AUC，提交 `id,target`，target 为分数（`scripts\verify_data.py`；Data 页说明，见 `docs/COMPETITION_RULES.md`）
- [x] public LB ≈ 2% 测试行（≈ 395 行），private = 其余部分，截止后公布（规则 3、Foundational 7a）；SE 表在 `docs/VALIDATION_PROTOCOL.md` §4 和 `scripts\lb_noise.py`。报告里要写明 **large gaps between validation, public and private scores are not a failure**（B1）

## B. Document（一个文件 = core report + extensions + contribution table）

生成 PDF：`.venv\Scripts\python.exe scripts\build_report.py`，它会打印每一部分的页数（`reports/README.md`）。

### B1. Core report（≤ 6 pages，English）——Report & analysis 30%

现状：AI 在 2026-11-01 写完 `reports/report_core.md` 全稿，并按三个 AI 审稿人的 77 条意见改过（`research/reports/review_round1.md`）；`build_report.py` 量出 6 页（用了 5.77 页）。队名、成员名字、成员核对说明和 Kaggle 勾选仍是 TODO。用到的材料：`reports/tables/r440_core_numbers.csv`（报告要用的 242 个数字的核查表，235 个与档案一致，7 个不一致的已列出）、`reports/tables/r440_journey.md`、`reports/tables/core_submission_history.md`（`scripts\submission_history.py` 生成）、`research/reports/verify.md` §2（claims table）。下面各节在成员改写或认可、并且能解释每一句之前，都不打勾。

- [ ] **Problem formulation**——A→T（`scripts\verify_data.py`；R001、R002，见 journal 10-22）
- [ ] **Validation protocol**——A→T（`docs/VALIDATION_PROTOCOL.md`、E1、D002、D010、D013）
- [ ] **Submission history**：多次提交的过程和 **what was learned**，不只是最终模型——A→T（`core_submission_history.md`、`r440_journey.md`、`registry/beliefs.md` B001–B008、journal）
- [ ] **Final solution**——A→T（`configs/final_s003_gmm2_hv.yaml`、D007、D010、D013、D014、README "Reproducing the final submission"）
- [ ] **Analysis**（包括 negative results）——A→T（`verify.md` §2、`registry/ideas.md` 的 rejected、没人负责的扩展作为支撑）
- [ ] **Citations**：报告的 References 已列出 `registry/references.md` 里全部 status = used 的条目（正文引用的一份、其余一份），软件行有版本号（2026-11-01）——**T** 核对
- [ ] **Generative AI disclosure**：`registry/ai_usage.md` 的草稿段落——A→T；段落里的 `[TODO]` 和表里的 Member / Verified by 两栏——**T**
- [ ] 页数 ≤ 6（`build_report.py` 打印的 core 页数；2026-11-01 是 5.77 页，Figure 1 上面只剩约 2.6 行空间，成员改写后每次重查）

### B2. Extensions（每个 ≤ 2 pages，个人 30%）

现状：四份草稿都在 `reports/extensions/`：标题正好是 Question / Protocol / Results / Conclusion，各有表或图，有可见的 AI 说明（11-01 按审稿意见 C2 写清 AI 做了什么，留了负责人 TODO）和引用。`build_report.py` 量出都在 2 页以内（页数见 `research/reports/deliverables_round1.md` 之后的 2026-11-01 journal 条目）。作者都还是 TODO，负责人还没有重跑或改写。

- [ ] 每个成员 **own ONE different extension**：成员数 = 计分的扩展数（1–3 个）；没人负责的扩展只能作核心报告的支撑材料——**T**：定分工，写进 `project.yaml` 和贡献表
- [ ] 每个扩展 **headed with the member's name**（作者行现在是 TODO）——**T**
- [x] 结构 Question / Protocol / Results（with a table or figure）/ Conclusion：四份草稿都符合
- [x] ≤ 2 pages：草稿试排符合（见上）
- [ ] 负责人读代码，可选在 Colab 上重跑，用自己的话改写，填 AI 说明行里的 TODO；改写后用 `build_report.py --parts e# --out $env:TEMP\e#.pdf` 重查页数——**T**。`fix_round1.md` §7 的遗留项已处理：E3 的 README 和 run 结论（10-22），E2 图例改成 "nominal 90%"、E4 图按印刷尺寸重画（11-01，`run.py --replot`）

**E1 Validation protocol design**（`reports/extensions/e1_validation_protocol.md`；run `20261022-064054_e1_validation_protocol`、`20261022-064148_e1_validation_protocol_qda_pair`）
- [x] 比较 hold-out（20%、30%）与 k-fold（k = 3, 5, 10），另有 5-fold × 10、× 20
- [x] 每个 scheme 50 个 split seed，**只改变 split 的 random seed**（`shuffle=True`；另有 `shuffle=False` 对照）
- [x] **固定模型 seed**
- [x] 报告估计值的 **spread**：sd(A − B)、MDD 及其 95% CI、p_correct
- [x] 回答 **which scheme would you trust to choose between two models that differ by 0.01 AUC?**：stratified 5-fold × 20，两个模型用同一组 split

**E2 Learning curve**（`reports/extensions/e2_learning_curve.md`；run `20261022-070248_e2_learning_curve`，另有 QDA、GMM 两个 run）
- [x] 训练集大小 50, 100, 150, 200, 250
- [x] 每个大小 10 个随机子集
- [x] 每个大小的 **mean** 和 **how widely they differ**（std、min、max）
- [x] **Extrapolate** 到 500 行
- [x] 说明外推的 **confidence**：名义 90% 区间在复刻上只覆盖 21%（R942），所以用 envelope 和约 ±0.04

**E3 Information leakage**（`reports/extensions/e3_leakage.md`；run `20261022-063908_e3_leakage`）
- [x] Filter：按 |corr| 保留 k 个特征
- [x] 方式 (1)：用全部 250 行选一次，再做 k-fold CV（leaky）
- [x] 方式 (2)：每一折只用该折训练部分重新选（honest）
- [x] 测量 (1) 高估多少，并与 **leaderboard score** 比较：S004（public 0.79136，预登记 B005）
- [x] 回答 gap 随 k 怎么变、为什么
- [x] k_feat 和 k_fold 两个维度都覆盖

**E4 Stability of the feature ranking**（`reports/extensions/e4_feature_stability.md`；run `20261022-071414_e4_feature_stability`）
- [x] 2000 次 bootstrap（250 行有放回抽 250），按 |corr| 排序，不拟合模型
- [x] 每个特征进入 top k 的频率，对照 chance rate k/150
- [x] 150 个特征的频率降序图，k = 5, 10, 13, 20
- [x] 回答是否 **drops sharply**、位置是否依赖 k（null 对照 R943）
- [x] 检验：陡降点以上的特征 vs 全部 150 个
- [x] 回答 being stably selected 与 being worth keeping 的关系

### B3. Contribution table（1 page）
- [ ] **maps every section and every extension to a member**（`reports/contribution_table.md`：骨架已在 11-01 做好，有 Evidence 和 AI use 两栏，AI use 引用 `registry/ai_usage.md` 第 1–21 行；名字、学号、display name、ID 和分工只能由成员填）——**T**

## C. Code
- [x] 足以复现最终提交：`src/ssc/`、`scripts/`、`configs/final_s003_gmm2_hv.yaml`、`requirements-lock.txt`、`README.md`（"Reproducing the final submission"）、`docs/SETUP.md`
- [x] 按 README 的 config 路线复现 S003（2026-10-22，Colab）：run `20261022-212507_final_s003_gmm2_hv`（作业 `J20261022-212641-pro1-ssc`），100/100 折 AUC 1.0，split signature `5708ce5c3828`，test_pred 与 S003 的 run 逐位相同，`make_submission.py --dry-run` 的 sha256 前缀 `3aae140e5837753f` 与 registry 的 S003 一致（D013 第 4 条）。另有新 VM 上的逐位复现 R910（`20261022-091211_r910_s003_gmm2_hv`）。在 Windows 笔记本上只一致到约 1e-11（README 已说明）
- [x] 第二个最终选择 S005（r201）的复现命令已写进 README（"Second final selection: S005"）：`colabemote.py run -s repro -- research/r201_sup_quadratic.py --only qda_block`。2026-11-01 按这条路线在 Colab 上重跑：run `20261101-123515_r201_qda_block`（作业 `J20261101-123642-pro1-repro`），fold AUC、oof、test_pred 和每折选的参数都与原 run 逐位相同，`make_submission.py --dry-run` 的 sha256 前缀 `3be09c0723286457` 与 registry 的 S005 一致。另有 R910 的新 VM 复现 `20261022-091224_r910_r201_qda_block`
- [ ] 第一次 git commit + 私有远端（D004、`docs/ARCHIVE_RULES.md` §9）——**T**
- [x] 文件夹改名（10-24）之后：editable 安装的路径和 Jupyter kernel "Python (ssc)" 已在 11-01 改到新路径（`docs/SETUP.md` §1 "项目文件夹改名或移动之后"）。`.venv\Scripts\*.exe`（pip、pytest、jupyter、kaggle）还记着旧路径，所以命令一律用 `.venv\Scripts\python.exe -m <模块>`

## D. Video（10%）
- [ ] **≤ 5 min regardless of team size**
- [ ] **all members presenting, faces shown**
- [ ] **≤ 300 MB**
- [ ] 讲稿（≤ 600 词）和幻灯片：AI 在 11-01 起草了 `video/SCRIPT.md`（3 / 2 / 1 人版，约 400–530 词）、`video/slides.tex` → `slides.pdf`（16:9，14 张）和录制指南 `video/RECORDING.md`——**T**：填名字和分工、改写成自己的话、计时排练；录制——**T**（本机没有 ffmpeg，可用 Clipchamp）

## E. Zip（每队一个）
- [ ] **one zip per team, named after the team**（`project.yaml: team.name` 还是 TODO，会挡住 `package_deliverables.py --build`）——**T**
- [ ] 内含：video + document（core report + extensions + contribution table）+ code
- [ ] 不含 Kaggle 数据、token、`sandbox/`、`.venv/`
- [x] 不含个人邮箱：`package_deliverables.py` 打包时把 `project.yaml` 里的地址换成 `<removed>`（本地文件不变，`colab/remote.py` 只用账号名 pro1 / pro2）；预检会把其他含邮箱的文件标成 CHECK（2026-11-01：没有）。`colab/README.md` 里的已在 2026-11-01 去掉

## F. Individual reflection（每人一个 PDF，单独在 Canvas 提交）
- [ ] (1) main role，owned 哪个 extension——**T**
- [ ] (2) 为 extension / final model 考虑过的其他 ideas，被团队 accepted or rejected 及原因——**T**
- [ ] (3) 团队和你个人的 main difficulties——**T**
- [ ] (4) one thing you believed about your model at the start that the leaderboard later showed to be wrong——**T**
- [ ] **Missing reflection ⇒ zero for the extension criterion**（模板：`reports/reflections/TEMPLATE.md`；事实和出处：`reports/reflections/FACTS.md`，只作材料，不能照抄）
- [ ] 每人在 `registry/beliefs.md` 写自己的预测、在 `registry/journal.md` 写自己的困难和想法。现在 B001–B008 和全部日志都是 AI 写的，Q2–Q4 需要成员自己的材料——**T**
- [ ] 核对反思 PDF 的 Canvas 截止时间是否晚于 private LB 公布（2026-11-29 23:59 HKT 之后）：晚于才能在 (4) 里用 private 分数，否则只用 CV / expected / public（`docs/ARCHIVE_RULES.md` §5）——**T**

## G. Competition rules（全程遵守；转述见 `docs/COMPETITION_RULES.md`）
- [x] **Max 5 submissions per day**（规则 3；Kaggle 日在 00:00 UTC = 09:00 JST 重置）；队规 ≤ 3（D008）：到 11-01 共 6 次，每个 Kaggle 日最多 3 次（`registry/submissions.csv`）
- [ ] **队伍 2 或 3 人，或个人**；成员在 checkpoint 截止时固定；每人一个 Kaggle 账号、只属于一个队（规则 2；Foundational 5a–5c）。队友先加入比赛，再在 Team 页受邀并接受（合队）；合队之前不共享仓库、预测和分数。现状：Kaggle 队伍只有 1 人（display name "Yongzi CHEN"）——**T**
- [ ] Canvas checkpoint 写明每个成员的 **Kaggle display name**（规则 1）——**T**
- [x] 只用比赛数据训练（规则 4）：测试集特征属于 Competition Data（Foundational 18a）；复刻数据不训练要提交的模型（D010 第 6 条）；没有外部数据（D007、D014）
- [x] 没有人工标注或人工预测测试行（Foundational 4b；D014）
- [ ] **everything used must be cited**（规则 4）：`registry/references.md` 已补全；核心报告的 References 已列出全部 status = used 的条目（11-01）——**T** 核对（B1）
- [x] **No private sharing outside the team**（规则 5；Foundational 5d、6a）：仓库不公开，Colab VM 只有我们自己的账号能访问（`research/reports/verify_rules.md` §1）。队友合队之前同样不共享（见上）
- [ ] 生成式 AI 在报告里 **disclose**（which tool, for what；规则 6）：草稿段落在 `registry/ai_usage.md`；**members must be able to explain every detail**——**T**（讲解会，见 `docs/TIMELINE.md`）
- [x] 开源许可证（Foundational 6c）：最终管道只用 BSD-3-Clause 的 numpy、scipy、scikit-learn、pandas，依赖是 BSD、MIT 或 Apache-2.0（2026-11-01 核查，D014）；TabPFN / TabICL 不进任何提交（D010 第 4 条）
- [x] No GPU needed

---

## 提交前最终检查（11-28 完成，11-29 只做补救）

- [ ] Kaggle：最终提交已按 D010/D013 勾选；`record_lb.py --sub S### --final yes` 已记录；`decisions.md` 有最终选择条目
- [ ] `registry/submissions.csv`：每一行都有 `public_lb`；没有空的 `expected_public`
- [ ] 报告：`scripts\build_report.py` 打印的页数 core ≤ 6、每个扩展 ≤ 2、contribution table 1；英文；所有图表在 100% 缩放下可读
- [ ] 报告里没有残留的 `TODO`（与 `reports/README.md` 同一个列表；只列自己队伍负责的扩展文件，`reports/README.md` 本身不在内）：`Select-String -Path reports\report_core.md, reports\contribution_table.md, reports\extensions\*.md -Pattern TODO`
- [ ] 每个扩展：标题下有成员姓名；四个标题正好是 Question / Protocol / Results / Conclusion；有表或图
- [ ] 报告中的每个数字都能追溯到一个 run_id 或 sub_id；没有 `sandbox/` 结果
- [ ] References 覆盖所有库和方法；AI 披露段落已写，每一句都属实
- [ ] 视频 ≤ 5:00、≤ 300 MB、每个成员露脸讲解（`video/PLAN.md`）
- [ ] 打包文件里没有个人邮箱；`.venv\Scripts\python.exe scripts\package_deliverables.py` 预检通过，再加 `--build` 生成 zip；zip 文件名 = 队名
- [ ] 解压 zip 检查一遍：文档能打开、视频能播放、代码目录完整
- [ ] 每个成员的 reflection PDF 已单独提交到 Canvas
- [ ] 截止后：所有提交的 private 分数已记录（`record_lb.py --private`），B004、B008 填 outcome
