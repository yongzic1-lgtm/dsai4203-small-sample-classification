# Journal（工作日志）

规则（`docs/ARCHIVE_RULES.md` §7）：按时间顺序在文件末尾追加，每次一个 `## YYYY-MM-DD — <author>` 条目。**困难一定要写**（卡在哪、怎么解决、花了多久）：反思 Q3 只能从这里取材。引用 run_id / sub_id / D### / I###。

## 模板

```markdown
## YYYY-MM-DD — <author id>
- Did: 做了什么（run_id / sub_id）
- Result: 关键数字和结论
- Difficulties (team / me): 遇到的问题、怎么解决、花了多久
- Next: 下一步
```

---

## 2026-10-22 — TODO（setup，使用 Claude Code）
- Did: 建立项目环境和归档制度。隔离的 venv `.venv`（Python 3.11.5，锁定版本见 `requirements-lock.txt`），Jupyter kernel "Python (ssc)"；下载 Kaggle 数据到 `data/raw/`，写入 `SHA256SUMS.txt` 并设为只读；用 seed 2026 生成合成数据 `data/synthetic/`；共享包 `src/ssc/`、`scripts/`、`extensions/e1..e4` 脚手架；归档制度与文档（`docs/`、`registry/*.md`、`reports/` 模板、`video/PLAN.md`）。
- Result: 数据：250 行训练集（113 个正例，45.2%），19,750 行测试集，150 个特征。按 Hanley–McNeil 公式，395 行 public LB 在 AUC 0.85 时的 SE ≈ 0.020（`docs/VALIDATION_PROTOCOL.md` §4）。
- Difficulties (team / me):
  - conda env 混进了用户级 site-packages（`%APPDATA%\Python\Python311`）中的旧包 → 改用 venv（D001）。
  - sklearn 1.9 弃用了 `LogisticRegression(penalty=...)` → config 里改用 `l1_ratio`（0 = L2，1 = L1）配合 liblinear / saga。
  - 这台机器没有 git 身份，用户要求暂不 commit → D004；第一次 commit 之前不做正式 run。
- Next: 每人填 `project.yaml` 的成员信息，设置 git 身份，做第一次 commit；确认 Canvas checkpoint 日期；EDA；baselines；首批提交（`docs/TIMELINE.md` 第 0 周）。

## 2026-10-22 — TODO（研究第 1 天，使用 Claude Code；计算在 Colab）
- Did: 搭好 Colab 执行链（D005）：用 WSL 里的官方 Colab CLI，加上 `colab/remote.py`，pro1 / pro2 两个账号都已登录。在 Colab 上跑了 R001 数据结构、R002 信息子空间、R003 直推式与监督式模型对比（8 个候选，都用官方 5×20 CV）。之后提交了 S001–S003。
- Result:
  - R001：13 个特征的方差明显偏大（8, 15, 26, 30, 46, 49, 64, 67, 77, 80, 98, 100, 112），相关矩阵有 8 个特征值接近 0，train 和 test 分布没有差异。
  - R002：在这 13 个特征上用 20,000 行无标签数据拟合 2 分量 GMM，两个分量的训练集正例比例分别是 0.000 和 0.991。"噪声"特征里也有和目标相关的（最大 |corr| 0.39，纯噪声的预期约 0.20），说明按方差选的特征集还不完整。
  - R003：直推式 GMM 的 CV AUC 为 1.0000（100 个 fold 全部是 1.0）。监督模型中，QDA 0.9958，L2 逻辑回归 0.8977，LDA 0.8779，L1 逻辑回归（全部特征）0.8469。QDA 远好于 LDA，说明两类的差别主要在协方差，不在均值。
  - 公开榜：S001 0.83589，S002 0.99277，S003 0.99923（公开榜第 1 名）。
  - 测试集上约 1%（约 200 行）的 GMM 后验落在 0.1–0.9 之间，后面能不能再提高就看这些边界样本。
- Difficulties (team / me):
  - Colab CLI 不支持 Windows，改装在 WSL 里；WSL 的 Ubuntu 没有 pip，而且 sudo 要密码，最后用已有的 micromamba 建了环境。
  - 第一个 Colab 会话（美国东部机房）的 websocket 经常卡住，`colab exec` 一次要等 75 秒以上。新建的会话（亚洲机房）每次只要 2–3 秒。为此改了 remote.py：日志只走一条长连接，断开后从断点续传；状态检查合并成一次调用；作业启动前如果会话没有响应就自动重建（约 1 小时）。
  - pandas 在 Windows 和 Linux 上写 CSV 的换行符不同，合成数据的哈希对不上，改成按内容比较。
  - gzip 头里带时间戳，导致每次都重新上传数据，已修复。
  - 有一小段时间代码里临时用的是 HKT 时钟，R003 的 run id 因此早了 1 小时（D006，不改名）。
  - 工作流的修复代理被意外复制出一个实例，两个同时改 `src/ssc`，停掉了其中一个。
- Next: 并行研究：复刻生成器、优化混合模型对边界样本的排序、监督模型对照组、预训练和贝叶斯模型、在真实数据上完整跑 E1–E4。每天最多提交 3 次。

## 2026-10-22 — TODO（研究第 1 轮，Claude Code 工作流，6 个探索方向，全部在 Colab 上跑）
- Did: 工作流 `ssc-research-round1`（run id `wf_13564004-279`）的 Explore 阶段：gen（生成器反推，pro1/rgen）、mix（混合模型精修，pro1/rmix）、sup（监督模型对照，pro2/rsup）、fm（基础模型和贝叶斯方法，pro2/rfm，L4 GPU）、ext12（E1+E2，pro1/rext12）、ext34（E3+E4，pro1/rext34）。各方向的报告在 `research/reports/*.md`，扩展报告草稿在 `reports/extensions/e1–e4*.md`。**Verify 阶段按用户要求没有运行**（用户要重启电脑），verify 刚启动就停掉了，没有产出。
- Result:
  - 生成器：`make_classification(n_informative=12, n_redundant=8, n_clusters_per_class=1, class_sep≈0.71, hypercube=True)`，之后每列做仿射变换 x·a+b（a~U(0.5,3)，b~U(-5,5)）。informative 列 = {6,15,30,46,52,61,64,80,100,112,142,144}，redundant 列 = {8,17,26,49,57,67,77,98}，其余 130 列是噪声。复刻库：`research/lib_gen_replica.py`。
  - S003 离贝叶斯上限（0.99995）只差约 2e-7；没有任何改进在复刻数据上显著超过它。public 0.99923 说明存在少量标签噪声（约 0.2%），所以 S003 的 private 期望值约 0.998，剩下的误差排序修不掉。
  - 监督 QDA 调好后与 GMM 不可区分（D009）；TabPFN v2 0.985–0.989，TabICL 0.993–0.999，只作为分析过的对照（规则 4 的疑问）；horseshoe logistic 0.926 是最好的线性模型。
  - E1：只有官方 5×20 方案能分辨 0.01 AUC 的差距（MDD 0.0061 [0.0051, 0.0076]），支持 D002。E2：L1 基线外推到 500 行约 0.906；QDA 约 0.997；GMM 在每个样本量下都是 1.0。E3：泄漏让 CV 虚高 0.073 ± 0.014（k_feat=20），E3 的 LB 对比还需要一次提交。E4：在第 7–8 个特征处有明显断崖，k≥8 后位置不随 k 变；最稳定的特征集并不是最值得保留的特征集。
- Difficulties: `remote.py` 有两个 bug：`prepare()` 漏了 return；`wsl --` 会经过外层 shell，把 pro2 前缀里的 `$A`、`$C` 提前展开成空串（改用 `wsl -e`）。ext34 因为第一个 bug 用了 `--no-sync`（代码包 `b413e6aa3bae4443`）。Colab 上另有 sb1–sb7 共 7 个 High-RAM 会话在跑，不属于本项目（compute_jobs.csv 里没有它们），没有去动。
- Next（重启后）：(1) 跑 Verify（核查、复现、规则合规）；(2) 提交 E3 最终模型做 LB 对比（`20261022-063908_e3_leakage`，预期 public 0.820 ± 0.021）；(3) 考虑提交诚实的监督 QDA（`20261022-071234_r201_qda_block`，预期约 0.998–0.999）作为提交历程的一环；(4) 团队信息、git 身份、第一次 commit。

## 2026-10-22 — TODO（重启电脑前归档）
- Did: 新增只读检查脚本 `scripts/archive_check.py`，已写进 ARCHIVE_RULES §12。补写了 R001–R003 共 10 个 run 的 Conclusion。打了快照 `backups/snapshot_20261022-084506.zip`（1890 个文件，46 MB，不含比赛数据和 .venv）。
- Result: 0 problem，3 个 warning 都能解释：2 个 run 按失败状态归档（原因已写在 notes.md）；sup.md 里的 `r20x_*` 是通配写法，不是缺文件。共 162 个 run，registry 有 162 行，52 个 Colab 作业，3 个提交，引用了 19 个代码包，都已归档。
- Next: 重启之后接着做本日 journal 里列的 Next（verify、E3 提交、团队信息、git）。

## 2026-10-22 — TODO（Verify 第 1 轮与更正，使用 Claude Code）
- Did:
  - Verify 第 1 轮：6 个 AI 核查代理（leak、repro、gen、cand、ext、rules）在 Colab 上独立复查第 1 轮的结论。新 run 是 R900–R944，共 36 个 run 文件夹（`experiments/20261022-091033_*` 到 `20261022-172258_*`，含 1 个失败的）。R945 是只读的引用清单脚本，不建 run。汇总代理把 6 份报告合成 `research/reports/verify.md`（结论表、更正清单、提交计划、最终选择）。
  - 16:08 JST 上传 S004（E3 最终模型，事先登记了 B005）。
  - 主会话写了 B005–B008 和 D010（最终选择规则，在看到 r201 / r209 的 public 分数之前写下）。建好了 S005（r201_qda_block）和 S006（r209_selftrain_qda_lw）并预登记（B006、B007）。写这条日志时它们还没有上传。
  - 本次更正（`verify.md` §4）：D011（更正 D009）、D012（补充 D005 / D006）、重写 `ai_usage.md` 的披露段落并修好表格、`references.md` 加 39 条并把 9 条改成 used、`ideas.md` 更新 I001 和 I003–I008 的状态、本条日志。
- Result:
  - 泄漏：R900 做了两种标签置换检验，都带阳性对照。Null A 下，诚实流程的均值是 0.477–0.496，没有一次超过 0.6；E3 泄漏对照是 0.697，20/20 次超过 0.6。Null B 下，读盘作弊对照 20/20 次是 1.000；诚实流程的均值是 0.480–0.573（GMM 类是双峰，因为标签只决定分量方向）。Run：`20261022-091506_r900_perm_null_a`、`20261022-092943_r900_perm_null_b`。R901 没有发现 id、行序、重复行的信号：AUC(id, y) 0.552（p 0.16），0 个完全重复行，adversarial AUC 0.499（`20261022-094812_r901_id_order_dupes`）。
  - 复现：S003、E3 最终模型和 r201 在新的 Colab VM 上逐位复现（R910 / R911）。r209、r210 被独立重写的代码逐位复现（r932）。笔记本和 Colab 之间，S003 的 test_pred 最多差 4.4e-11，提交 CSV 里只有 1 个值不同。笔记本时钟比 VM 快约 100 秒（D012）。
  - 生成器和 Bayes 差距：R920 不看标签，独立地得到同一组信息列（领先 13.0 nats）；class_sep 0.70（约 0.63–0.78）。R921：S003 比 Bayes 上限低 2.57e-7（约 2.6e-7，1000 个复刻里 100% 为正），相当于约 9,400 万对样本里排反 24 对。
  - 标签噪声和 private：R922 的 flip_y 似然峰值在 0.003，支持范围 0–0.015。flip_y = 0 被压低约 5:1（LR 0.20），但没有被排除。S003 的 private 预测（均匀先验）：中位数 0.9973，90% 区间 [0.9923, 0.9996]，也就是约 0.997（`reports/tables/r922_posterior_by_prior.csv`）。P(private ≥ 0.999) 约 0.15，所以 B004（70%）过于自信，已另登记 B008。
  - 候选：r936 的 222 个复刻上，没有候选显著超过 S003（与 S003 的 private 配对差：r209 −6.9e-5，r255 −1.16e-4，r210 −1.68e-4，r201 −2.57e-4；`reports/tables/r936_private.csv`）。第二个最终选择最多只多 1.45e-5 的期望 AUC。D009 的"不可区分"字面上对，但没有信息量：同样的 CV 检验 222/222 次不显著，而监督 QDA 的真实 AUC 每次都更低。措辞已由 D011 更正。r210 的 reg 0.001 是网格下边界（r930）。
  - 扩展：E2 的外推区间名义上是 90%，在复刻上实际只覆盖 21%（R942，偏差 −0.031）。四份草稿排版后 2.16–2.71 页（E1 2.55–2.66，E2 2.57–2.71，E3 2.16–2.22，E4 2.19–2.23），都超过 2 页；有 15 处数字和档案对不上（`research/reports/verify_ext.md`）。
  - 规则风险：用测试集特征（D007）还没有规则原文或讲师的确认（规则原文没有存档，"Foundational Rules 4b" 找不到出处）。TabPFN / TabICL 只能作分析对照。pro2 账号归属和 Kaggle 合队还没有确认。另外，`ai_usage.md` 有一句与事实不符的话（本次已改），引用缺 39 条（本次已补），单元测试没有覆盖 TransductiveGMM，全部 run 都是 no-commit（核查时 191/191），README 的复现路径走不通（`research/reports/verify_rules.md`）。
  - S004：public 0.79136。比泄漏 CV（0.867）低 0.075，比诚实 CV（0.793）只低 0.002。B005 confirmed。
- Corrections（只追加，不改旧条目）:
  - 本日"研究第 1 轮"条目里的"public 0.99923 说明存在少量标签噪声（约 0.2%），所以 S003 的 private 期望值约 0.998"作废，改为：flip_y 似然支持 0–0.015（MLE 0.003），S003 private 约 0.997（90% 0.992–0.9996）（R922）。同一条目里的"监督 QDA 调好后与 GMM 不可区分（D009）"按 D011 理解。
  - `20261022-063712_r300_block_l2_c01` 和 `20261022-063713_r300_block_l2_c1` 的 `config.yaml` 第 37 行写着 `uses_unlabeled_test_features: false`。但它们用的固定 13 列块是在 train+test 特征上无标签地选出来的（`research/lib_ext12.py:36`）。run 文件夹不改（只追加）。报告里把这两个 run 算作 B 组（用了测试集的无标签特征），不算严格监督。
  - D010 写下了最终选择规则：默认 S003 + r201；讲师明确允许用测试集特征，才换成 S003 + r209；讲师明确禁止，就改选两个严格归纳的提交；任何情况下都不选 FM run 和 S004；不反推生成器的种子或隐藏标签。
- Difficulties (team / me):
  - cand 核查上午被用量上限打断。下午接着做，沿用了已有的脚本和 run（r930、r932、r933 s0-7），没有重做（`research/reports/verify_cand.md` 开头）。r933 的复刻从 09:44（s0-7）一直停到 16:05（s8-71）才继续。
  - R900 Null A 第一次失败：阳性对照用了带 lru_cache 的模块级函数，loky 子进程无法反序列化（BrokenProcessPool）。改成在 fit 里读 train.csv 后重跑。失败的 `20261022-091033_r900_perm_null_a` 按 failed 归档。
  - rules 核查想跑 `scripts/package_deliverables.py` 的 dry run，被权限系统挡住，没有跑成。
  - 核查者之间有分歧：S003 的 private 预期、第二个最终选择、E3 的 ±0.014 是 SD 还是 SE。汇总时逐条裁决（`research/reports/verify.md` §3）。
  - 用时（按 run id）：核查 run 在 09:10–10:06 和 16:05–17:23 JST；汇总约 17:55–18:30。
- Next:
  - 主会话上传 S005、S006 并 record_lb。规则见 D010，分数不改变选择。
  - 队员：存档 Kaggle 规则原文，私下问讲师测试集特征（D007）和预训练模型两个问题；确认 D007 和 D009–D012；填 `ai_usage.md` 的 Member 和 Verified by；设置 git 身份并做第一次 commit；同步 Windows 时间；确认 pro2 账号属于谁；在 Kaggle 合队。
  - 扩展负责人：按 `verify.md` §4 第 15–19 条改数字，加一行可见的 AI 说明，每份删 250–400 词。
  - 维护者：修 README 的复现路径；给 TransductiveGMM 加单元测试；让 `remote.py` 在 `--no-sync` 下也记录代码包。

## 2026-10-22 — TODO（修正轮 + 研究第 2 轮，使用 Claude Code；计算在 Colab）
- Did: 工作流 `ssc-fix-and-research2`（`wf_5abda016-8ac`，9 个代理）。修正：4 个代理按 Verify 汇总的 29 条更正改了扩展草稿、registry 和第 1 轮研究报告；研究第 2 轮在 Colab 上跑了 R400–R404（严格归纳的最终选项、边界行排序）、R420–R426（flip_y 网格到 0.03、分层 public、线性上限）、R440–R441（核心报告数字核查、标签预算重跑）；fix_code 改了代码和测试；终检写了 `research/reports/fix_round1.md`。主会话：D010、D013，B005–B008，S004 上传，S005/S006 建好，`kaggle_submit.py` 加队规上限，按新 config 在 Colab 上重跑 S003。
- Result:
  - 29 条更正：25 条已做，3 条部分完成（ai_usage 的成员签字、扩展草稿的作者、S003 复现路线），1 条（删 npz）按规则不做，改为不放进代码包。扩展草稿 XeLaTeX 试排都在 2 页以内（E1 1.94–1.97，E2 1.89–1.97，E3 1.74，E4 1.79–1.94）。pytest 214 个通过；archive_check 0 problem。
  - S003 private 预期（R922 平坦先验，222 个复刻扩到 flip_y 0.03）：均值 0.99683，90% [0.99218, 0.99958]，P(≥0.999) 0.16（R425），与 B008 一致。
  - 给定 S003 的 public 0.99923，S005（r201）的 public 预测 0.99868（B006 的点估计 0.9988 偏高 1–3e-4），S006（r209）0.99910（B007 准确）（R425、R421）。
  - 严格归纳的最佳组合：ext-grid QDA（r930）+ r201（R404），写进 D013。
  - D011 的两条证据都复核了：R441 重现 R206（严格监督写"80–120 个标签"）；R404 在 222 个复刻的边界行上 S003 0.804 vs r255 0.678（222/222）。
  - 核心报告会用到的 242 个数字中 235 个与档案一致（R440，`reports/tables/r440_core_numbers.csv`）；journey 表 `reports/tables/r440_journey.md`。
  - 按 README 新路线（`configs/final_s003_gmm2_hv.yaml`）在 Colab 上重跑 S003（`20261022-212507_final_s003_gmm2_hv`）：100/100 折 AUC 1.0，test_pred 与原 run 逐位相同，提交文件 sha256 前缀 3aae140e5837753f。
- Difficulties (team / me):
  - Verify 的 cand 代理和汇总代理碰到用量上限中断（约 10:15 JST），13:10 重置后从断点续跑（已完成的代理用缓存）。
  - pro2 的第二个会话两次分不到机器（同账号另有 4 个 sb* 会话，不是本项目的，没有动）；r2_inductive 改成在一个会话上顺序跑。
  - 本机 Kaggle CLI 没有凭据，上传靠 Chrome 网页。16:10 之后 Chrome 扩展断开；重连后这个 Chrome 没有登录 Kaggle，S005/S006 还没上传（等成员登录）。
  - 3 个 npz（12 MB）让每个代码包从 0.5 MB 涨到 13.5 MB，`logs/code_bundles` 到了 75 MB；现在单独走 research data 通道，代码包 1.3 MB。
- Next: 成员登录 Kaggle 后上传 S005、S006 并 record_lb（今天的 Kaggle 日到 10-23 09:00 JST，已用 1 次）；向讲师确认 D007；填团队信息、合并 Kaggle 队伍、确认 Canvas checkpoint；git 身份和第一次 commit；核心报告开写（数字从 `r440_core_numbers.csv` 取）。

## 2026-10-22 — TODO（S005、S006 上传）
- Did: 成员在 Chrome 里登录 Kaggle 之后，上传了 S005（r201，严格归纳 QDA）和 S006（r209，自训练 QDA-LW），都在上传前预登记过（B006、B007），最终选择规则 D010 也在看到分数之前就写好了。今天（Kaggle 日 2026-10-22 UTC）一共上传 3 次（S004–S006），正好用满队规的上限。
- Result: S005 public 0.99953（预期 0.9988，+0.6 SE），S006 public 0.99956（预期 0.9991，+0.4 SE），都落在预登记的 80% 区间内，而且都比 S003 的 0.99923 高。事先估计"高于 S003"的概率分别只有约 14–20% 和 27%（R425）。public 只有 395 行，0.0003 的差距只相当于约 12 对排序，在 public 差的噪声（sd 3–6e-4）以内；复刻上两者的 private 期望都比 S003 低（−2.4e-4、−6e-5）。按 D010，这两个分数不改变最终选择：默认仍是 S003 + r201（S005）。
- Difficulties: Chrome 扩展重连后连的是一个没登录 Kaggle 的窗口，等成员登录后才能上传。
- Next: 向讲师确认 D007；写核心报告时刷新 journey 表（`research/r440_core_numbers.py`：先 --pack，再在 Colab 上跑）。

## 2026-11-01 — TODO（交付物第 1 轮：核心报告、PDF、视频材料、反思素材；使用 Claude Code，计算在 Colab）
- Did:
  - 文件夹改名之后修好环境：10-24 项目从 `D:\polyu_class\ml` 改名为 `D:\polyu_class\新建文件夹`。editable 安装的 `.pth` 改到新路径（用 UTF-8 写；用 GBK 写会让 Python 起不来）；Jupyter kernel 重新注册。`.venv\Scripts\*.exe` 还记着旧路径，所以命令一律用 `python -m`。
  - 在已登录的 Chrome 里只读地看了 Kaggle 的 Rules、Overview、Data、Team 页。Canvas 要 NetID 登录，没有读到任何日期。
  - 工作流 `ssc-deliverables-round1`（`wf_f9aee159-d9b`，12 个代理，约 3 小时）：
    - 核心报告全稿，图表由 `scripts/make_core_figures.py` 生成；
    - PDF 构建脚本 `scripts/build_report.py`，附测试；
    - E2、E4 的图按印刷尺寸重画（`run.py --replot`）；
    - 反思素材 `reports/reflections/FACTS.md` 和贡献表骨架；
    - 规则转述 `docs/COMPETITION_RULES.md` 和 D014；更新清单、时间表和文档里的旧路径；
    - 视频讲稿、幻灯片和录制指南；
    - 三个只读的 AI 审稿人（数字、论断、评分标准）共提出 77 条意见，按意见改了核心报告；
    - 最终检查，结果在 `research/reports/deliverables_round1.md`。
  - 主会话收尾：
    - 审稿原文和处理结果存档，见 `research/reports/review_round1.md`；
    - 改了核心报告以外的 14 条意见：扩展的 AI 说明行和几处论断、贡献表的 "sole author"、FACTS.md 的三处错误、PLAN.md；
    - 讲稿和幻灯片里有同样的说法（审稿时它们还没写完），一并改掉；
    - 新增决定 D015；改了 D002、D007、D014 的 Status 行；
    - 打包时去掉 `project.yaml` 里的邮箱；
    - 在 Colab 上实测 README 里 S005 的复现路线。
- Result:
  - 报告 PDF（`reports/report.pdf`）共 15 页。扩展负责人还没定，所以四个扩展都在里面。各部分都在限额内：core 6 页（实际用 5.77），E1 1.89，E2 1.80，E3 1.64，E4 1.88，贡献表 0.94。
  - 视频幻灯片 14 张。讲稿最长的是 3 人版，533 词：严格估计 4:52，按 130 词/分钟约 4:06。
  - S005 按 README 路线在 Colab 上重跑（run `20261101-123515_r201_qda_block`，作业 `J20261101-123642-pro1-repro`）。100 折 AUC、oof、test_pred 和每折选的参数都与原 run 逐位相同。`make_submission.py --dry-run` 给出的 sha256 前缀 3be09c0723286457 与 S005 一致。没有写提交文件，也没有上传。
  - 审稿意见 77 条，改了 76 条。其中 C26 只改了一半，另一半与档案不符，没有采纳。U12 要等成员真正复核之后才能改。
  - pytest 248 个全部通过；archive_check 0 个问题（4 个是以前就有的警告）；共 215 个 run。
  - 规则：测试集特征属于 Competition Data（Foundational 18a）。规则 4 只禁止比赛以外的数据，4b 禁止的是人工标注测试行，所以 D007 的依据有了出处（D014）。讲师还没被问到；"仍在询问"的说法由 D015 第 4 条更正。
- Difficulties (team / me):
  - 工作流里几个代理同时改报告、构建 PDF，中间有一次构建出 7 页（Figure 2 的位置变了）；最后统一重建、重量页数。核心报告版面很紧，Figure 1 上方只剩约 2.6 行，成员改写后每次都要重查。
  - 工作流没有保存审稿人的原文（最终检查发现），由主会话从工作流结果里补存。讲稿和幻灯片不在审稿范围内，里面有和报告同样的问题说法（"covariance, not means"、"no candidate beats S003"），主会话按同样的意见改掉。
  - Colab 作业结束时 `colab exec` 超时两次，自动重试后成功取回了结果。
- Corrections（只追加，不改旧条目）:
  - `registry/ai_usage.md` 第 10 行的 "What for" 一栏在 10-22 22:23 被原地改写过（最终检查对照 10-22 的快照发现）。内容属实，但按规则这一栏不该原地改。这里记下，不再改回。
  - D014 正文第 2 条和 `docs/COMPETITION_RULES.md` 说讲师确认"仍在询问"，实际上还没有问过；由 D015 第 4 条更正。
- Next:
  - 队员（T）：
    - 定队伍和分工：填 `project.yaml` 的 team 和 `report.extensions_owned`，合并 Kaggle 队伍；
    - 用 NetID 在 Canvas 查 checkpoint 和反思的截止日；
    - 按 D015 第 4 条问讲师三个问题；
    - 确认 D002、D007、D010–D015；
    - 把规则页存成 PDF；
    - 设置 git 身份，做第一次 commit；
    - 改写核心报告和自己的扩展，填 AI 说明行里的 TODO；
    - 把讲稿改成自己的话，排练，录视频；
    - 每人写反思（FACTS.md 只作材料）；
    - 11-23 至 11-25 在 Kaggle 勾选最终提交。
  - 维护者：成员改写后，用 `build_report.py` 重建 PDF、查页数；分工定下后，按 `video/RECORDING.md` §2 重建幻灯片。

## 2026-11-15 — TODO（第一次 commit，推到 GitHub 私有仓库，使用 Claude Code）
- Did: 按用户要求把项目放到 GitHub，方便分享给组员（D016）。账号 `yongzic1-lgtm`，私有仓库 `dsai4203-small-sample-classification`。提交前检查了要提交的文件：没有 token 或密钥；`project.yaml` 里两个 Colab 账号的 Gmail 地址换成了标签；Colab 代码包（59/60 个带着旧 project.yaml 里的 Gmail）不进 git，只留在本机和快照里；`archive_check.py` 对没有代码包的克隆只给一条警告。git 身份只设在本仓库，用 GitHub 隐藏邮箱。
- Result: 见 D016 和第一次 commit。比赛数据、`.venv`、快照、`*.npy`、提交 CSV 都不在仓库里。
- Difficulties: 代码包里藏着旧的 project.yaml（含个人邮箱），压缩包用 grep 搜不到，要解包检查。
- Next: 组员把 GitHub 用户名发给仓库所有者，加为协作者；组员 clone 后按 `docs/SETUP.md` 建环境，从 Kaggle 下载数据，跑 `scripts/verify_data.py` 校验。

## 2026-11-15 — TODO（仓库改为公开）
- Did: 用户要求公开仓库。AI 先列出规则风险（Kaggle 6a/6b、课程规则 5，比赛 11-30 00:59 JST 才截止），给了三个选项；用户选择现在就公开。公开前检查了学号、密码、token 和个人邮箱，都没有。已改为 PUBLIC（D017）。
- Next: 组员可以直接 clone；如果以后要改回私有：`gh repo edit yongzic1-lgtm/dsai4203-small-sample-classification --visibility private --accept-visibility-change-consequences`（已有的 clone/fork 收不回）。

## 2026-11-23 — TODO（报告润色：Times New Roman、新图、核心报告改写，使用 Claude Code）
- Did: 按用户要求（"文章写好一点，图好看，丰富一点"；"字体用新罗马，图太单调"）跑了一个 12 个代理的工作流（ai_usage 第 24 行）：
  - 排版：`build_report.py` 全文改用 Times New Roman（正文、标题、表格、图注、代码片段），新增 `<!-- small -->` 标记（只用于参考文献，目前没有用）。
  - 核心报告的图从 2 张变成 6 张，全部从档案里已有的结果画：数据结构（F1）、验证协议（F2，E1 山脊图 + R900 置换零假设）、提交历程（F3，"几个 9"刻度）、最终选择（F4，R936 + R922）、模型全景（F5，100 折蜂群图）、无标签数据的价值（F6，R441 + R404）。每张图先画、再由评审代理挑问题、再改一轮。
  - 四个扩展各换一张新图（`ext_e1`–`ext_e4`），只改图和图注，正文不动；E3 原来没有图，现在有一张。
  - 核心报告围绕新图重写；三路审查（数字 13 条、表述 22 条、读者 34 条），修改轮改掉了所有中高优先级问题，低优先级的大多改了，没改的记在报告头部注释里。
- Result: `reports/report.pdf` 15 页；核心 5.91/6，E1 1.90、E2 1.87、E3 1.95、E4 1.93（上限 2），贡献表 0.95/1；pdffonts 只有 Times New Roman 三个字形。没有新计算，没有新的 Kaggle 提交。
- Difficulties: 核心报告只剩约 4 行余量，成员的 TODO 填进去后可能挤掉一节的开头，每次改完都要 `--parts core` 重建查页数。第一次把代码字体换成 Times 时用了 `\setmonofont`，把正文字距弄坏了（18 行出界），改用 `	tdefault` 解决。
- Next: 成员通读新版核心报告和图，确认每个数字自己能解释（规则 6）；贡献表和 ai_usage 第 24 行的 TODO 由成员填。

## 2026-11-24 — TODO（报告按正式论文重写，使用 Claude Code）
- Did: 按用户要求（"你当正式文章来写和改一遍吧"）把核心报告和四个扩展按正式学术论文重写（ai_usage 第 25 行）。
  - 先定规范：`SPEC.md`（不改事实、保留所有限制说明、页数目标、正式语体）和 `HOUSE_STYLE.md`（全文统一的参考文献格式、术语表、图注和数字格式），都在会话临时目录。档案编号（R936、D010 等）不再当名词用，只在句末用方括号标出处；每部分开头有一句说明这个约定。
  - 第一轮（9 个代理）：核心报告三个写作代理各写一版（照评分项、以论点为主线、以方法为主），两个评审（阅卷人角度；编辑兼核对事实）打分，都选"以论点为主线"那版；四个扩展各出一版正式稿。
  - 第二轮（9 个代理）：以选中的那版为底稿并入另两版的长处；核心报告三路审查（数字和出处、内容和评分项、语言）共 69 条意见，扩展两路审查共 69 条；修改代理逐条核实后才改（核心 58 条采纳、8 条因篇幅或不成立没采纳；扩展 55 条采纳、6 条没采纳）。
  - 主会话：Figure 2 里 r155 的标签原来写成 "tight prior"，按该 run 的 config 改成 "converged EM"（同样的先验，EM 收敛到 tol 1e-8），只重画了 Figure 2；补全了 Figure 2、5 的图注；另有一个代理做了最后的通读。
- Result: 核心 5.88/6 页，E1 1.93、E2 1.94、E3 1.94、E4 1.94（上限 2），贡献表 1 页。数字只在档案证明原稿有误时才改：E2 复刻偏差 0.031 → 0.030、幂律增量 0.040 → 0.041；E3 |bias| 上界 0.004 → 0.0044 和 k_feat = 7 的说法；E4 几处表述（都记在各文件头部注释的 History 里）。核心报告正文的数字没有改动。
- Difficulties: 核心报告 1–5 页都是满的，任何一处多一行都会把后面整节推到下一页（试过一次补全图注就到了 6.03 页），所以每处改动都单独排版量过。成员把 TODO 换成真实内容时要注意：第 6 页只剩约 6 行，扩展第 2 页各剩约 3 行。
- Next: 成员通读新版，确认每句话自己能解释（规则 6），扩展负责人改成自己的话；ai_usage 第 25 行和贡献表的 TODO 由成员填。

## 2026-11-25 — TODO（去掉 AI 声明；参考文献改成文末完整列表，使用 Claude Code）
- Did: 按用户要求（ai_usage 第 26 行）：
  - 报告里的 AI 声明先全部去掉：核心报告的 "Use of generative AI" 一节和局限 (3)、四个扩展开头的 AI 说明。用户之后会和贡献表一起写（课程规则 6 要求最终报告里有 AI 披露）。当前 PDF 也先不放贡献表那一页。
  - 参考文献改成完整的学术格式（用户选择"文末统一一个列表"）：新脚本 `scripts/make_references.py` 从 `registry/references.md` 生成 `reports/references.md`（所有 status used 的条目加上各部分引用到的条目，共 60 条；作者、年份、标题、期刊或书名、卷期、页码、出版社，有 DOI 的给 DOI），并检查每个正文引用都有条目、每个条目都被引用。`build_report.py` 新增 "refs" 部分（放在扩展之后，没有页数限制，悬挂缩进）。
  - 核心报告原来参考文献里的软件版本和只在探索性 run 里用过的方法，改成第 5 节末尾一段带引用的正文（Software and other sources），这样列表里的每一条都在正文里被引用。
- Result: 核心 5.41/6 页，E1–E4 1.80–1.86/2 页，参考文献 1.96 页（第 15–16 页）。课程说明里核心报告的 6 页只列了五个部分（问题界定、验证协议、提交历史、最终方案、分析），没有提参考文献。
- Next: 用户写 AI 披露和贡献表之后，用 `build_report.py`（不加 --parts）重建完整文档；以后改了引用或 registry，先跑 `make_references.py`。

## 2026-11-25 — TODO（参考文献改成编号制，使用 Claude Code）
- Did: 用户认为作者–年份制"不是正规的 reference，没有数字"，改成 IEEE 编号制（ai_usage 第 27 行）。正文改用 registry 键引用（`[@hanley1982]`，pandoc 写法），`build_report.py` 按全文首次出现的顺序编号，正文印成 [1]、[3, 7]、[2–4]，文末参考文献表按编号列出（IEEE 格式，期刊用标准缩写，编号单独成列、悬挂缩进）。上一个提交里生成的 `reports/references.md` 不再需要，删掉了。
- Result: 60 条文献全部解析；核心 5.23/6 页，E1–E4 1.80–1.86 页，参考文献 1.94 页（第 15–16 页）。
- Next: 以后加引用只要在 registry 里有条目，正文写 `[@键]`，编号自动生成；`scripts/make_references.py` 可以不排版先检查引用。

## 2026-11-25 — TODO（推送与归档）
- Did: 按用户要求把 e3bf9c0（去掉 AI 声明、文末参考文献）和 f9634f8（编号制参考文献）推到 GitHub 公开仓库（推送前检查过，没有邮箱、密钥或比赛邀请链接）；然后按 §12 归档：`verify_data.py` 通过，`archive_check.py --snapshot` 0 个问题，快照 `backups/snapshot_20261125-020212.zip`（2761 个文件，191.8 MB，不进 git）。
- Result: 剩下 4 条警告都是旧的：3 个 failed run（按 append-only 保留，原因在各自 notes.md）和 `research/reports/sup.md` 引用了不存在的 `tables/r20x_candidates.csv`。
- Next: 成员写 AI 使用说明和贡献表后，用 `build_report.py`（不加 --parts）重建完整文档。
