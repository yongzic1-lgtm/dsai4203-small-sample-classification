# 时间计划（2026-11-01 更新，到 2026-11-29 截止）

截止：**2026-11-29（日）23:59 HKT = 11-30 00:59 JST**（规则 8；D006）。Kaggle 日在 09:00 JST 重置。**Canvas checkpoint 的日期还不知道（TODO）**：查到之后，checkpoint 要做的事优先于下面的一切。

10-22 写的原计划（E1→E4 每周做一个）已经不适用：研究、四个扩展的正式 run、6 次提交和最终选择规则都在 10-22 完成了（`registry/journal.md` 10-22）。原文在 `backups/snapshot_20261022-213717.zip` 的 `docs/TIMELINE.md`。剩下的主要是只有队员能做的事：组队和 Canvas、扩展分工和改写、报告文字、视频、反思。

记号：**T** = 只有队员能做；**A→T** = AI 可以先做（队员在会话里提出），队员核对、改写，并且能解释。AI 不做 commit；只有队员授权时才上传 Kaggle。

## 已经备好的材料（2026-11-01）

| 交付物 | 已有材料 | 还缺什么 |
|---|---|---|
| 最终提交 | S003、S005、S006 已上传；选择规则 D010 + D013；S003 的复现路线已在 Colab 验证（`20261022-212507_final_s003_gmm2_hv`） | 讲师答复；在 Kaggle 勾选（11-23 至 11-25） |
| 验证协议 | `docs/VALIDATION_PROTOCOL.md`（D002，D015 提议冻结、不作修改；§5 由 D010 取代）；E1 的证据 | 团队确认 D015 |
| 核心报告（≤ 6 页） | `reports/tables/r440_core_numbers.csv`（242 个数字的核查表）、`r440_journey.md`、`reports/tables/core_*` 和 `reports/figures/core_*`（脚本生成）、`research/reports/verify.md` §2、`registry/references.md`、`registry/ai_usage.md` 的草稿段落 | AI 全稿（11-01 写完，按 77 条 AI 审稿意见改过，`research/reports/review_round1.md`；6 页，用 5.77）需要成员审读、改写，并能解释；队名、名字、Kaggle 勾选 |
| 扩展 E1–E4（各 ≤ 2 页） | 四份草稿，`build_report.py` 量出都 ≤ 2 页；正式 run 都已归档；E2、E4 的图 11-01 按印刷尺寸重画 | 负责人、作者行、AI 说明行的 TODO、改写 |
| 贡献表（1 页） | 骨架（11-01）：每节和每个扩展一行，Evidence 和 AI use 两栏已填 | 名字、ID 和分工 |
| 视频（≤ 5 分钟） | `video/PLAN.md`；讲稿 `video/SCRIPT.md`（3 / 2 / 1 人版）、幻灯片 `video/slides.pdf`（14 张）、录制指南 `video/RECORDING.md`（11-01 AI 起草） | 填名字和分工、改写、排练、录制 |
| 代码与复现 | `README.md`（S003 的 config 路线，S005 的 research 脚本路线）、`configs/final_s003_gmm2_hv.yaml`、`requirements-lock.txt` | 第一次 git commit |
| PDF | `scripts\build_report.py`（拼装各部分，打印每部分的页数；`reports/README.md`） | — |
| 比赛规则 | `docs/COMPETITION_RULES.md`（D014） | 规则页 PDF（队员保存） |
| 反思 | `reports/reflections/TEMPLATE.md`；事实和出处 `reports/reflections/FACTS.md`（11-01，只作材料） | 每人自己的 beliefs、journal 和反思 |

## 第 1 周：11-01（日）– 11-08（日）队伍、Canvas、规则、分工

| 日期 | T（队员） | A→T |
|---|---|---|
| 11-01 – 11-03 | 1. 用 NetID 登录 Canvas，查：checkpoint 截止日；反思 PDF 的截止日和篇幅；团队登记状态（10-16 截止）；视频语言；AI 披露有没有格式要求。写进 `docs/ASSIGNMENT_CHECKLIST.md` 和 README 的关键日期。<br>2. 定队伍：2 或 3 人，或个人。队友先在 Kaggle 加入比赛、接受规则，再由队长在 Team 页邀请，队友接受（合队）。合队之前不共享仓库、预测和分数。<br>3. 填 `project.yaml` 的 `team`（姓名、学号、Kaggle display name、负责的扩展）。<br>4. 规则页存成 PDF 放进 `docs/`（步骤在 `docs/COMPETITION_RULES.md` 开头）。 | — |
| 11-02 – 11-05 | 5. 私下、笼统地问讲师（不提分数和方法细节）：能否在无监督步骤里用测试集的无标签特征；能否用预训练的表格基础模型。答复记为新决定。<br>6. 设置 git 身份，做第一次 commit，建私有远端（D004）；同步 Windows 时钟（D012）。<br>7. 重新注册 Jupyter kernel（`docs/SETUP.md` §1，"项目文件夹改名或移动之后"）。<br>8. 读并确认 D007、D009–D014；填 D001–D003 的 Decided by。 | 冻结 D002 的决定草稿；填 `docs/VALIDATION_PROTOCOL.md` 的 `TODO(E1)` |
| 11-06 – 11-08 | 9. 定扩展分工：每人一个，不重复。负责人开始读自己扩展的代码和草稿。<br>10. checkpoint 截止之前：在 Canvas 写明每个成员的 Kaggle display name。 | 核心报告初稿（11-01 起草）按队员的意见修改；References、AI 披露段落、贡献表骨架。数字只取 `r440_core_numbers.csv` 和它引用的档案 |

## 第 2 周：11-09（一）– 11-15（日）扩展改写，核心报告第一版

- **T，每个扩展负责人**：读代码；可选在 Colab 上重跑（`.venv\Scripts\python.exe colab\remote.py run -- extensions/<e#_name>/run.py`）；用自己的话改写；填作者行；处理 `research/reports/fix_round1.md` §7 的遗留项；用 `scripts\build_report.py --parts e#` 查页数。
- **T，每人**：在 `registry/beliefs.md` 写至少 2 条自己的预测（例如 S003 的 private 分数、private 与 public 的差），在 `registry/journal.md` 写自己的困难。这是反思 Q2–Q4 的材料。
- **T**：核心报告每节指定一个负责人，审读初稿，删掉没人能解释的内容。
- **A→T**：按反馈改核心报告；把 S005（r201）的复现命令写进 README（先在 Colab 上试一遍）。
- **Kaggle**：不需要上传。只有讲师明确禁止用测试集特征时，才按 D013 预登记并上传 ext-grid QDA（r930）。

## 第 3 周：11-16（一）– 11-22（日）能解释每个细节，第二版，视频讲稿

- **T，讲解会**（规则 6）：每人能讲清官方 CV、D007 的依据（规则 4，Foundational 18a、4b）、GMM/EM 与 QDA、生成器识别、public LB 的噪声（Hanley–McNeil）和自己的扩展。讲不清的内容从报告里删掉。
- **T**：贡献表定稿（每节、每个扩展对应一个成员）；`registry/ai_usage.md` 的 Member / Verified by 两栏和披露段落里的 `[TODO]`。
- **T**：视频讲稿和幻灯片（AI 初稿 11-01 已在 `video/`）改成自己的话，填名字和分工，计时排练；核心报告第二版（成员改写），用 `build_report.py` 查页数。
- **11-22**：讲师已经答复的话，按 D010 写决定。之后选择规则不再改。

## 第 4 周：11-23（一）– 11-29（日）最终选择、视频、打包

| 日期 | 事项 | 谁 |
|---|---|---|
| 11-23 – 11-25 | 按 D010 / D013 机械地选 2 个最终提交（默认 S003 + S005），在 Kaggle Submissions 页勾选；`record_lb.py --sub S### --final yes`；在 `decisions.md` 写"最终选择 = …，依据 D010" | T |
| 11-24 – 11-26 | 录视频：全员露脸，≤ 5:00，≤ 300 MB（`video/PLAN.md`） | T |
| 11-26 – 11-27 | 报告定稿；打包前从 `project.yaml` 去掉个人邮箱；`package_deliverables.py` 预检；每人写反思 | T（A→T 可以帮着检查） |
| 11-28（六） | `build_report.py` 生成 `reports\report.pdf`；`package_deliverables.py --build` 生成以队名命名的 zip；上传 Canvas；每人上传自己的反思 PDF | T |
| 11-29（日） | 缓冲，只做补救。23:59 HKT（11-30 00:59 JST）截止 | — |
| 截止后 | 记录所有提交的 private 分数（`record_lb.py --private`）；B004、B008 填 outcome | T 或 A→T |

每周日晚：`scripts\archive_check.py`（`docs/ARCHIVE_RULES.md` §12）。

## 提交预算

- 已用 6 次（S001–S006），每个 Kaggle 日最多 3 次（队规，D008；Kaggle 上限 5 次）。
- 计划中的上传只剩两种情况：(1) 讲师明确禁止用测试集特征，上传 ext-grid QDA（r930；D013 已给出预登记的预测）；(2) 最终提交的文件出了问题，需要重新生成。
- 其他时候不上传：public 只有约 395 行，不能用来挑模型（`docs/VALIDATION_PROTOCOL.md` §4），D010 也不按 public 分数改选择。
- 每次上传前照旧预登记（`make_submission.py --expected-public` 加 `registry/beliefs.md` 一行），上传后立刻 `record_lb.py`。
- 深夜上传前先查当天已用次数：Kaggle 日从 09:00 JST 开始，所以 JST 00:00–08:59 的上传算前一个 Kaggle 日。
