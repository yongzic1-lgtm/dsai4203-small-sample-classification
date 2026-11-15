# Decisions（ADR 决定日志）

规则（`docs/ARCHIVE_RULES.md` §7）：编号 D### 递增；只追加，不删不改正文；只能更新 **Status**（proposed / accepted / superseded by D0xx）。影响验证协议、流程、最终模型的决定都要记。反思 Q2 和报告的 Validation protocol / Final solution 从这里取材。

## 模板

```markdown
## D0xx: <一句话标题>
- Date: YYYY-MM-DD
- Status: proposed | accepted | superseded by D0yy
- Decided by: <member ids>
- Context: 为什么需要做这个决定（问题、约束、证据）
- Decision: 决定了什么（具体到参数 / 文件）
- Alternatives considered: 备选方案及放弃的原因
- Consequences: 带来的影响、代价、以后要注意什么
- Evidence: run_id / sub_id / 文件
```

---

## D001: 用隔离的 venv（.venv）而不是 conda env
- Date: 2026-10-22
- Status: accepted
- Decided by: TODO（setup 由 Claude Code 完成，队员确认）
- Context: 在 conda env 里，用户级 site-packages `%APPDATA%\Python\Python311\site-packages` 中的旧包漏进了 `sys.path`，导入的包版本与环境声明的不一致，结果不可复现。
- Decision: 用 Anaconda 的 Python 3.11.5 建 venv `D:\polyu_class\ml\.venv`（`include-system-site-packages = false`；venv 会自动关闭用户级 site-packages）。精确版本冻结在 `requirements-lock.txt`；`ssc` 以 editable 方式安装；Jupyter kernel "Python (ssc)"。
- Alternatives considered: 在 conda env 里设 `PYTHONNOUSERSITE=1`（每个终端和 kernel 都要记得设置，容易漏）；删除用户级 site-packages（会影响这台机器上的其他项目）。
- Consequences: 永远用 `.venv\Scripts\python.exe`；新增包需要走 decision + 更新锁文件。重建方法见 `docs/SETUP.md`。
- Evidence: `.venv\pyvenv.cfg`；`.venv\Scripts\python.exe -c "import site; print(site.ENABLE_USER_SITE)"` 输出 `False`。

## D002: 官方验证协议 = stratified 5-fold × 20 repeats，seed 2026
- Date: 2026-10-22
- Status: accepted（暂定；E1 完成后确认或修订，目标 2026-11-01）→ 2026-11-01 由 D015 提议冻结、不作修改（团队确认 TODO）
- Decided by: TODO
- Context: 只有 250 行，单次 split 的 AUC 噪声很大；模型间比较必须配对（相同 split）；要能报告 mean ± std。
- Decision: `project.yaml: official_cv = {kind: kfold, n_splits: 5, n_repeats: 20, seed: 2026}`；repeat r 用 `StratifiedKFold(5, shuffle=True, random_state=2026 + r)`；所有预处理和特征选择都在 pipeline 内；模型 seed 固定。每个模型报告 mean AUC ± std across repeats；比较用配对差 + Nadeau–Bengio corrected t-test。所有正式 run 用同一个 split signature。
- Alternatives considered: 单次 hold-out（方差太大）；10-fold（每折只有 25 行验证，单折 AUC 极不稳定）；leave-one-out（无法按折计算 AUC）；5 × 10（更便宜，但配对比较的 J 只有 50）。
- Consequences: 每个模型 100 次拟合；线性模型只需几秒，GBM 需要几分钟。改协议必须写新 decision，并重跑基线。
- Evidence: 待 E1（`reports/tables/e1_scheme_spread.md`、`e1_paired_difference.md`）；`docs/VALIDATION_PROTOCOL.md`。

## D003: 合成数据沙盒与真档案分离
- Date: 2026-10-22
- Status: accepted
- Decided by: TODO
- Context: 测试和调试会产生大量 run。如果它们混进真档案，报告会引用无效结果，registry 也会被污染。
- Decision: `--data raw`（默认）写真档案（项目根下的 `experiments/`、`registry/`、`submissions/`、`reports/figures|tables/`）；`--data synthetic` 写 `sandbox/`（不进 git）。`SSC_ARCHIVE_ROOT` 可整体重定向档案（pytest、AI 代理开发用）。`data/synthetic/` 用 seed 2026 生成一次，之后不再重生成。合成数据的结果永远不作为关于真实问题的证据。
- Alternatives considered: 只用一个档案，加 `data_source` 列过滤（容易误用）；不用合成数据（无法端到端测试，也无法模拟 public LB 噪声）。
- Consequences: 所有脚本都要接受 `--data {raw,synthetic}`；正式 run 前确认 `SSC_ARCHIVE_ROOT` 没有设置。
- Evidence: `src/ssc/paths.py`（`archive_root`）。

## D004: 队员设置 git 身份之前不做 commit
- Date: 2026-10-22
- Status: superseded by D016（2026-11-15 第一次 commit）
- Decided by: 用户（setup 时要求不 commit）
- Context: `git init` 已完成，但这台机器没有配置 `user.name` / `user.email`，而且用户要求 setup 阶段不 commit。
- Decision: setup 不产生任何 commit。第一个 commit 由队员在设置 git 身份（`docs/ARCHIVE_RULES.md` §9）后亲自完成。
- Consequences: 第一次 commit 之前做的 run，`meta.json` 里记为 `git.commit = no-commit`，无法精确复现。所以**第一次 commit 之前不做正式研究 run**。
- Evidence: `git status` 显示 "No commits yet"。

## D005: 正式计算默认在 Google Colab 上跑；commit 之前用代码包保证可复现
- Date: 2026-10-22
- Status: accepted (amended by D012)
- Decided by: 用户（"最好用 colab CLI，不要用本地跑。额度用不完"；pro1、pro2 两个账号都可用）
- Context: 本机只做开发冒烟测试；官方 Colab CLI 只支持 Linux/macOS，所以装在 WSL Ubuntu-24.04 里。用户要求暂不 commit（D004），而 D004 原来的推论是"第一次 commit 之前不做正式 run"，这会挡住全部研究。
- Decision: 所有正式 run 都用 `colab/remote.py run [-a pro1|pro2] [-s 会话] -- <脚本> [参数]` 提交到 Colab。remote.py 会把发到 VM 的**完整代码包**存到 `logs/code_bundles/<hash>.tar.gz`，并把 hash 写进每个 run 的 `meta.json`（`compute.code_bundle`），同时记下本地 git 状态（SSC_GIT_COMMIT/SSC_GIT_DIRTY）、账号、会话、硬件和 job id。这样即使还没有 commit，每个 run 的代码也能原样找回。VM 上用 Python 3.11 venv，包版本与本地锁定版本一致（`colab/requirements-colab.txt`）。这一条修订 D004 的 Consequences：commit 之前的 run 是有效记录。
- Alternatives considered: 本地跑（用户不希望）；Colab 网页 notebook（无法自动归档）；等 commit 之后才开始（拖慢进度，而且代码包已经能保证可复现）。
- Consequences: 本机休眠时日志流会断，但作业继续跑，可以用 `remote.py fetch` 取回。会话用完 `remote.py down`。R001/R002 的代码包没有保存（这个机制是在它们之后加的）；R003 用的代码包是 `logs/code_bundles/b8534fcd0d68fc89.tar.gz`（06:01 JST 生成，R003 的 8 个 run 都用它）。
- Evidence: `colab/README.md`、`registry/compute_jobs.csv`、`logs/colab/*.log`。

## D006: 档案时钟统一用 JST（UTC+09:00）
- Date: 2026-10-22
- Status: accepted (amended by D012)
- Decided by: 用户（人在日本，时间都按 JST 说）；AI 实施
- Context: Colab VM 用 UTC，笔记本用 JST，比赛和 Canvas 用 HKT。run_id 带时间戳，各机器时钟不一致的话排序会乱。
- Decision: `ssc.tracking.ARCHIVE_TZ` 固定为 UTC+09:00，run_id 和 `created_at` 都用它，与运行机器无关；remote.py 也把 VM 的 TZ 设成 Asia/Tokyo（`project.yaml: timezone`）。截止时间：Kaggle 关闭于 **2026-11-30 00:59:59 JST**（= 11-29 23:59 HKT，2026-10-22 在 Kaggle 页面核实）；Kaggle 每日提交额度在 00:00 UTC = **09:00 JST** 重置。
- Consequences: 06:02 JST 之前有一小段时间代码里临时用的是 HKT 时钟，所以 R003 的 8 个 run（`20261022-0506xx_r003_*`）的 id 比实际时间早 1 小时，排序在 R001/R002 之前，尽管它们是之后生成的。按只追加原则**不改名**，在 journal 里说明。
- Evidence: `src/ssc/tracking.py`、`tests/test_tracking.py::test_run_id_and_created_at_use_archive_clock`。

## D007: 允许用 train+test 的无标签特征做直推式（transductive）建模
- Date: 2026-10-22
- Status: accepted（报告里必须披露并说明理由；规则依据的出处见 D014；讲师确认：截至 2026-11-01 还没有问（档案里没有提问记录；更正此前"仍在询问"的说法，见 D015 第 4 条），TODO 队员）
- Decided by: AI 提出（I009），团队确认 TODO
- Context: R001/R002 显示信号集中在 13 个高方差特征上；在 20,000 行无标签特征上做 2 分量 GMM，就能几乎完美地把两类分开。
- Decision: 允许在**特征**上做无监督步骤（标准化、按方差选列、PCA、GMM），可以用到全部 train+test 行；标签只能用当前训练折的那部分。规则核对：(1) 第 4 条只禁止用**比赛之外**的数据，测试集特征属于比赛数据；(2) Foundational Rules 4b 禁止的是对测试数据做人工标注或人工预测，算法上的无监督建模不属于此类；(3) 不涉及任何共享。因为无监督步骤不看标签，CV 估计不会被泄漏污染。
- Alternatives considered: 只用 250 个带标签的行（R003 里监督模型最好也只有 QDA 的 0.9958，线性模型只有 0.85–0.90）。
- Consequences: 报告的 Problem formulation 和 Final solution 要写清楚这一点，并引用 GMM / EM 的文献。CV AUC 已经饱和（1.0000），之后比较候选模型要改用别的依据：复刻生成器的模拟实验、BIC、对数损失。
- Evidence: R003 的 run（`20261022-050708_r003_gmm2_hv` 等）、S003（public 0.99923）。

## D008: Kaggle 提交节奏与最终选择
- Date: 2026-10-22
- Status: accepted
- Decided by: 用户授权 AI 按计划自主提交
- Context: 每天最多 5 次（每日重置在 09:00 JST）；同分时先提交者排前；private 榜最多选 2 个提交。
- Decision: 每天最多 3 次。每次提交前先用 `make_submission.py --expected-public` 预登记预期分，上传后立刻用 `record_lb.py` 记录结果。不追 public LB：public 榜只有约 395 行，在 0.999 附近 SE 约 0.0015。最终 2 个选择由 CV、复刻模拟和 BIC 决定，两个提交在方法上应该互补。
- Evidence: S001–S003。

## D009: 更正 R003 的结论：调好正则的监督 QDA 与直推式 GMM 在 CV 上不可区分
- Date: 2026-10-22
- Status: superseded by D011
- Decided by: AI（研究第 1 轮的 sup、fm、gen、mix 四个方向），团队确认 TODO
- Context: R003 用的 QDA 是 reg 0.1，正则化过强（0.9958）。如果嵌套调参改用 inner log-loss，只用 250 个标签、而且特征块在每折训练集内部重新推导（完全不用测试集特征）的 QDA，CV 就能到 0.9999–1.0000（r201_qda_block、r210_qda_block_nll）。经验贝叶斯 NIW 先验的贝叶斯 QDA 也是 1.0000（r255_bqda_eb_hv）。
- Decision: 报告里不再写"直推式方法在 CV 上胜过监督方法"。无标签数据的价值改用三条证据来写：(1) 标签预算曲线：要达到 AUC 0.99，GMM 只需 4 个标签，自训练约 20 个，严格监督约 80 个（R206）；(2) 在复刻数据上，GMM 对边界样本的排序明显更好（0.825 vs 贝叶斯 QDA 0.702，r254）；(3) GMM 离贝叶斯上限只差约 2e-7（R103、R153）。D007 仍然有效（允许直推式建模）。S003 仍是主力方案。
- Evidence: `research/reports/sup.md`、`fm.md`、`mix.md`、`gen.md`。

## D010: 最终两个选择的规则（在 r201 / r209 的 public 分数出来之前写下）
- Date: 2026-10-22
- Status: proposed（第 3 种情况由 D013 更新；团队确认 TODO；D007 的规则依据已由 D014 存档，规则页 PDF 待队员保存；讲师确认 TODO）
- Decided by: AI（Verify 第 1 轮的汇总，`research/reports/verify.md` §5），团队确认 TODO
- Context: 在 222 个复刻上，第二个选择对期望 private AUC 的影响都在 1.5e-5 以内（r209 +1.45e-5，r255 +1.37e-5，r210 +5.0e-6，r201 +4.1e-6；`reports/tables/r936_choice.csv`），不到 private 标准差（3–6e-4）的 1/20。真正的一阶风险是规则解释：S002、S003 和所有直推式候选（包括 r209、r255）都依赖 D007（可以用测试集的无标签特征），而规则原文还没有归档，讲师也还没有确认。
- Decision:
  1. 默认（现在，以及讲师没有答复时）：S003（`20261022-050708_r003_gmm2_hv`）+ r201_qda_block（`20261022-071234_r201_qda_block`，严格归纳：特征块在每个训练折内重新推导，不用任何测试集特征）。
  2. 讲师明确允许使用测试集的无标签特征：改为 S003 + r209_selftrain_qda_lw（`20261022-080632_r209_selftrain_qda_lw`；r255_bqda_eb_hv 与它等价）。
  3. 讲师明确禁止：S002、S003 都不选，改选两个严格归纳的提交（例如 r201 加 r210，或 r255_bqda_eb_trainonly），选之前先做一次复刻比较。
  4. 任何情况下都不选基础模型（TabPFN / TabICL）的 run，也不选 S004（E3 演示用）。
  5. r201、r209 的 public 分数不改变这条规则：两者与 S003 的 public 差的标准差是 3.2–6.5e-4，比任何真实差距都大。
  6. 不尝试反推生成器的 random_state 或隐藏的测试标签；复刻数据的行永远不用来训练提交的模型。
- Evidence: `research/reports/verify.md`、`verify_cand.md`（r936）、`verify_rules.md`（规则风险）、`verify_gen.md`（R922，private 区间）。

## D011: 更正 D009 的措辞和证据：CV 已饱和，分不出监督 QDA 和 GMM，但这不等于两者一样好
- Date: 2026-10-22
- Status: accepted（团队确认 TODO）
- Decided by: AI（Verify 第 1 轮：cand 核查 R930–R936 和汇总 `research/reports/verify.md` §4 第 2 条），团队确认 TODO
- Context: D009 写的是"调好正则的监督 QDA 与直推式 GMM 在 CV 上不可区分"。字面上没错，但读起来像"两者一样好"，这不对。cand 的检验功效分析表明：在这个分数段，官方 5×20 CV 分辨不出 ≤ 3e-4 的差距。D009 的证据 (2)（复刻上边界行 AUC：GMM 0.825，贝叶斯 QDA 0.702，r254）只基于 6 个复刻，没有核查者重跑过。证据 (1)（R206 标签预算）也没有被重跑。另外，r210 选到的 reg 0.001 是网格下边界，不是最优点（r930）。
- Decision:
  1. D009 里"不可区分"那句话，在报告和 `research/reports/sup.md` 里一律改成："250 个标签下，调好正则的严格监督 QDA 的 CV 也饱和（0.9999–1.0000），所以 CV 分不出它和 GMM（配对差 95% 区间约 ±2–5e-4；100 折中 0 折更好，1–4 折更差）。这是 CV 分辨率不够，不是两者一样好：在 222 个复刻上，同样的检验 222/222 次不显著，而监督 QDA 的真实 AUC 每次都更低（log-loss 调参低 1.7e-4，AUC 调参低 2.6e-4）。"
  2. 证据 (2) 改为引用 `reports/tables/r936_private.csv`：在 222 个复刻上，各候选与 S003 的 private 配对差分别是 r209 −6.9e-5、r255 −1.16e-4、r210 −1.68e-4、r201 −2.57e-4，没有一个候选显著超过 S003。r254 的 0.825 vs 0.702 如果还要引用，只能写成"6 个复刻，未复核"。
  3. 证据 (1)（R206：GMM 4 个标签、自训练约 20 个、严格监督约 80 个）标为"未复核"。已经复核过的相关结果是 R304（`20261022-070600_r304_label_requirement`）：GMM 用 2 个标签时 50/50 次 AUC 都是 1.0；QDA 从约 75 个标签起 ≥ 0.99（`research/reports/verify_ext.md`）。
  4. r210 的 reg 0.001 不写成"最优"。r930 扩展网格后，inner log-loss 选的是 1e-4（53 折）、3e-4（36 折）、1e-3（11 折），CV AUC 不变（`20261022-100052_r930_reg_edge_summary`）。
  5. D009 的其他内容不变：D007 仍然有效；S003 仍是主力方案；报告里不写"直推式方法在 CV 上胜过监督方法"。
- Evidence: `research/reports/verify_cand.md` §1；`research/reports/verify.md` §2（C4、C5、C12、C13）、§3 C4、§4 第 2 条；`20261022-172146_r936_final_selection`（`reports/tables/r936_private.csv`、`r936_cv_power.csv`）；`20261022-100052_r930_reg_edge_summary`。

## D012: 补充 D005（代码来源）和 D006（时钟）的说明
- Date: 2026-10-22
- Status: accepted
- Decided by: AI（Verify 第 1 轮：repro 核查 R910–R913，`research/reports/verify_repro.md`；汇总 `research/reports/verify.md` §2 E6、E8，§4 第 4 条），团队确认 TODO
- Context: D005 把 S003 的代码包写成 `b8534fcd0d68fc89`，看起来像是上传时就记录下来的。repro 核查发现：S003 是 D005 之前的 run，meta.json 里没有代码包；这个包自己的 `colab/remote.py` 还没有存包的代码，所以它一定是事后保存的。E3 最终模型（`20261022-063908_e3_leakage`）用了 `--no-sync`，meta.json 里没有包 hash；notes 里写的 `b413e6aa3bae4443` 在日志里找不到记录。D006 假设各机器的时钟一致，但 51 个 Colab 作业显示笔记本和 VM 差约 100 秒。
- Decision:
  1. D005 的补充说明：S003 的代码包 `b8534fcd0d68fc89` 是事后保存的副本，和这个 run 一致，但不能证明 VM 上跑的就是它。其中 r003 脚本和 `src/ssc/{transductive,validation,data,models,features}.py` 与现在的文件逐字节相同；只有 `tracking.py`（HKT 时钟、没有 compute 块）不同，这和 S003 的 meta.json（created_at 是 +08:00、没有 compute 块）对得上。E3 的代码包没有记录（`--no-sync`）。两者的代码来源问题都由 R910 的逐位复现覆盖：在新的 Colab VM 上重跑 S003、E3 最终模型和 r201，结果逐位相同（`20261022-091211_r910_s003_gmm2_hv`、`20261022-091223_r910_e3_final_k20`、`20261022-091224_r910_r201_qda_block`；对比见 `20261022-091852_r911_repro_compare`）。
  2. D006 的补充说明：笔记本时钟比 Colab VM 快约 100 秒（51 个作业里，作业最早的 run_id 比它的 job_id 早，中位数 −99 s，IQR −101 到 −94 s）。job_id 和 `submitted_at_utc` 按笔记本时钟记，run_id 按 VM 时钟记，所以 VM 上建的 run_id 会比它自己的 job_id 还早。没有对照外部时间源，所以不确定是哪一边不准（Colab VM 一般会做 NTP 同步）。处理：队员同步 Windows 时间（TODO）。已有的 run_id 和时间戳不改（只追加原则）。
- Consequences: 报告里不能写"每个 run 的代码包都有记录"。共有 47 个 `--no-sync` run 没有代码包和 env hash。S003、E3、r201 的代码来源靠 R910 的逐位复现；r209、r210 靠 r932 的独立重写逐位复现（`20261022-092058_r932_real_repro_agreement`）。以后要进入最终选择的 run 不用 `--no-sync`（verify.md 表 E5）。`submitted_at_utc` 的偏差只在 09:00 JST 重置前后约 100 秒内有影响。
- Evidence: `research/reports/verify_repro.md`（§B Code bundles；§F 第 1 条 Clock skew）；`research/reports/verify.md` §2 E1、E5、E6、E8；`registry/compute_jobs.csv`。

## D013: 根据研究第 2 轮更新 D010 的第 3 种情况；确认 D011 的两条证据；记录修正轮的代码改动
- Date: 2026-10-22
- Status: proposed（团队确认 TODO）
- Decided by: AI（研究第 2 轮 r2_inductive、r2_noise、r2_audit；修正轮 fix_code；终检 `research/reports/fix_round1.md`），团队确认 TODO
- Context: D010 第 3 种情况（讲师禁止使用测试集特征）原来写的是"r201 加 r210，或 r255_bqda_eb_trainonly，选之前先做复刻比较"。研究第 2 轮做了这个比较。D011 把 D009 的证据 (1)（R206 标签预算）标为"未复核"，证据 (2) 只有 r254 的 6 个复刻。修正轮改了代码，D012 第 1 条"与现在的文件逐字节相同"对 `models.py` 已经不成立。
- Decision:
  1. D010 第 3 种情况改为：选 ext-grid QDA（`20261022-095943_r930_qda_block_nll_ext`，嵌套 reg 网格扩到 1e-5）+ r201（S005）。在 222 个复刻上（tn0，CSV 8 位，R156 和 R922 平坦先验两种权重），ext-grid QDA 比 r201 高 1.94e-4（MC SE 8.8e-6，P(胜) 0.93），比 S003 低 6.3e-5；配对收益 +4.8e-6，同时防住 ext-grid QDA 在训练标签有错时的脆弱性（r936 tn1）。r201 + r210 只有 +1.00e-4（`reports/tables/r404_private.csv`、`r404_pairs.csv`；`research/reports/r2_inductive.md`）。ext-grid QDA 还没上传；只有第 3 种情况触发时才上传它（先预登记：给定 S003 public 时预测 0.99910，sd 0.00042，`r404_public_abc.csv`）。D010 的第 1、2、4、5、6 条不变。
  2. D011 的证据 (1) 改为"已复核"：R441（`20261022-182022_r441_label_budget_rerun`）在 80 个新折、新的标签抽样上重现了 R206：GMM 4 个标签（2 个就够），自训练约 20 个，块来自 train+test 的 QDA 约 40 个；37 个格子最大 |z| 1.98。限定：严格监督 QDA 在 reg 固定为 0.003 时要 80 个，真正的 S005 嵌套流程要 120 个，所以写"约 80–120 个"。
  3. D011 的证据 (2) 改为"已复核"：在 222 个复刻的贝叶斯边界行（后验 0.1–0.9，每个复刻约 185 行）上，S003 的 AUC 0.804（贝叶斯 0.806），r255 0.678，r209 0.718，r201 0.664；S003 在 222/222 个复刻里对每个流程都更好（`reports/tables/r404_boundary.csv`）。r254 的 0.825 vs 0.702 方向正确。D011 第 1 条的区间写成"±1.6–5.1e-4"（r210 ±1.6e-4，r201 ±5.1e-4；R440 P49–P50）。
  4. 修正轮的代码改动（补充 D005、D012）：`research/` 下大于 1 MB 的 .npz/.npy 不再放进代码包，改为单独的"research data"通道（`logs/code_bundles/rdata_<hash>.json`，meta.json 的 `compute.research_data`）；`--no-sync` 的 run 记录 VM 上已有的代码/环境标记和 `compute.no_sync`；meta.json 新增 `threadpools`（BLAS）、`compute.cpu_model` 和 `estimator`（全部参数）。`make_submission.py` 和 `kaggle_submit.py` 都执行队规每天 3 次（`project.yaml competition.team_daily_limit`，`--force-limit` 可越过队规但不能越过 Kaggle 的 5 次）。新增 `transductive_gmm` 的 config 构造器和 `configs/final_s003_gmm2_hv.yaml`：在 Colab 上按 README 的路线重跑（`20261022-212507_final_s003_gmm2_hv`，作业 `J20261022-212641-pro1-ssc`），100/100 折 AUC 1.0，test_pred 与 S003 的 run 逐位相同，make_submission --dry-run 得到 sha256 前缀 3aae140e5837753f。
  5. D012 第 1 条的"逐字节相同"现在只对 `transductive.py`、`data.py`、`features.py` 和 r003 脚本成立（2026-10-22 21:40 JST 与代码包 `b8534fcd0d68fc89` 逐文件比较）；`models.py`（修正轮加了构造器）和 `validation.py` 已经和包里的版本不同。S003 的代码来源仍以代码包 `b8534fcd0d68fc89` 加 R910 和上面的重跑为证据。
- Evidence: `research/reports/r2_inductive.md`、`r2_noise.md`、`r2_audit.md`、`fix_round1.md`；上面列出的 run 和表。

## D014: 比赛规则已存档（转述）；D007 的规则依据有了出处；队伍规则；最终管道的开源许可证核查
- Date: 2026-11-01
- Status: accepted（团队确认 TODO；规则页 PDF 待队员保存；D007 的讲师确认：截至 2026-11-01 还没有问，正文第 2 条的"仍在询问"由 D015 第 4 条更正）
- Decided by: AI（Claude Code，2026-11-01 工作流的 rules_docs 代理；规则页在已登录 Kaggle 的浏览器里读取），团队确认 TODO
- Context: 比赛规则原文一直没有存档。D007 引用的 "Foundational Rules 4b" 在仓库里找不到出处，verify_rules 把规则解释列为最终选择的一阶风险（D010）。2026-11-01（JST）在登录状态下读了比赛的 Rules 页：课程写的比赛规则 1–8，加上 Kaggle Foundational Rules。规则在 2026-10-22 已接受。同一天还看了 Overview、Data、Team 页。
- Decision:
  1. **规则存档**：`docs/COMPETITION_RULES.md`。它是转述，用我们自己的话写，只引用规则 4 的一句；写明了 URL、读取时间，以及页面上的其他事实（Data 页许可证 CC0；Overview 只列开始和截止；Team 页本身允许 10 人）。精确记录由队员用 Ctrl+P 把规则页存成 PDF 放进 `docs/`（TODO）。两者不一致时以 PDF 为准。
  2. **D007 的依据现在有出处**：比赛规则 4 只禁止用比赛之外的数据训练模型；Foundational 18a 把测试集（public 和 private 两部分）算作 Competition Data；Foundational 4b 禁止的是来自人工标注或人工预测验证 / 测试记录的信息。我们的流程里没有任何人工标注或人工预测：测试行的分数和 S006 的伪标签都由算法产生。D010 第 6 条另外禁止反推隐藏标签和生成器种子，复刻数据也不训练要提交的模型。D007 的 Status 保持 accepted（只在它的 Status 行加了指向本条的说明），讲师确认仍在询问，D010 的三种情况不变。
  3. **队伍规则**（比赛规则 1、2、5；Foundational 5a–5d、18i）：2 或 3 人，或个人；成员在 Canvas checkpoint 截止时固定；每人只有一个 Kaggle 账号、只属于一个队；checkpoint 里写明每个成员的 Kaggle display name。Kaggle 意义上的"队内"指在平台上合并后的队伍：队友先加入比赛，再由队长在 Team 页邀请、队友接受。在这之前不向他们共享仓库、预测和分数。合队条件（5c）：合并后不超过人数上限（课程规定 3 人）；合并双方的提交总数不超过一个队到申请日可交的次数（我们只用了 6 次）；在任何合队截止日和比赛截止之前完成。Overview 页没有单独的合队截止日，实际以 Canvas checkpoint 为准。2026-11-01 时 Kaggle 队伍只有 1 人。
  4. **开源许可证核查**（Foundational 6c）：最终选项（默认 S003 + S005；D010/D013 的备选 S006 和 r930 ext-grid QDA）的模型只用 numpy 2.4.6、scipy 1.17.1、scikit-learn 1.9.1、pandas 3.0.6，四个都是 BSD-3-Clause。依赖：joblib、threadpoolctl（BSD-3-Clause），python-dateutil（BSD-3-Clause / Apache-2.0 双许可）、six（MIT）、tzdata（Apache-2.0），读 config 的 PyYAML（MIT）。numpy、scipy 的 wheel 里还带 OpenBLAS（BSD-3-Clause）、LAPACK（BSD-3-Clause-Open-MPI）和 GCC 运行库（GPL-3.0-or-later 加 GCC Runtime Library Exception 3.1）；scikit-learn、pandas 的 Windows wheel 带微软 VC++ 运行库（系统运行库，不是开源代码）。这些开源许可证都不限制商业使用，满足 6c。版本取自 `requirements-lock.txt`，许可证取自 `.venv` 里已安装包的元数据（2026-11-01）；Colab VM 装同样的版本（`colab/requirements-colab.txt`）。以后最终管道如果加了新包，要重做这项核查。
  5. TabPFN / TabICL 的权重不进任何提交或集成（D010 第 4 条不变）。除了规则 4 的疑问，TabPFN 权重的 Prior Labs 许可证（Apache-2.0 加署名条款，据存档的模型卡）算不算 6c 要求的 OSI 许可证也不确定。
  6. 最终选择照 D010 / D013。按 18c，我们自己在 Submissions 页勾选（不勾的话 Kaggle 会自动选）。同分时先提交的排前（7b）。
- Alternatives considered: 把规则页原文整份复制进仓库（那是别人写的长文本；改为转述，加上队员自存的 PDF）；等讲师答复后一起记（规则文本现在就能存档，答复以后另写决定）。
- Consequences: 报告的 Problem formulation 和 Final solution 可以用规则 4、Foundational 18a 和 4b 说明直推式建模的依据，同时如实写讲师确认的情况。队友合队之前不共享仓库。checkpoint 之前完成合队，并在 Canvas 写明 display name。最终管道有变化时，打包和最终选择之前复查许可证表。
- Evidence: `docs/COMPETITION_RULES.md`；Kaggle 比赛的 Rules、Overview、Data、Team 页（2026-11-01 JST 读取）；`requirements-lock.txt`、`colab/requirements-colab.txt`；`.venv` 已安装包的元数据；`research/reports/verify_rules.md` §1；`registry/submissions.csv`（6 次上传，每个 Kaggle 日最多 3 次）。

## D015: 冻结 D002（不作修改）；验证协议 §5 的选择规则由 D010 取代；讲师问题要同时问两种用法
- Date: 2026-11-01
- Status: proposed（团队确认 TODO）
- Decided by: AI 提出（Claude Code 主会话，2026-11-01；依据 AI 审稿意见 C11、C12、C24，见 `research/reports/review_round1.md`），团队确认 TODO
- Context: D002 的 Status 写着"E1 完成后确认或修订，目标 2026-11-01"。E1 的真实数据 run 已完成，并经过 verify 第 1 轮复核（R940、R941）。`docs/VALIDATION_PROTOCOL.md` §5 事先写了一条最终选择规则：候选来自 clean git commit，取最好 CV，划容差带，在带内选最简单的。这条规则没有执行：139 个官方 run 里有 30 个在全部 100 折上 AUC 都是 1.0；214 个 run 全是 `no-commit`（D004）。后来是 D010（+D013）取代了它，但这件事只记在 `reports/reflections/FACTS.md`，没有写成决定。另外，D007、D014 和 `docs/COMPETITION_RULES.md` 写"讲师确认仍在询问"，而清单 A 和 FACTS.md 都说还没有问。计划中的问题只问无监督步骤，没有覆盖 S006 用测试行伪标签做的自训练。
- Decision:
  1. **D002 冻结，不作修改**：stratified 5-fold × 20，seed 2026，split signature `5708ce5c3828` 不变，不重跑任何 run。依据：E1 测过的方案里，只有 5 × 20 的 MDD 95% 置信区间整个低于 0.01（0.0061 [0.0051, 0.0076]）；5 × 10 的上限是 0.0104；hold-out 和单次 k-fold 都不够（run `20261022-064054_e1_validation_protocol`、`20261022-064148_e1_validation_protocol_qda_pair`，比较见 R306，28 对线性模型的复核见 R940）。E1 的两条限制照实写进报告：按 80% power 算，5 × 20 也不够（0.0086 [0.0071, 0.0107]）；在 CV 天花板附近，这个协议分不出候选（D011）。
  2. **§5 的选择规则作废，由 D010（+D013）取代**：它要求的 clean commit 不存在；30 个 run 并列 1.0，容差带也分不出候选。报告第 2 节已经写了这次替换和原因（`reports/report_core.md` "Final-selection rule"）。`docs/VALIDATION_PROTOCOL.md` 只在状态行和 §5 开头各加一句说明，原文保留。
  3. **§5.1 的 clean-commit 条件随 §5 作废**：代码来源改由代码包加逐位复现证明（D012、D013 第 4 条、R910）。第一次 commit 之后要不要在 Colab 上重跑最终候选，由队员决定（`docs/ASSIGNMENT_CHECKLIST.md` A 节）。
  4. **讲师问题**：截至 2026-11-01，档案里没有任何提问记录，所以 D007、D014 的"仍在询问"应为"还没有问"。D007、D014 只改了 Status 行；D014 正文第 2 条按本条理解。提问要同时问两种用法：(a) 在测试集的无标签特征上做无监督拟合（S003）；(b) 用测试行上由算法产生的伪标签做自训练（S006）。另外问能否用预训练的表格基础模型（I014）。答复记为新决定：两种都允许 → D010 第 2 种情况（S003 + S006）；只允许 (a) → 默认组合（S003 + S005）；都不允许 → 第 3 种情况（D013）。
- Alternatives considered: 改成 5 × 50（MDD 更小，但成本是 2.5 倍，也解决不了天花板问题）；改成 5 × 10（区间上限超过 0.01）；保留 §5，等第一次 commit 之后重跑（commit 只能由队员做，而且 30 个并列的 run 仍然分不出来）。
- Consequences: 不需要重跑，已有结果全部有效。`docs/VALIDATION_PROTOCOL.md` 状态改为"frozen by D015（proposed）"。清单 A 节的冻结项和讲师问题相应更新。`docs/COMPETITION_RULES.md` §4 第 1 条改为"还没有问"。
- Evidence: `reports/extensions/e1_validation_protocol.md` 表 1；`reports/tables/e1_paired_difference.csv`、`e1_qda_paired_difference.csv`、`r306_e1_pair_comparison.md`；`research/reports/verify.md` F1–F2；`registry/experiments.csv`（git_commit 列 214/214 为 `no-commit`）；`research/reports/review_round1.md`（C11、C12、C24）。

## D016: 第一次 commit，推到 GitHub 私有仓库，供队员共享
- Date: 2026-11-15
- Status: accepted（第 1 条的"私有、不改成公开"由 D017 取代）
- Decided by: 用户（要求"传到 github，分享给小组成员"，并选了账号、仓库名、提交邮箱和邮箱处理方式），AI（Claude Code）执行
- Context: D004 规定队员设置 git 身份之前不 commit，到 11-15 仓库还没有任何 commit。用户要把项目分享给组员。比赛规则禁止把代码、数据、预测和结果分享到队伍以外（`docs/COMPETITION_RULES.md`，D014），所以只能用私有仓库。
- Decision:
  1. 仓库：GitHub 账号 `yongzic1-lgtm` 下的**私有**仓库 `dsai4203-small-sample-classification`，只邀请本队成员。不改成公开。
  2. 第一次 commit 的作者：仓库内（local）git 身份 `yongzic1-lgtm` + GitHub 隐藏邮箱（`<id>+yongzic1-lgtm@users.noreply.github.com`），不暴露真实邮箱。D004 由本条取代。
  3. 不进 git 的东西（`.gitignore`）：比赛数据 `data/raw/*`（组员自己从 Kaggle 下载，按 `data/raw/SHA256SUMS.txt` 校验）、`.venv/`、`sandbox/`、`backups/`、`experiments/**/*.npy`、`submissions/*.csv`，以及 **Colab 代码包 `logs/code_bundles/*.tar.gz`**：60 个包里有 59 个带着旧的 `project.yaml`，里面有两个 Colab 账号的 Gmail 地址；这些包只留在本机和 `backups/` 快照里。`archive_check.py` 在没有任何代码包的克隆里只给一条警告。
  4. `project.yaml` 的 `compute.accounts` 去掉 Gmail 地址，只留 pro1 / pro2 标签（代码只用标签）。
  5. 以后的 run 都在 commit 之后做，`meta.json` 会记录真实的 commit；之前的 run 仍是 `no-commit`（代码来源靠代码包 + 逐位重跑，D005、D012、D013）。
- Consequences: 组员 `git clone` 后按 `docs/SETUP.md` 建环境、下载数据即可复现。组员需要被加为仓库协作者（collaborator）才能看到仓库。
- Evidence: 本条之后的第一次 commit；`gh repo view yongzic1-lgtm/dsai4203-small-sample-classification`。

## D017: 仓库改为公开（用户知情后决定）
- Date: 2026-11-15
- Status: accepted
- Decided by: 用户。AI 先说明了规则风险，用户在三个选项（保持私有并邀请组员 / 截止后再公开 / 现在就公开）里选了"现在就公开"。
- Context: D016 把仓库建成私有。用户随后要求公开。存档的规则（`docs/COMPETITION_RULES.md`）：Kaggle 6b 规定比赛期间代码只能在 Kaggle 上（本比赛的论坛或 notebooks）公开分享，6a 规定违规可能被取消资格；课程规则 5 禁止把代码和结果分享给队伍以外的人，明确包括其他组的同学。比赛 2026-11-30 00:59 JST 截止，此前排行榜上还有其他 DSAI4203 小组。
- Decision: 2026-11-15 把 `yongzic1-lgtm/dsai4203-small-sample-classification` 改为公开（`gh repo edit --visibility public`）。公开前检查过：仓库里没有比赛数据、学号、token、密钥或个人邮箱；名字只以 Kaggle 显示名出现（排行榜上本来就公开）。
- Consequences: 风险由用户承担：可能被认为违反 Kaggle 6b / 课程规则 5（可能被取消资格或引起学术诚信问题）；公开后别人可能已经 clone 或 fork，改回私有也收不回。报告的 GenAI / 合规部分如果写到分享方式，要如实写。
- Evidence: `gh repo view yongzic1-lgtm/dsai4203-small-sample-classification` 显示 PUBLIC。
