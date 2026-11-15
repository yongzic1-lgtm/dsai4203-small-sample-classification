# 归档制度（Archive Rules）

> 适用于全体队员和所有 AI 助手。与其他文档冲突时以本文件为准。
> 本文件的实质性修改必须在 `registry/decisions.md` 记一条决定（D###）。

报告的 30%、核心任务里"验证协议"的论证、每个人的反思 PDF，都要求我们拿出证据：试过什么、为什么这样选、哪些预测错了。**写报告时只引用档案，不靠回忆。** 档案没记的事情，在报告里就不存在。

## 速查（每天用到的命令；细则见后面各节）

```powershell
Remove-Item Env:SSC_ARCHIVE_ROOT -ErrorAction SilentlyContinue   # 正式 run 前：档案不要被重定向（§1.4）
$env:SSC_AUTHOR = "<你的 id>"
.venv\Scripts\python.exe scripts\run_experiment.py --config configs\<name>.yaml --data synthetic   # 先在沙盒试
.venv\Scripts\python.exe scripts\run_experiment.py --config configs\<name>.yaml                   # 正式（先 commit，§9）
.venv\Scripts\python.exe scripts\list_runs.py --official-only
.venv\Scripts\python.exe scripts\compare_runs.py <run_id> <baseline_run_id>                        # 判读规则 §6
.venv\Scripts\python.exe scripts\make_submission.py --run <run_id> --desc "<问题>" --expected-public 0.80
.venv\Scripts\python.exe scripts\record_lb.py --sub S### --submitted-at now --public 0.812         # 上传后立即（§5）
.venv\Scripts\python.exe scripts\submission_history.py                                            # 报告 §3 的表
```
然后：`notes.md` 写 Conclusion，更新 `registry/*.md`（§7），commit 归档。

---

## 1. 原则

1. **Append-only（只追加）**：`experiments/`、`registry/*.csv`、`submissions/` 只增不删不改。记错了就追加更正，不覆盖。失败的 run、被否定的想法、负面结果一律保留——作业明确说负面结果有分（"did not help, here is the evidence"）。
2. **Immutability（不可变）**：run 文件夹生成后，除 `notes.md` 的 "Conclusion" 一节外不再改动。已用于正式 run 的 config 不再编辑，要变体就新建文件。
3. **可复现三元组 = git commit + config + data checksum**。三者由脚本自动写进 `experiments/<run_id>/meta.json`（`git.commit`、`git.dirty`、`data_fingerprint`）和 `config.yaml`。所以**正式 run 之前先 commit**（§9）。
4. **真档案 vs 沙盒**：

   | `--data` | 数据 | 写到哪里 | 是否是证据 |
   |---|---|---|---|
   | `raw`（默认） | `data/raw/` Kaggle 真数据 | 真档案：项目根下 `experiments/`、`registry/`、`submissions/`、`reports/figures/`、`reports/tables/` | 是 |
   | `synthetic` | `data/synthetic/` 模拟数据（已知答案） | `sandbox/`（同样的子目录结构，不进 git，可随时删） | 否，只证明"代码在已知答案的数据上做对了" |

   环境变量 `SSC_ARCHIVE_ROOT` 会把档案整体重定向到别处（pytest 和 AI 代理开发时用）。脚本启动时打印的 `archive:` 行会显示 `(from $SSC_ARCHIVE_ROOT)`。**正式 run 前确认它没被设置**：
   ```powershell
   Remove-Item Env:SSC_ARCHIVE_ROOT -ErrorAction SilentlyContinue
   ```
   注意：重定向后 raw 和 synthetic **共用一个扁平档案**（同一个 registry），sub_id 和"今天已上传几次"会把两种来源的行混在一起数；脚本会按 `data_source` 过滤 run 并拒绝跨来源提交。需要分开时给每种来源各设一个 `SSC_ARCHIVE_ROOT`。（保留扁平结构是因为 `colab/remote.py` 按 `<root>/experiments` 读取 VM 上的档案。）
   `$env:SSC_DATA = "synthetic"` 会改变所有 `scripts/` 和 `extensions/*/run.py` 在不写 `--data` 时的默认来源。
5. **先登记，后看结果（pre-registration）**：假设在 run 之前写（config 的 `hypothesis` 会被复制进 `notes.md`）；提交的 `expected_public` 在上传之前写（`make_submission.py` 强制必填）。事后不改预测。
6. **机器档案只由脚本写**：`experiments/`、`registry/*.csv`、`submissions/`、`reports/figures|tables/`、`logs/` 只由 `scripts/`、`extensions/*/run.py`、`research/r###_*.py` 和 `colab/remote.py`（取回 Colab 结果时）写入。人只写 Markdown：`registry/*.md`、`notes.md` 的 Conclusion、`reports/**/*.md`。
7. **不要用 Excel 保存 registry CSV**。可以用 Excel 打开看，但保存会改编码/格式并破坏 append-only。查看用 `scripts\list_runs.py`。Excel 打开时文件被锁：脚本会等 15 s，然后报错 "is locked by another program"。`run_experiment` 的 run 本身已完成时，registry 行存进该 run 的 `registry_row.json`，报错信息里给出补写命令。

---

## 2. 目录规则

| 路径 | 放什么 | 命名 | 谁写 | 可变？ | 进 git？ |
|---|---|---|---|---|---|
| `data/raw/` | Kaggle 原始数据 + `SHA256SUMS.txt` | Kaggle 原名 | 下载一次（人） | **只读**（文件已设 ReadOnly 属性） | 只跟踪 `SHA256SUMS.txt` |
| `data/synthetic/` | 模拟数据（seed 2026）+ `test_labels.csv` + `truth.csv` | 固定 | `scripts/make_synthetic.py`（已生成，**不要重跑**） | 不改 | 否 |
| `configs/` | 核心实验的 config（YAML）；模板 `_template.yaml` | `<group>_<description>.yaml` | 人 | 用过于正式 run 的不改 | 是 |
| `experiments/<run_id>/` | 一次正式 run 的全部输出 | run_id（§3） | `run_experiment.py`、`extensions/*/run.py` | 只能补写 `notes.md` 的 Conclusion | 是（`*.npy` 除外） |
| `registry/experiments.csv` | 每个 run 一行 | — | 脚本（`ssc.tracking.Run.finish`） | append-only，禁止手改 | 是 |
| `registry/submissions.csv` | 每个提交一行 | sub_id | `make_submission.py` 建行；`record_lb.py` 填 LB 字段 | 只有 `submitted_at_utc / public_lb / private_lb / selected_final / notes` 可事后填 | 是 |
| `registry/*.md` | 日志（§7） | 固定文件名 | 人 | 只追加；旧条目只可更新 Status / Outcome 字段 | 是 |
| `submissions/` | 上传用 CSV | `S###_<run_id>.csv` | `make_submission.py` | 不改 | 否（sha256 已记在 registry） |
| `reports/figures/`、`reports/tables/` | 报告用图表（派生产物） | `core_*`、`e1_*`..`e4_*` | 脚本 | 可由同一脚本重新生成（覆盖）；来源 run 记在 manifest 或图注里 | 是 |
| `reports/*.md`、`reports/extensions/`、`reports/reflections/` | 报告草稿与模板 | 固定 | 人 | 可改 | 是（个人反思可以不提交） |
| `extensions/e#_*/` | 扩展脚手架：`README.md`、`config.yaml`、`run.py` | 固定 | 该扩展负责人 | 代码可改（正式 run 前 commit） | 是 |
| `notebooks/` | 探索 | `NN_author_topic.ipynb` | 人 | 可改 | 是（提交前清空输出，见 `notebooks/README.md`） |
| `scripts/`、`src/ssc/`、`tests/` | CLI、共享包、pytest | — | 代码维护者 | 走 git；影响结果的改动记 decision | 是 |
| `colab/` | 在 Google Colab 上运行脚本的工具（`colab/README.md`）；取回的结果按同样的规则进入真档案或沙盒 | — | 代码维护者 | 走 git | 是 |
| `research/` | 探索性分析脚本（数据结构、子空间等），每个脚本写一个 `kind = analysis` 的 run | `r###_<topic>.py`（三位序号递增） | 人 | 用过于正式 run 的不改，新想法新建文件 | 是 |
| `registry/compute_jobs.csv` | 每个 Colab 作业一行（job_id、硬件、命令、退出码、耗时、产生的 run_id、git） | job_id `J<YYYYMMDD-HHMMSS>` | `colab/remote.py` | append-only | 是 |
| `logs/colab/` | Colab 作业的完整日志 `<job_id>.log`（产生了 run 的作业，日志另有一份 `colab_job.log` 在 run 文件夹里） | job_id | `colab/remote.py` | 不改 | 是（小文本；不进交付 zip） |
| `docs/` | 制度、环境、计划 | 固定 | 人 | 可改；ARCHIVE_RULES / VALIDATION_PROTOCOL 的实质修改记 decision | 是 |
| `project.yaml` | 队伍信息、`official_cv` | — | 人 | `team` 字段照填；`official_cv` 只能通过 decision 改 | 是 |
| `sandbox/` | 合成数据 run、草稿 | 同真档案 | 脚本 | 随意，可整个删掉 | 否 |
| `video/` | `PLAN.md`、讲稿；视频文件 | — | 人 | 可改 | 只文本（`*.mp4/mov/mkv` 被忽略） |
| `deliverables/` | 最终 zip | 以队名命名 | `package_deliverables.py --build` | 重新生成 | 否 |

**run 文件夹里有什么**（由 `ssc.tracking.Run` 写）：
`config.yaml`（实际使用的完整 config）、`meta.json`（作者、时间、git commit/dirty、包版本、数据指纹、命令行；2026-10-22 起还有 `threadpools`（BLAS 线程池）、`estimator`（模型全部参数），Colab run 的 `compute` 里有 `code_bundle`、`env_hash`、`research_data`、`cpu_model`，`--no-sync` 时另有 `no_sync`，见 D013）、`fold_scores.csv`（每个 repeat×fold 的 AUC）、`summary.json`（汇总分数、split_signature、status）、`oof.npy`、`test_pred.npy`（如有）、`notes.md`（假设 + 结论），以及该 run 额外保存的表和图。

---

## 3. 命名规范

| 对象 | 规则 | 例子 |
|---|---|---|
| run_id | `YYYYMMDD-HHMMSS_<slug>`，**一律用档案时钟 JST**（固定 UTC+09:00，`ssc.tracking.ARCHIVE_TZ`；队员的笔记本在日本），与运行机器或 Colab VM 的时区设置无关，所以 run_id 在任何机器上都按时间正确排序；`meta.json` 的 `created_at` 同样是 `+09:00`。截止和 Kaggle 日仍按 HKT / UTC 写（JST = HKT + 1 h：截止 2026-11-29 23:59 HKT = 11-30 00:59 JST；Kaggle 日在 08:00 HKT = 09:00 JST 重置）。slug 来自 config 的 `name`（小写，非字母数字变 `-`，≤40 字符）；同秒重名自动加 `-2` | `20261024-101500_baseline_l1_logreg` |
| sub_id | `S001`、`S002`…，全队连续，由 `make_submission.py` 分配 | `S007` |
| 提交文件 | `submissions/S###_<run_id>.csv` | `submissions/S007_20261024-101500_baseline_l1_logreg.csv` |
| Kaggle 上的 description | `S### <run_id> <desc>`，方便对账（`kaggle_submit.py` 自动这样填） | `S007 20261024-101500_baseline_l1_logreg <desc>` |
| 图 / 表 | `reports/figures/<prefix>_<topic>.png`、`reports/tables/<prefix>_<topic>.csv\|md`；prefix ∈ `core`、`e1`、`e2`、`e3`、`e4`、`r###`（`research/r###_*.py` 的探索分析）；`scripts\lb_noise.py` 写的 `lb_noise_*` 与 `scripts\submission_history.py` 写的 `core_submission_history.*` 算作 core；来源 run 记在 `<prefix>_manifest.json`（E1–E4 自动写）或报告图注里 | `e1_estimates_by_scheme.png`、`core_submission_history.md` |
| notebook | `NN_author_topic.ipynb`（NN 两位序号全队递增；author = 成员 id） | `01_ab_eda.ipynb` |
| config | `configs/<group>_<description>.yaml`，group ∈ `baseline`、`explore`、`final`；从 `configs/_template.yaml` 复制；`name:` 与文件名相同（不写时默认取文件名）；必须写 `hypothesis`；**不写 `cv` 块**（自动使用 `project.yaml` 的 `official_cv`） | `configs/baseline_l2_logreg.yaml`、`configs/explore_corr10_l1_logreg.yaml` |
| 日志编号 | 决定 `D###`、预测 `B###`、想法 `I###` | `D005`、`B012`、`I020` |
| 作者 id | `project.yaml` 中 `team.members[].id`（如姓名缩写），所有 author 字段统一用它 | `ab` |

作者设置：脚本加 `--author <id>`；或者在自己的 PowerShell 会话里设 `$env:SSC_AUTHOR = "<id>"`；都没有时用 `project.yaml` 的 `default_author`。多数脚本接受 run_id 的唯一前缀。

---

## 4. 实验流程

```
想法/假设 -> config -> 沙盒试跑 -> commit -> 正式 run -> 读结果 -> notes.md 写结论 -> 更新日志 -> commit 归档
```

1. **写假设**：`registry/ideas.md` 加一行（I###，status = pending）。如果有可检验的预测（例如"比基线高 ≥ 0.01 CV AUC"），在 `registry/beliefs.md` 加一行（B###，写置信度）。
2. **写 config**：复制 `configs/_template.yaml` 或最接近的现有 config 为新文件，填 `name`、`hypothesis`（预测什么、多大、凭什么、什么结果会推翻它）。所有预处理和特征选择必须写在 `pipeline:` 里（每折重新拟合），不允许在 CV 外先选特征。不写 `cv` 块。
3. **沙盒试跑**（写入 `sandbox/`，不进真档案）：
   ```powershell
   .venv\Scripts\python.exe scripts\run_experiment.py --config configs\<name>.yaml --data synthetic
   ```
4. **commit** 代码和 config（§9）。
5. **正式 run**：
   ```powershell
   .venv\Scripts\python.exe scripts\run_experiment.py --config configs\<name>.yaml --author <id>
   ```
   `--override key=value`（点号路径，例如 `--override pipeline.model.params.C=0.05`）适合快速探索；override 之后的完整 config 会被复制进 run 文件夹。要长期保留的变体请新建 config 文件，便于别人在 `configs/` 里找到。**正式 run 不要 override `cv`**：split signature 会变（归档为 `official_cv: false`），结果与其他 run 不可比。`--override cv.n_repeats=2` 只用于沙盒冒烟测试。
6. **读结果**：
   ```powershell
   .venv\Scripts\python.exe scripts\list_runs.py --official-only
   .venv\Scripts\python.exe scripts\compare_runs.py <run_id> <baseline_run_id>
   ```
   以及 `experiments\<run_id>\summary.json`。
7. **写结论**：在 `experiments\<run_id>\notes.md` 的 "Conclusion" 下写：数字（mean ± std across repeats，与基线的配对差、corrected p、win rate）、与假设是否一致、决定（采纳 / 放弃 / 下一步）。这是 run 文件夹里唯一允许手改的地方。
8. **更新日志**：`ideas.md` 的 status 和 evidence（run_id）；`beliefs.md` 的 outcome 和 lesson；`journal.md` 当天条目（包括困难）。
9. **commit 归档**（`experiments/<run_id>/`、`registry/`）。

**失败与作废**：
- 崩溃的 run 会自动记为 `status = failed: <ExceptionName>`。不删，在 `notes.md` 写原因，修好后重跑（得到新 run_id）。
- 事后发现某 run 因 bug 无效：不删除；在它的 `notes.md` Conclusion 写 `INVALID: <原因>; superseded by <新 run_id>`，在 `journal.md` 记录；如果它支撑过某个决定，在 `decisions.md` 新增一条修正决定。

**扩展（E1–E4）**同样走这个流程，入口是 `extensions\<e#_name>\run.py`：
```powershell
.venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --data synthetic --quick   # 冒烟测试
.venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --author <id>              # 正式
```
正式运行写一个 `kind = extension` 的 run 文件夹，并把报告用表图写到 `reports/tables/e#_*`、`reports/figures/e#_*`（来源 run 记在 `e#_manifest.json`）。E1–E4 的 `--quick` 输出都只放在 run 文件夹的 `report_preview/` 下，不会覆盖报告图表；但 quick run 本身仍会进入所选档案的 registry，所以 **`--quick` 只和 `--data synthetic` 一起用**。`--data` 不写时取 `$SSC_DATA`，再不写就是 raw。

**探索分析（`research/r###_*.py`）与 Colab**：探索脚本同样通过 `ssc.tracking.Run` 写 `kind = analysis` 的 run，表图用 `r###_` 前缀。在 Colab 上跑的作业（`colab/remote.py`，见 `colab/README.md`）结果被取回并合并进本地档案，另在 `registry/compute_jobs.csv` 记一行、在 `logs/colab/` 存日志。远端没有 `.git`，所以 `remote.py` 把本地的 commit 和 dirty 状态通过 `SSC_GIT_COMMIT` / `SSC_GIT_DIRTY` 传给 VM，`ssc.tracking.git_info()` 优先使用它们（`meta.json` 的 `git.from_env = true`）。commit 规则（§9）对 Colab 作业同样适用。`colab/README.md` 里"正式计算一律放到 Colab"的团队约定还没有写成 decision：`TODO`（队员）确认后记为 D005，并在这里引用。

---

## 5. 提交流程（Submission）

**限制**：每天最多 5 次；Kaggle 日在 00:00 UTC = **08:00 HKT** 重置（HKT 00:00–07:59 的提交计入前一个 Kaggle 日）。**团队计划每天 ≤ 2–3 次**（理由见 `docs/TIMELINE.md`）。

**防撞号**：sub_id 按本地 registry 递增，两台机器同时建提交会得到同一个号。同一时间只由一人做提交（先在群里说"我在交 S0xx"），开始前 `git pull`，结束后立即 commit + push `registry/submissions.csv`。

1. **选 run**：必须是正式 run（`data_source = raw`，`cv = 5foldx20`，`cv_seed = 2026`，status = done），文件夹里有 `test_pred.npy`。
2. **预登记**：先在 `beliefs.md` 写一行（"S0xx 的 public 预计 0.xx，95% 区间约 ±0.04 ≈ 2×SE"），然后：
   ```powershell
   .venv\Scripts\python.exe scripts\make_submission.py --run <run_id> --desc "<这次提交要回答的问题>" --expected-public 0.80
   ```
   生成 `submissions\S###_<run_id>.csv` 和 registry 行。`expected_public` 默认取该 run 的 CV `auc_mean`；如果你相信 public 会系统性偏离，写你真正相信的数并在 beliefs 里写理由。写入后不可改。
   第一次做提交前先用合成数据演练整套流程：`make_submission.py --data synthetic ...` 会用隐藏标签直接给出 public/private 分数，并写在 `sandbox/` 里。
3. **上传**（二选一）：
   - 网页（不需要 token）：比赛页 → Submit Prediction → 上传该 CSV，description 填 `S### <run_id> <desc>`。
   - API（需要 token，见 `docs/SETUP.md`）。脚本会检查文件 sha256、当天已用次数，然后要求输入 `yes` 确认（`--yes` 跳过确认，`--dry-run` 只显示不上传）；上传成功后自动记录 `submitted_at_utc`：
     ```powershell
     .venv\Scripts\python.exe scripts\kaggle_submit.py --sub S### --dry-run
     .venv\Scripts\python.exe scripts\kaggle_submit.py --sub S###
     ```
4. **立即记录**（拿到分数当场做，不要攒）：
   ```powershell
   # 网页上传：同时记录时间和分数
   .venv\Scripts\python.exe scripts\record_lb.py --sub S### --submitted-at now --public 0.812
   # API 上传（时间已自动记录）：只记分数
   .venv\Scripts\python.exe scripts\record_lb.py --sub S### --public 0.812
   ```
   `--submitted-at` 接受 `now` 或 ISO 时间，统一存成 UTC。脚本会打印 surprise：public 与 expected_public、与 CV 均值之差，并换算成 public LB 的 SE 倍数。
5. **写日志**：`beliefs.md` 填 outcome（实际 vs 预测，是否落在 ±2SE 内）和 lesson；`journal.md` 记一行。
6. **一次提交只回答一个问题**。public LB 只有约 395 行，SE ≈ 0.02，分不出 0.01 的差异。适合用提交回答的问题："CV 0.80 的模型在 public 上是否落在 0.76–0.84？""提交管道有没有 bug？""CV 与 LB 的关系大致如何？"。不适合的问题："A 和 B 哪个好 0.01？"——这个用 `compare_runs.py`。

**最终选择**：
- 选择规则在最后一周（11-23）之前写进 `decisions.md`（草案见 `docs/VALIDATION_PROTOCOL.md` §5），最后照规则执行，不临时根据 public LB 改主意。
- 在 Kaggle 的 My Submissions 页面勾选最终提交（可选数量以比赛页为准）。若不勾选，Kaggle 通常会自动按 public 分数选——这正是我们要避免的。
- 记录：`.venv\Scripts\python.exe scripts\record_lb.py --sub S### --final yes`，并在 `decisions.md` 记"最终选择 = S###，依据 D0xx"。

**截止后**（2026-11-29 23:59 HKT 之后 private LB 公布）：
- 对**所有**提交（不只是最终选择的）记录 private 分数。Kaggle 的 My Submissions 页每一行都会显示 private score；批量记录（PowerShell，每个提交调用一次 `record_lb.py`）：
  ```powershell
  $p = @{ S001 = 0.7954; S002 = 0.8012 }   # 从 My Submissions 抄
  $p.GetEnumerator() | ForEach-Object { .venv\Scripts\python.exe scripts\record_lb.py --sub $_.Key --private $_.Value }
  ```
- **核心报告用不到 private 分数**：private LB 在截止时才公布，而报告和 zip 也在截止前交（`docs/TIMELINE.md`：11-28 定稿）。所以报告的 Analysis 和 Submission history 只用 **CV / expected / public** 三列；private 分数记进档案，只有当个人反思的 Canvas 截止时间晚于 private LB 公布时，才在反思 Q4 里用（`docs/ASSIGNMENT_CHECKLIST.md` F 节核对日期）。

---

## 6. 比较规则

- **只有 split signature 相同的 run 才能直接比较**（`registry/experiments.csv` 的 `split_sig` 列）。在官方协议 `official_cv`（stratified 5-fold × 20 repeats，seed 2026，见 `project.yaml`）下，所有 run 的 signature 相同。
- 用 `compare_runs.py RUN_A RUN_B`，或 `compare_runs.py --ref RUN_REF RUN_1 RUN_2 ...` 做配对比较：同一组 split 上的逐折差值、配对均值差的 **split SE**（= `std_diff_repeat / √n_repeats`）、fold win rate、Nadeau–Bengio corrected t-test。split signature 或数据不同时脚本会拒绝比较。
- 不可比：不同 cv 设置、raw vs synthetic、不同数据指纹（`meta.json` 的 `data_fingerprint`）、真档案 vs 沙盒。
- 自己写比较代码时：`ssc.validation.paired_compare` 会用 `fold_scores.csv` 的 `val_id`（每折验证集的哈希）检查两边的验证集是否相同；旧 run 没有这一列时只按 (repeat, fold) 编号对齐，必须先确认两个 run 的 `split_sig` 相同。
- 报告写法：每个模型写 mean ± std（std across repeats）；比较写配对差的 mean、split SE、win rate、corrected p。两种噪声视角：|差值| < 2 × split SE 表示换一组 split seed 就可能翻转符号（"within split noise"）；corrected p 是数据抽样的视角（换一批行还成立吗），保守得多。**只有 |差值| > 2 × split SE 且 corrected p < 0.05 才算赢**；win rate 接近 0.5 也视为"分不出"（与 `compare_runs.py` 打印的 Guide 一致）。注意 `std_diff_repeat` 本身是**单个** repeat 的波动，约为 split SE 的 √20 ≈ 4.5 倍，不能直接当门槛（E1 README §6）。
- public LB 不作为模型间比较的依据（理由见 `docs/VALIDATION_PROTOCOL.md` §4）。
- 改协议只能通过 `decisions.md` 的新条目；改后旧 run 与新 run 不可比，基线必须重跑。

---

## 7. 日志文件（`registry/*.md`）

| 文件 | 记什么 | 什么时候写 | 喂给 |
|---|---|---|---|
| `journal.md` | 按日期：做了什么、结果、**困难（团队的 + 个人的）**、下一步 | 每个工作日结束前 | 反思 Q3；Submission history 的叙述 |
| `decisions.md` | ADR：D###，背景 / 决定 / 备选 / 后果 / 证据 | 任何影响协议、流程、最终模型的决定 | Validation protocol、Final solution；反思 Q2 |
| `beliefs.md` | 预注册的预测、置信度、检验方式、结果、教训 | 每次提交前至少一条；重要实验前 | 反思 Q4；Analysis |
| `ideas.md` | 想法、提出人、accepted / rejected / pending、证据 run_id、原因 | 有想法就记；状态变了就更新 | 反思 Q2；Analysis 的负面结果 |
| `ai_usage.md` | 规则 6 披露：日期、成员、工具、用途、涉及文件、谁核查 | 每次用 AI 写代码 / 文字 / 分析 | 报告 "Use of generative AI" |
| `references.md` | 规则 4：所有用到的库、方法、教程、帖子 | 用到的当天 | 报告 References |

各文件的细则：
- **journal**：按时间顺序追加 `## YYYY-MM-DD — <author>`。困难一定写：卡在哪、怎么解决、花了多久。反思 Q3 只能从这里取材。
- **decisions**：编号递增，不删不改正文，只更新 Status（proposed / accepted / superseded by D0xx）。
- **beliefs**：每人都要有自己的若干条——反思 Q4 问的是"你"一开始相信、后来被 LB 证伪的一件事。结果出来后填 outcome 和 lesson，不改原来的预测。
- **ideas**：proposer 写成员 id；AI 提出的写 `AI (Claude Code)` 并在 `ai_usage.md` 记录。rejected 也要写原因和证据。
- **ai_usage**：本项目的环境、归档制度和代码脚手架本身就是用 Claude Code（Anthropic Claude）生成的，已记为第一条。规则 6 要求每个成员都能解释每个细节，所以"谁核查过"一栏必须填。
- **references**：报告 References 只从这里拷贝。

---

## 8. 团队共享规则

- **只在队内共享**（规则 5）：代码、数据、预测、分数、报告草稿都不给队外任何人（包括其他队、公开论坛、公开群）。
- 不建公开 repo，不发布公开 Kaggle notebook，不在比赛讨论区贴代码或具体结果。
- 共享渠道：私有 Git 仓库（只邀请队员）或私有网盘。
- **数据不进 git**：`.gitignore` 已排除 `data/raw/*`（`SHA256SUMS.txt` 除外）和 `data/synthetic/`。每个队员自己从 Kaggle 下载，用 `scripts\verify_data.py` 校验。
- **token 不进 git、不发给队友、不贴给 AI**：`.gitignore` 已排除 `kaggle.json`、`access_token*`、`credentials.json`、`.env`、`*.env`、`.env.*`。每人用自己的 token。
- 大文件（`*.npy`、`submissions/*.csv`）不进 git：需要时用同一 commit + config 重跑生成（固定 seed，结果确定），或在队内私下传，再用 registry 里的 sha256 核对。
- 每次 commit 前看一眼 `git status`，确认暂存区里没有数据、token、npy。

---

## 9. git 规范

- **现状**：`git init` 已做，**setup 时没有做任何 commit**，也没有设置 git 身份（D004）。第一次 commit 前，每人在仓库里执行（只影响本仓库）：
  ```powershell
  git config user.name "<Your Name>"
  git config user.email "<your email>"
  ```
- **正式 run 之前先 commit**：`meta.json` 会记录 `git.commit` 和 `git.dirty`。`commit = no-commit` 或 `dirty = true` 的 run 无法精确复现；报告引用的结果应来自 clean commit 上的 run。
- **一个 run 的顺序**：commit 代码 + config → 跑 → commit 归档（`experiments/<run_id>/`、`registry/`）。
- **commit message**：`<type>(<scope>): <summary>`，在 summary 里引用 run_id / sub_id / D### / I###。
  - type：`code` `cfg` `exp` `sub` `ext` `report` `docs` `fix`
  - scope：`core` `e1` `e2` `e3` `e4` `ssc` `scripts` `registry`
  - 例：`cfg(core): add explore_corr10_l1_logreg (I004)`；`exp(core): 20261024-101500_explore_corr10_l1_logreg results + conclusion`；`sub: S007 public 0.79 recorded`；`docs: D005 freeze validation protocol`
- **不改写历史**：不 force push，不 amend 已推送的 commit。append-only 也包括 git 历史。
- **冲突**：`registry/experiments.csv` 两人都追加时保留双方所有行。可选：在 `.gitattributes` 写一行 `registry/experiments.csv merge=union` 让 git 自动合并。`submissions.csv` 不能用 union（`record_lb.py` 会改写行），只能靠"单人提交 + 先 pull"避免冲突。
- 分支：默认都在 `main`。扩展代码可以在个人分支开发，但正式 run 建议在已合并到 `main` 的 clean commit 上跑。

---

## 10. 交付物 / 评分项 / 反思问题 → 档案来源

| 交付物 / 评分项 / 问题 | 权重 | 档案来源 | 写到哪里 |
|---|---|---|---|
| Core task：final solution + validation protocol + LB 名次（上/中/下三分之一） | 30% | `configs/<final>.yaml`、`experiments/<final_run_id>/`、`registry/submissions.csv`（`selected_final`）、`registry/decisions.md`（D002 协议、最终选择规则）、`docs/VALIDATION_PROTOCOL.md`、`reports/*/e1_*` | `reports/report_core.md` §2、§4 |
| Report：Problem formulation | 30%（报告整体） | `scripts\verify_data.py` 输出（`ssc.data.summary`）、EDA notebook 的结论（用脚本复现后存为 `reports/figures/core_*`） | `report_core.md` §1 |
| Report：Submission history | 同上 | `registry/submissions.csv` + `registry/experiments.csv` → `scripts\submission_history.py` → `reports/tables/core_submission_history.{csv,md}`；`beliefs.md`、`journal.md` | `report_core.md` §3 |
| Report：Analysis（含负面结果） | 同上 | `experiments/*/notes.md` 的 Conclusion、`compare_runs.py` 输出、`ideas.md`（rejected）、CV / expected / public 对照（private 在截止后才公布，见 §5） | `report_core.md` §5 |
| Report：References / AI 披露 | 同上（规则 4、6） | `registry/references.md`、`registry/ai_usage.md` | `report_core.md` 末尾 |
| Extensions（个人） | 30% | `extensions/e#_*/`、`experiments/<e# run_id>/`、`reports/tables/e#_*`、`reports/figures/e#_*`（E1、E2 另有 `e#_manifest.json`） | `reports/extensions/e#_*.md` |
| Contribution table | 文档必需 | `experiments.csv` / `submissions.csv` 的 author 列、`journal.md` 作者、`git log` | `reports/contribution_table.md` |
| 复现最终提交的代码 | zip 必需 | `src/ssc/`、`scripts/`、`extensions/`、`research/`、`colab/`、`configs/<final>.yaml`、`requirements-lock.txt`、`README.md`、`docs/`（含 `SETUP.md`）、`data/raw/SHA256SUMS.txt`（`package_deliverables.py` 自动收入） | zip 内代码目录 |
| Video | 10% | `video/PLAN.md`、`reports/figures/*` | 视频文件 |
| 反思 Q1 角色与扩展 | 缺交 → 扩展 0 分 | `contribution_table.md`、`journal.md`、`git log` | 个人反思 PDF |
| 反思 Q2 其他想法及取舍 | 同上 | `ideas.md`、`decisions.md` | 同上 |
| 反思 Q3 困难 | 同上 | `journal.md` | 同上 |
| 反思 Q4 被 LB 证伪的信念 | 同上 | `beliefs.md`、`submissions.csv`（expected_public vs public_lb；private_lb 只在反思截止晚于 private LB 公布时可用，§5） | 同上 |

---

## 11. AI 助手的额外规则

- 开发和测试时把 `SSC_ARCHIVE_ROOT` 指向自己的临时目录，不写真档案。
- 没有队员明确要求，不在真数据上做研究性 run，不提交到 Kaggle。
- 不运行 `git commit`，除非队员要求。
- AI 的每次实质贡献都记进 `registry/ai_usage.md`。

---

## 12. 每周归档检查清单（每周日晚，轮值负责人）

先跑自动检查：`.venv\Scripts\python.exe scripts\archive_check.py --snapshot`。它是只读的，会核对：数据校验和；每个 run 文件夹的文件是否齐全、是否写了 Conclusion；registry 和文件夹是否一一对应；Colab 作业的日志和代码包是否都在；提交文件的 sha256；报告里引用的图表是否存在。PROBLEM 必须修好；warning 要么修好，要么能解释。`--snapshot` 会把整个归档（不含比赛数据和 `.venv`）打成 `backups/snapshot_<JST>.zip`，重大操作或重启电脑前也跑一次。

- [ ] `scripts\archive_check.py` 的结果是 0 PROBLEM；剩下的 warning 都有解释
- [ ] 本周所有正式 run 的 `notes.md` 都写了 Conclusion
- [ ] `registry/experiments.csv` 中没有未解释的 `failed` 行（原因写在 notes.md）
- [ ] 每个提交都有 `public_lb`，并在 `beliefs.md` 填了 outcome
- [ ] 本周提交次数没有超出计划预算（每天 ≤ 2–3 次）
- [ ] `journal.md` 每个工作日都有条目，且写了困难
- [ ] `ideas.md` 状态已更新（rejected 写了原因和证据）
- [ ] 协议 / 流程 / 模型选择的变化都有 `decisions.md` 条目
- [ ] `ai_usage.md`、`references.md` 已补全本周新用的工具、库、方法
- [ ] `.venv\Scripts\python.exe scripts\verify_data.py` 通过（数据未被改动）
- [ ] 报告中没有引用任何 `sandbox/` 的结果
- [ ] 所有改动已 commit 并推送到私有远端；暂存区里没有数据 / token / npy
- [ ] 本周完成的扩展已更新 `reports/extensions/e#_*.md` 草稿
- [ ] 对照 `docs/TIMELINE.md` 检查进度，必要时调整下周计划
