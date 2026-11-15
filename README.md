# DSAI4203 Small-Sample Classification Challenge

团队项目仓库：Kaggle 比赛 [small-sample-classification-challenge](https://www.kaggle.com/competitions/small-sample-classification-challenge)。
二分类问题，250 行有标签训练数据、19,750 行测试数据、150 个数值特征（大多是噪声），指标是 ROC AUC。
我们要交付：最终模型及其 Kaggle 提交、验证协议、核心报告（≤ 6 页）、每人一个扩展（E1–E4，各 ≤ 2 页）、contribution table、≤ 5 分钟视频、可复现的代码；另外每人单独提交一份 reflection PDF。

**仓库仅限队内使用**：不公开，数据和 token 不进 git（`docs/ARCHIVE_RULES.md` §8）。

## 黄金法则

1. 只用 `.venv\Scripts\python.exe`，不随手 `pip install`（需要新包时先记 decision）。
2. 真实结果只来自脚本，档案只追加、不修改；调试一律加 `--data synthetic`（结果写入 `sandbox/`）。
3. 只比较 split signature 相同的 run（official_cv：5-fold × 20、seed 2026）；所有预处理和特征选择都放在 pipeline 内。
4. 提交前用 `make_submission.py` 预登记 `expected_public`（可先加 `--dry-run` 只检查、不写文件），上传后立即 `record_lb.py`；每个 Kaggle 日最多 3 次（当天已记录 3 次上传时 `make_submission.py` 会停，`--force-limit` 才继续），不按 public LB 挑模型。
5. 代码、数据、预测、分数不出队；用到的库、方法、AI 都记入 `registry/references.md` 和 `registry/ai_usage.md`。

## 快速开始（PowerShell，项目根目录）

下面的命令都用相对路径，在项目根目录执行即可，与项目放在哪里无关。项目根目录现在是 `D:\polyu_class\新建文件夹`（2026-10-24 从 `D:\polyu_class\ml` 改名；路径含中文，`cd` 时加引号）。改名后 `.venv\Scripts\` 下的 `pip.exe`、`pytest.exe`、`jupyter.exe`、`kaggle.exe` 等启动器还指向旧路径、不能用，所以一律写 `.venv\Scripts\python.exe -m <模块>`；Jupyter kernel 要重新注册（`docs/SETUP.md` §1"项目文件夹改名或移动之后"）。

```powershell
cd "<项目根目录>"                      # 现在是: cd "D:\polyu_class\新建文件夹"
# 可选：激活环境（不激活也行，下面都写了 .venv 里的解释器）
.\.venv\Scripts\Activate.ps1          # 报"禁止运行脚本"时先执行: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 1. 检查数据（校验和、形状、列）
.venv\Scripts\python.exe scripts\verify_data.py

# 2. 在合成数据上试跑（写入 sandbox/，不进真档案）
.venv\Scripts\python.exe scripts\run_experiment.py --config configs\baseline_l2_logreg.yaml --data synthetic

# 3. 正式 run（真数据 -> experiments/ + registry/experiments.csv；先 commit，见 ARCHIVE_RULES §9）
$env:SSC_AUTHOR = "<你的 id>"
.venv\Scripts\python.exe scripts\run_experiment.py --config configs\baseline_l2_logreg.yaml
.venv\Scripts\python.exe scripts\list_runs.py --official-only
.venv\Scripts\python.exe scripts\compare_runs.py <RUN_A> <RUN_B>

# 4. 提交：预登记 -> 上传（网页或 kaggle_submit.py）-> 立即记录分数
.venv\Scripts\python.exe scripts\make_submission.py --run <run_id> --desc "<问题>" --expected-public 0.80
.venv\Scripts\python.exe scripts\record_lb.py --sub S001 --submitted-at now --public 0.79

# 5. 扩展（先 quick 冒烟测试，再正式运行）
.venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --data synthetic --quick

# 6. 测试、Jupyter（kernel "Python (ssc)"）
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m jupyterlab
```

其他脚本：`make_synthetic.py`（重建模拟数据，一般不需要）、`lb_noise.py`（public LB 噪声表）、`submission_history.py`（报告 §3 的提交历史表）、`kaggle_submit.py --sub S### [--dry-run]`（API 上传，需要 token）、`package_deliverables.py`（默认 dry run，`--build` 生成 zip）、`build_report.py`（把核心报告、扩展、贡献表拼成 `reports\report.pdf` 并打印每部分页数，见 `reports/README.md`）。每个脚本都支持 `--help`。

## 目录

| 路径 | 内容 |
|---|---|
| `configs/` | 核心实验 config（YAML）；模板 `_template.yaml`，baseline 为 `baseline_*.yaml`；最终模型 S003 为 `final_s003_gmm2_hv.yaml`（复现步骤见文末） |
| `data/raw/` | Kaggle 原始数据（只读，不进 git；`SHA256SUMS.txt` 进 git） |
| `data/synthetic/` | 带隐藏标签的模拟数据（seed 2026，只用于测试和沙盒） |
| `docs/` | 制度、环境、计划、作业清单、验证协议 |
| `experiments/` | 真档案：每个正式 run 一个文件夹（只追加） |
| `extensions/` | E1–E4 扩展脚手架：`config.yaml`、`run.py`（`--data`、`--quick`） |
| `research/` | 探索分析脚本 `r###_<topic>.py`（每个写一个 `kind = analysis` 的 run，表图前缀 `r###_`） |
| `colab/` | 在 Google Colab 上运行脚本的工具（见 `colab/README.md`）；作业记录在 `registry/compute_jobs.csv`，日志在 `logs/colab/` |
| `notebooks/` | 探索用 notebook（不是档案，见 `notebooks/README.md`） |
| `registry/` | `experiments.csv`、`submissions.csv`（由脚本写）+ 日志 `journal / decisions / beliefs / ideas / ai_usage / references`（人写） |
| `reports/` | 报告模板与草稿；`figures/`、`tables/` 存放脚本生成的图表；`extensions/`、`reflections/` |
| `sandbox/` | 合成数据 run 和临时文件（不进 git，可随时删除） |
| `scripts/` | 命令行脚本（run_experiment、compare_runs、make_submission、record_lb……） |
| `src/ssc/` | 共享 Python 包（paths、data、validation、features、models、tracking、submission） |
| `submissions/` | 上传用的 `S###_<run_id>.csv`（不进 git） |
| `tests/` | pytest |
| `video/` | 视频计划；最终视频文件（不进 git） |
| `deliverables/` | 最终 zip（不进 git） |

## 关键日期

| 日期 | 事项 |
|---|---|
| 2026-10-16 | Team registration 截止（已过；我们是否登记了：TODO，查 Canvas） |
| 2026-10-22 | Setup 完成；研究、E1–E4 的正式 run、提交 S001–S006 完成；最终选择规则写入 decisions（D010，D013 补充） |
| 2026-11-01 | 比赛规则存档（`docs/COMPETITION_RULES.md`，D014） |
| TODO（查 Canvas，需要队员的 NetID） | Canvas checkpoint：写明每个成员的 Kaggle display name；之后队伍成员固定（2–3 人或个人） |
| 2026-11-23 – 11-25 | 按 D010 / D013 在 Kaggle 勾选 2 个最终提交 |
| 2026-11-28 | 报告、视频、zip 定稿上传；反思 PDF（截止日 TODO，查 Canvas） |
| **2026-11-29 23:59 HKT**（= 11-30 00:59 JST） | 最终截止（规则 8；Kaggle 日在 09:00 JST = 08:00 HKT 重置） |

逐周计划见 `docs/TIMELINE.md`。

## 文档

- [docs/ARCHIVE_RULES.md](docs/ARCHIVE_RULES.md)：**归档制度（必读）**
- [docs/COMPETITION_RULES.md](docs/COMPETITION_RULES.md)：比赛规则的转述（2026-11-01 读取）和它对我们意味着什么（D014）
- [docs/SETUP.md](docs/SETUP.md)：环境、数据、Kaggle token、队友入门
- [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md)：验证协议（英文，报告 §2 草稿）
- [docs/TIMELINE.md](docs/TIMELINE.md)：周计划与提交预算
- [docs/ASSIGNMENT_CHECKLIST.md](docs/ASSIGNMENT_CHECKLIST.md)：作业要求清单与提交前检查
- [reports/README.md](reports/README.md)：如何拼装最终 PDF
- [video/PLAN.md](video/PLAN.md)：视频计划与压缩
- [notebooks/README.md](notebooks/README.md)：notebook 规则

本项目的环境、归档制度和代码脚手架由 Claude Code（Anthropic Claude）协助生成，详见 `registry/ai_usage.md`。

## Reproducing the final submission (English, for markers)

Requirements: Windows, Python 3.11, the competition data in `data/raw/` (download `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle competition page; they are not included, only their checksums `data/raw/SHA256SUMS.txt` are). The original Kaggle `.zip` listed in `SHA256SUMS.txt` is optional: `verify_data.py` skips it when it is absent. Environment setup: `docs/SETUP.md` §1 ("重建环境"), i.e. create `.venv`, `pip install -r requirements-lock.txt`, `pip install --no-deps -e .`.

The final model is submission **S003** (`registry\submissions.csv`): a transductive Gaussian mixture (`ssc.transductive.TransductiveGMM`, 2 full-covariance components) fitted on the train+test *features* of the label-free high-variance block. Labels enter only through the training part of each fold. It was made by run `20261022-050708_r003_gmm2_hv` (`research\r003_transductive_vs_supervised.py`, candidate `gmm2_hv`; 8 s on a Colab CPU VM), and the uploaded file has sha256 prefix `3aae140e5837753f`. The config `configs\final_s003_gmm2_hv.yaml` builds the same estimator for `scripts\run_experiment.py`:

```powershell
.venv\Scripts\python.exe scripts\verify_data.py                       # checksums must match data/raw/SHA256SUMS.txt (shipped in the zip)
$env:SSC_ARCHIVE_ROOT = "$env:TEMP\ssc_repro"                          # write to a scratch archive, not the team's records
.venv\Scripts\python.exe scripts\run_experiment.py --config configs\final_s003_gmm2_hv.yaml
.venv\Scripts\python.exe scripts\make_submission.py --run <run_id printed above> --desc "reproduction of S003" --expected-public 0.99923 --dry-run
.venv\Scripts\python.exe scripts\make_submission.py --run <run_id printed above> --desc "reproduction of S003" --expected-public 0.99923
Remove-Item Env:SSC_ARCHIVE_ROOT
```

What to compare:
- `run_experiment.py` should print `auc_mean 1.0000`, split signature `5708ce5c3828` and `official_cv yes`, as in the archived `experiments\20261022-050708_r003_gmm2_hv\summary.json` (AUC 1.0 in all 100 folds).
- `make_submission.py` prints the sha256 prefix of the new file (`--dry-run` prints it and writes nothing). The file appears in `%TEMP%\ssc_repro\submissions\`. Registry row S003 has `3aae140e5837753f`.
- The sha256 matches only if the test predictions are bit-identical. On Colab they are: the team re-ran S003 on a fresh Colab VM and got identical fold AUCs, out-of-fold scores and test predictions (run `20261022-091211_r910_s003_gmm2_hv`, `research\reports\verify_repro.md`). On a Windows laptop the mixture agrees with Colab only to about 1e-11, because a different OpenBLAS build (kernel) changes the floating-point summation order inside the EM fit. The fold AUCs stay identical, but in the team's laptop re-run 1 of the 19,750 values in the 8-decimal file changed, so the sha256 differed. The archived file was written on Windows (CRLF line ends); write the reproduction there too.
- Bit-identical route (team setup, `colab\README.md`): `.venv\Scripts\python.exe colab\remote.py run -- scripts/run_experiment.py --config configs/final_s003_gmm2_hv.yaml`, then the two `make_submission.py` lines above on the laptop. The original route is `research\r003_transductive_vs_supervised.py --only gmm2_hv`.
- The config route was checked against the r003 estimator on synthetic data (`tests\test_transductive.py`: same fold AUCs, bit-identical scores). For the real data the evidence is the R910 re-run above.

### Second final selection: S005 (strictly inductive QDA)

The final-selection rule D010 pairs S003 with **S005** by default: a QDA trained on the labelled rows only, with no test features (`lib_sup.Tuned`: per training fold, the high-variance block is re-derived from the 200 training rows, standardised, and `reg_param` is chosen by an inner 5-fold × 2 CV). It was made by run `20261022-071234_r201_qda_block` (`research\r201_sup_quadratic.py`, candidate `qda_block`; about 110 s on a 2-core Colab VM), and the uploaded file has sha256 prefix `3be09c0723286457`. There is no `run_experiment.py` config for the nested tuning, so the route is the research script, run on Colab for a bit-identical result:

```powershell
.venv\Scripts\python.exe colab\remote.py run -s repro -- research/r201_sup_quadratic.py --only qda_block
.venv\Scripts\python.exe scripts\make_submission.py --run <run_id printed above> --desc "reproduction of S005" --expected-public 0.9988 --dry-run
.venv\Scripts\python.exe colab\remote.py down -s repro
```

What to compare: the job prints `auc=0.9999 +/- 0.0002` and the run's `summary.json` has `auc_mean` 0.999903 and split signature `5708ce5c3828`; `make_submission.py --dry-run` prints sha256 prefix `3be09c0723286457`, as registry row S005. Tested on 2026-11-01: run `20261101-123515_r201_qda_block` (job `J20261101-123642-pro1-repro`) has the same fold AUCs, out-of-fold scores, test predictions and per-fold choices (reg_param, block columns) as the original run, bit for bit (also R910: `20261022-091224_r910_r201_qda_block`). A Colab run merges into the team archive (`experiments/`, `registry/experiments.csv`); markers who only want to check the number can run the script locally with `$env:SSC_ARCHIVE_ROOT` set to a scratch folder, as for S003 (on Windows expect agreement only to floating-point precision, not bit for bit; see above).
