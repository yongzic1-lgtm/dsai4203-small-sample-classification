# 在 Colab 上跑实验（colab/remote.py）

团队约定：**正式计算一律放到 Google Colab 上跑**，本机只做几秒钟的开发冒烟测试。
Colab compute units 很充足（2026-10-22 余额 1674 CU），不用省。

## 结构

```
Windows (本项目, .venv)  --wsl.exe-->  WSL Ubuntu-24.04: ~/.local/bin/colab (官方 Colab CLI 0.7.5)
                                                     |
                                                     v
                                    Colab VM: /content/ml (代码+数据), /content/venv (Python 3.11 + 锁定版本)
```

- 官方 Colab CLI（PyPI `google-colab-cli`）只支持 Linux/macOS，所以装在 WSL 里：
  `~/.venvs/colab-cli`（micromamba 建的 Python 3.12 环境），`~/.local/bin/colab` 是软链接。
- 登录已完成（OAuth，token 在 WSL 的 `~/.config/colab-cli/token.json`）。过期时在终端运行
  `wsl -d Ubuntu-24.04 -- ~/.local/bin/colab usage`，按提示打开链接、登录，把 code 粘回终端。
- VM 上的 venv 用 `colab/requirements-colab.txt`，版本与本地 `requirements-lock.txt` 一致
  （xgboost 用同版本的 `xgboost-cpu`）。**复现程度**：Colab 与 Colab 之间逐位相同（R910 在新 VM 上重跑
  S003、E3 最终模型、r201，fold AUC、oof、test_pred 全部 bit-identical，单线程也一样）；本机（Windows）
  与 Colab 只一致到约 1e-11：S003 的 GMM 在本机 test_pred 最大差 4.4e-11，`%.8f` 提交文件里有 1 个值不同，
  fold AUC 相同。原因是两边的 OpenBLAS（构建和 kernel）不同，GaussianMixture（EM）里浮点求和顺序变了
  （`research/reports/verify_repro.md` §A，run `20261022-091211_r910_s003_gmm2_hv`）。所以要逐位复现就在
  Colab 上跑；每个 run 的 `meta.json` 现在记录 `threadpools`（BLAS/OpenMP 库、版本、kernel、线程数）、
  `compute.cpu_model` 和 joblib/threadpoolctl 版本，可以直接对比两台机器。

## 常用命令（在项目根目录，PowerShell）

```powershell
.venv\Scripts\python.exe colab\remote.py doctor                 # CLI 版本、余额、活动会话
.venv\Scripts\python.exe colab\remote.py up                     # 建会话 + 装环境 + 同步代码和数据
.venv\Scripts\python.exe colab\remote.py run -- scripts/run_experiment.py --config configs/baseline_l1_logreg.yaml
.venv\Scripts\python.exe colab\remote.py run --high-mem -- extensions/e1_validation_protocol/run.py --n-jobs -1
.venv\Scripts\python.exe colab\remote.py jobs                   # 看已记录的 Colab 作业
.venv\Scripts\python.exe colab\remote.py down                   # 停掉会话，释放 VM
```

`--` 后面就是平常在本地会敲的脚本和参数（不用写 `python`）。

**两个账号**（对应的 Google 账号见 `project.yaml: compute.accounts`）：`-a pro1`（默认账号）和
`-a pro2`（第二个账号，登录状态在 WSL 的 `~/.colab-accounts/pro2`）。`-s <名字>` 选会话；
同一时间可以用不同的会话名并行跑多个作业。新账号登录（在项目根目录的 PowerShell 里运行）：
`wsl -d Ubuntu-24.04 -- bash colab/login_account.sh <名字>`。
WSL 从当前的 Windows 文件夹启动，所以这个相对路径不管项目放在哪里都能用（项目现在在
`D:\polyu_class\新建文件夹`，WSL 里是 `/mnt/d/polyu_class/新建文件夹`；2026-11-01 实测）。`remote.py` 自己换算路径，
文件夹改名后不用改。

**可复现性（D005）**：每次同步都会把发到 VM 的完整代码包存到 `logs/code_bundles/<hash>.tar.gz`，
hash 记在每个 run 的 `meta.json → compute.code_bundle`，所以还没 commit 时也能找回当时的代码。
这些 `.tar.gz` 只存在发起作业的那台电脑和 `backups/` 快照里，不进 git（D016：旧包里的 `project.yaml` 有个人邮箱）；
从 GitHub 克隆的副本里没有它们，`archive_check.py` 只给一条警告。需要时向仓库所有者要快照。

- **research 数据文件不进代码包**：`research/` 下超过 1 MB 的 `*.npz` / `*.npy`（打包的 replica 数组，例如
  `research/r935_r933_pooled.npz` 3.5 MB）单独作为 "research data" 上传，只在它们变化时上传一次，解压到 VM 上
  同样的位置（`/content/ml/research/`），所以 `research/r936_final_selection.py`、r404、r425、r426 等读它们的脚本
  不用改。它们的 sha256 清单存在 `logs/code_bundles/rdata_<hash>.json`，hash 记在 `compute.research_data`。
  以前代码包因为这些文件涨到 13.5 MB（`logs/code_bundles/49934b9d99722ab5.tar.gz`），去掉后约 1.3 MB
  （2026-10-22 本地测量）。
- **`--no-sync`**：不上传任何东西，但会读 VM 上次同步留下的标记（`/content/ml/.sync_code`、`.sync_env`、
  `.sync_rdata`），把它们写进 `compute.code_bundle` / `env_hash` / `research_data`，并加 `compute.no_sync: true`。
  本地代码与 VM 上的不同时会打印提示（不会上传）。读不到标记时记成 `unsynced (--no-sync)`，和以前一样。

**会话卡住**：有的会话（遇到过美国东部机房）新建的 websocket 会卡死。`run` 启动作业前发现会话没有响应，
会自动停掉并重建（`--no-heal` 关闭这个行为）；作业运行中断线时会自动重连，最多等 30 分钟。

`run` 的步骤：
1. 没有会话就新建（默认名 `ssc`，CPU；`--high-mem` 要高内存/多核机型，`--gpu` 本项目用不到）；
2. 首次装环境；代码包、数据、research 数据有变化才重新上传（按哈希判断）；
3. 在 VM 上后台启动命令，`SSC_ARCHIVE_ROOT` 指向这次作业的独立目录，同时把本地 git commit、作者、机器信息传过去；
4. 每 10 秒拉一次日志，实时显示。按 Ctrl-C 只是断开，作业继续跑，之后用
   `remote.py fetch <JOB_ID> --data raw` 取回结果；
5. 打包下载，合并进**本地归档**：run 文件夹进 `experiments/`，行追加到 `registry/experiments.csv`，
   报告图表进 `reports/figures|tables/`，日志存到 `logs/colab/<JOB_ID>.log`，每个 run 文件夹里也放一份
   `colab_job.log`，作业本身记一行到 `registry/compute_jobs.csv`。

`--data synthetic` 的作业只会合并进 `sandbox/`，不会污染正式归档。

## 规则

- 提交文件（`make_submission.py`）、`record_lb.py`、上传 Kaggle 都在本地做，不在 VM 上做。
- 正式 run 之前先提交 git，这样 `meta.json` 里记的是干净的 commit（目前用户要求暂不提交，会记成 `no-commit`）。
- 会话不用时 `remote.py down`；空闲太久 Colab 也会自动回收，下次 `run` 会自动重建、重装环境。
- 本机休眠时轮询会停，VM 可能因为空闲被回收。长作业期间别让电脑睡眠，或者断开后再 `fetch`。
- 自检：`remote.py run -- colab/selftest.py --data synthetic`（30 秒左右，结果进 sandbox）。
