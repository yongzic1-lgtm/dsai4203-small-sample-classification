# 环境与数据设置（Setup）

所有命令在 PowerShell 中、在**项目根目录**下执行。命令都用相对路径，与项目放在哪里无关。项目根目录现在是 `D:\polyu_class\新建文件夹`（2026-10-24 从 `D:\polyu_class\ml` 改名）；路径含中文，`cd` 时加引号：`cd "D:\polyu_class\新建文件夹"`。改名之后要注意的事见 §1 末尾"项目文件夹改名或移动之后"。

---

## 1. Python 环境

| 项 | 值 |
|---|---|
| 解释器 | `.venv\Scripts\python.exe`（隔离的 venv） |
| 基础 Python | Anaconda `D:\Anacondar\anaconda3\python.exe`，Python 3.11.5 |
| 主要包 | scikit-learn 1.9.1、numpy 2.4.6、pandas 3.0.6、scipy 1.17.1、lightgbm 4.7、xgboost 3.2、matplotlib 3.11、seaborn 0.13、statsmodels 0.15、pytest 9、pyyaml、tabulate、kaggle 2.2.4 |
| 精确版本 | `requirements-lock.txt`（`pip freeze` 结果）；直接依赖的下限在 `requirements.txt` |
| 项目包 | `ssc`，以 editable 方式从 `src/ssc` 安装（改代码立即生效，不用重装） |
| Jupyter kernel | `Python (ssc)`（kernelspec 名 `ssc`） |

**规则**：
- 永远用 `.venv\Scripts\python.exe`，不用系统 / Anaconda 的 `python` 或 `pip`。
- 不要随手 `pip install`。确实需要新包时：在 `registry/decisions.md` 记一条 → 更新 `requirements.txt` → 安装 → 重新生成 `requirements-lock.txt` → 通知队友。

**激活（可选）**：不激活也能用，直接写 `.venv\Scripts\python.exe` 最不容易用错解释器。要激活的话：
```powershell
.\.venv\Scripts\Activate.ps1
```
如果提示"禁止运行脚本"（execution policy），只对当前窗口放行：
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**检查环境是否正确**：
```powershell
.venv\Scripts\python.exe -c "import sys, site, sklearn, ssc; print(sys.prefix, site.ENABLE_USER_SITE, sklearn.__version__, ssc.__file__)"
```
期望：`sys.prefix` 是 `<项目根目录>\.venv`（现在是 `D:\polyu_class\新建文件夹\.venv`），`ENABLE_USER_SITE` 是 `False`，sklearn `1.9.1`，`ssc` 指向 `<项目根目录>\src\ssc\__init__.py`。2026-11-01 实测四项都对。

### 为什么不用 conda env（D001）

这台机器的用户级 site-packages `%APPDATA%\Python\Python311\site-packages` 里装着很多旧包。conda env 默认会把这个目录加入 `sys.path`，于是旧版本的包"漏"进了 conda env，`import` 到的版本和 `pip list` 显示的不一致，结果不可复现。venv 会自动关闭用户级 site-packages（`site.ENABLE_USER_SITE = False`），并设置 `include-system-site-packages = false`（见 `.venv\pyvenv.cfg`），所以环境只包含 `requirements-lock.txt` 里的包。

### 重建环境

```powershell
# 1. 删除旧环境（仅在需要重建时）
Remove-Item -Recurse -Force .venv
# 2. 用 Anaconda 的 Python 3.11 建 venv
D:\Anacondar\anaconda3\python.exe -m venv .venv
# 3. 按锁定版本安装
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
# 4. 以 editable 方式安装 ssc（依赖已在第 3 步装好）
.venv\Scripts\python.exe -m pip install --no-deps -e .
# 5. 注册 Jupyter kernel "Python (ssc)"
.venv\Scripts\python.exe -m ipykernel install --user --name ssc --display-name "Python (ssc)"
# 6. 测试
.venv\Scripts\python.exe -m pytest
```

### 项目文件夹改名或移动之后

2026-10-24 项目从 `D:\polyu_class\ml` 改名为 `D:\polyu_class\新建文件夹`。venv 里有几处记着旧的绝对路径：

| 东西 | 改名后的状态 | 怎么办 |
|---|---|---|
| `.venv\Scripts\python.exe` | 照常能用（`.venv\pyvenv.cfg` 只记基础 Python 的位置） | 所有命令都通过它运行 |
| editable 安装的 `ssc`（`.venv\Lib\site-packages\__editable__.ssc-0.1.0.pth` 里一行 `src` 的绝对路径） | 2026-11-01 已改成新路径 | 以后再改名：`.venv\Scripts\python.exe -m pip install --no-deps -e .` |
| `.venv\Scripts\` 下的 `.exe` 启动器（`pip.exe`、`pytest.exe`、`jupyter*.exe`、`kaggle.exe` 等） | 写死了旧的 python 路径，不能用（2026-11-01 实测 `pytest.exe --version` 直接以退出码 1 结束）。激活 venv 后直接敲 `pip`、`pytest`、`kaggle` 用的也是它们 | 一律写 `.venv\Scripts\python.exe -m <模块>`：`-m pip`、`-m pytest`、`-m jupyterlab`、`-m kaggle` |
| `activate.bat`（cmd） | 写死了旧路径 | 用 PowerShell 的 `Activate.ps1`：它按自己的位置算路径，不受影响 |
| Jupyter kernel "Python (ssc)"（`%APPDATA%\jupyter\kernels\ssc\kernel.json`） | 2026-11-01 已在本机重新注册，指向新路径 | 以后再改名：重新注册（覆盖同名 kernel）`.venv\Scripts\python.exe -m ipykernel install --user --name ssc --display-name "Python (ssc)"` |
| WSL / Colab | `colab/remote.py` 自己把 Windows 路径换算成 `/mnt/d/...`，不用改 | 见 `colab/README.md` |

最彻底的办法是按上面"重建环境"重建 `.venv`，所有启动器都会重新生成。

### Jupyter

```powershell
.venv\Scripts\python.exe -m jupyterlab
```
在 notebook 里选 kernel **Python (ssc)**。启动时出现的 "frozen modules" debugger 提示无害。notebook 的规则见 `notebooks/README.md`。

注意：**不要用 `python -m jupyter lab`、`jupyter lab` 或 `.venv\Scripts\jupyter.exe lab`**（在没有激活 venv 时）。`jupyter <子命令>` 会在 PATH 里查找 `jupyter-<子命令>`，而 PATH 中 Anaconda 排在前面，于是实际启动的是 Anaconda 的 JupyterLab 3.6（setup 时实测）。用模块形式（`-m jupyterlab`、`-m nbconvert`）或先激活 venv，就能确保用的是 `.venv` 里的版本。

---

## 2. 数据

### 下载

1. 在 Kaggle 比赛页 <https://www.kaggle.com/competitions/small-sample-classification-challenge> 接受规则（Rules），然后在 Data 页下载 zip。
2. 放到 `data\raw\small-sample-classification-challenge.zip` 并解压到 `data\raw\`：
   ```powershell
   Expand-Archive data\raw\small-sample-classification-challenge.zip -DestinationPath data\raw
   ```
   得到 `train.csv`（250 行：id, target, "0".."149"）、`test.csv`（19,750 行）、`sample_submission.csv`（id, target）。
3. 设为只读，防止误改：
   ```powershell
   attrib +R data\raw\*.csv
   attrib +R data\raw\*.zip
   ```
   （已配置好 API token 的话，也可以用 `.venv\Scripts\python.exe -m kaggle competitions download small-sample-classification-challenge -p data\raw` 下载。）

### 校验

```powershell
.venv\Scripts\python.exe scripts\verify_data.py
```
检查 `data/raw/SHA256SUMS.txt` 中的校验和（2026-10-22 下载时记录）、形状、列名、NaN、id 唯一性、`sample_submission` 的 id 顺序。退出码 0 = 通过。**校验不通过时不要跑任何实验。**

`SHA256SUMS.txt` 是 `data/raw/` 里唯一进 git 的文件；队友用它确认手上的数据和我们的一模一样。手动核对单个文件：
```powershell
Get-FileHash data\raw\train.csv -Algorithm SHA256
```

### 模拟数据

`data/synthetic/` 已用 `ssc.synthetic.write()`（seed 2026）生成，布局与真数据相同，另有隐藏标签 `test_labels.csv`（含 public 掩码）和 `truth.csv`（真正有信息的特征）。**不要重新生成**——沙盒 run 和测试都基于这份数据。只有它丢失时才运行 `scripts\make_synthetic.py`（已存在时它会拒绝覆盖）。检查：
```powershell
.venv\Scripts\python.exe scripts\verify_data.py --data synthetic
```

---

## 3. Kaggle API token（由账号本人操作）

**只有 `scripts\kaggle_submit.py`（以及命令行下载）需要 token。网页上传提交完全不需要。** token 等同于你账号的密码：不进 git、不放在项目目录、不发给队友、不贴进聊天或 AI 工具。每个队员用自己的 token。

1. 登录 Kaggle → 右上角头像 → **Settings** → **API** 一节 → 生成新 token（Generate / Create New Token）。
2. 按 Kaggle 文档把它放到以下任一位置（已安装的 kaggle 2.2.4 按这个顺序查找）：
   - **新式 API token**：把 token 字符串单独存成文件 `%USERPROFILE%\.kaggle\access_token`（一行，无扩展名）；或者设置环境变量 `KAGGLE_API_TOKEN`。
   - **旧式 `kaggle.json`**（包含 username 和 key）：放到 `%USERPROFILE%\.kaggle\kaggle.json`；或者设置环境变量 `KAGGLE_USERNAME` 和 `KAGGLE_KEY`。
   - 也可以运行 `.venv\Scripts\python.exe -m kaggle auth login` 走浏览器登录（OAuth），不需要手动保存 token。
3. 不建议把 token 直接敲在命令行里（会留在 PowerShell 历史中），优先用文件。
4. 测试（只读，列出你在本比赛的提交记录）：
   ```powershell
   .venv\Scripts\python.exe -m kaggle competitions submissions small-sample-classification-challenge
   ```
5. token 泄露或怀疑泄露：回到 Settings → API 让旧 token 失效（Expire），再生成新的。

`.gitignore` 已排除 `kaggle.json`、`access_token`、`.env`。

---

## 4. 队友入门（onboarding）

1. **拿到仓库**：通过私有仓库（只邀请队员）或私有网盘获得项目文件。不使用公开仓库（规则 5，见 `docs/ARCHIVE_RULES.md` §8）。
2. **装 Python 3.11**：Anaconda 或 python.org 均可，建议 3.11.5 以保证完全一致。
3. **建环境**：按 §1 "重建环境" 的第 2–6 步执行（把第 2 步的路径换成自己的 Python 3.11）。
   - 注意：`requirements-lock.txt` 在 Windows 上生成，含 Windows 专用包（如 `pywinpty`）。macOS / Linux 队友如果锁文件装不上，改用 `pip install -r requirements.txt`，然后对比 `pip freeze` 与锁文件中的核心包版本（scikit-learn、numpy、pandas、scipy、lightgbm、xgboost）是否一致；不一致的结果不能与我们的 run 直接比较。命令里的 `.venv\Scripts\python.exe` 在 macOS / Linux 上对应 `.venv/bin/python`。
4. **Kaggle**：确认已加入 Kaggle 队伍并接受比赛规则；按 §2 下载数据并运行 `scripts\verify_data.py`。
5. **身份**：
   - 在 `project.yaml` 的 `team.members` 填写自己的条目（`id`、`name`、`student_id`、`kaggle_display_name`、`extension`）。
   - 每次开 PowerShell 设 `$env:SSC_AUTHOR = "<你的 id>"`，或者运行脚本时加 `--author <id>`。
   - 设置 git 身份（`docs/ARCHIVE_RULES.md` §9）。
6. **读文档**：`docs/ARCHIVE_RULES.md`（必读）→ `docs/VALIDATION_PROTOCOL.md` → `docs/TIMELINE.md` → `docs/ASSIGNMENT_CHECKLIST.md`。
7. **试跑**（只写沙盒）：
   ```powershell
   .venv\Scripts\python.exe -m pytest
   .venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --data synthetic --quick
   ```
8. 在 `registry/ai_usage.md` 中补充自己会用到的 AI 工具（如有）。
