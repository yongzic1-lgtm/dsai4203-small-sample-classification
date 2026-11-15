# E3 信息泄漏 (Information leakage)

脚本：`extensions/e3_leakage/run.py`，参数：`extensions/e3_leakage/config.yaml`。
报告部分 (<= 2 页，英文，标题写 owner 姓名) 结构固定为 **Question / Protocol / Results / Conclusion**，见第 4 节的对应关系。

---

## 1. 问题 (Question)

Filter = 每个特征与 target 的 |corr|，保留 |corr| 最大的 k 个特征。两种用法：

1. **leaky**：先用全部 250 行选 top-k，再在缩减后的数据上做 k-fold CV；
2. **honest**：k-fold CV，每个 fold 只用该 fold 的 training part 重新选特征。

(1) 在选特征时看到了之后要当 validation 的行的 label，所以它的 CV 分数应该偏高。要回答：
**偏高多少**（两个 CV 分数互相比较，并与 leaderboard 比较）？**gap 如何随 k 变化，为什么？**

注意 brief 里的 "k" 有两个含义，报告里必须明确写出两者都覆盖了：
`k_feat` = 保留的特征数，`k_fold` = CV 折数。本扩展在 `k_feat x k_fold` 的网格上测量 gap。

---

## 2. 协议清单 (Protocol checklist)

| Brief 要求 | run.py 中的实现 | config.yaml 参数 |
|---|---|---|
| filter = 150 个特征各自与 target 的 \|corr\|，保留 top-k | `ssc.features.CorrelationTopK(k, method)`（内部是 `abs_correlation` + `top_k_indices`，并列时取较小列号） | `selection.method: pearson` |
| (1) 用全部 250 行选一次，再做 k-fold CV | `leaky_columns()`：`CorrelationTopK` 在全部行上 fit 一次 -> 列号冻结在 `Pipeline(scaler, FixedColumns(cols), model)` -> `ssc.validation.cross_validate` | – |
| (2) 每个 fold 内只用 training part 重新选 | `build_pipeline(pipeline + selector corr_topk)` = `Pipeline(scaler, CorrelationTopK(k_feat), model)`，selector 在每个 fold 内重新 fit | `pipeline`（不含 selector，脚本自动加） |
| 两种方法用**完全相同**的 splits | 每个 `k_fold` 只调用一次 `make_splits(y, Scheme(kfold, k_fold, n_repeats, seed))`，两种方法和所有 `k_feat` 共用；`split_signature` 不一致直接报错；表里有 `split_sig` 列 | `cv.seed: 2026`, `cv.stratify: true` |
| 覆盖 k = 特征数 | 网格维度 1 | `grid.k_feat: [5, 10, 20, 50, 100, 150]` |
| 覆盖 k = 折数 | 网格维度 2 | `grid.k_fold: [3, 5, 10]` |
| 每个格子重复多个 split seed，报告 spread 而不只是均值 | repeat r 用 `StratifiedKFold(shuffle=True, random_state=seed + r)`；每个 repeat 的 CV 估计 = 该 repeat 所有 fold AUC 的均值；报告 mean / SD / q05 / q95 / min / max | `cv.n_repeats: 20` |
| 模型本身的随机性固定 | `pipeline.random_state` 固定（liblinear 是确定性的） | `pipeline.random_state: 0` |
| 比较两个 CV 分数 | 每个格子：`leaky_mean/std`、`honest_mean/std`、`gap_mean/std/q05/q95`、`frac_repeats_leaky_higher`、`win_rate_fold`、Nadeau-Bengio corrected paired t-test (`ssc.validation.paired_compare`) -> `e3_grid.csv` | – |
| 与 leaderboard 比较 | 用全部 250 行 fit 最终模型 `scaler + CorrelationTopK(k) + model`（全数据上两种方法得到的模型完全相同），test 预测存为运行目录里的 `test_pred.npy`；`e3_leaderboard.csv` 放 leaky CV / honest CV / public LB；LB 从 `registry/submissions.csv` 读取（`ssc.tracking.read_registry` + `ssc.paths.submissions_csv`） | `leaderboard.k_feat: 20`, `leaderboard.k_fold_reference: 5` |
| "why"：证明偏差来自 selection 而不是 signal | null experiment：label 随机置换后两种方法都重跑；honest 应约 0.5，leaky 明显 > 0.5 | `null_experiment.*`（10 次置换 x 5 个 seed，5-fold） |
| sanity check | `k_feat = 150` 时两种方法是同一个模型 -> 每个 fold 的差必须**恰好**为 0（`max_abs_fold_diff == 0`）；null 下 leaky > honest、honest ≈ 0.5；控制台打印 `[PASS]/[FAIL]`，任何 FAIL 时退出码为 2 | – |
| 合成数据上的额外检查 | synthetic 的 hidden test labels 已知：直接给出最终模型的真实 public/private AUC（`synthetic_public_auc/private_auc`），以及选中的真实特征数 | – |

两种方法中 `StandardScaler` 都在 fold 内 fit，**唯一的区别是特征选择用了哪些行**，所以 gap 只量化 selection leakage。

---

## 3. 运行方式 (How to run)

在项目根目录运行：

```powershell
# 1) 冒烟测试 (synthetic，写入 sandbox/)，约 5-10 s
.venv\Scripts\python.exe extensions\e3_leakage\run.py --data synthetic --quick

# 2) 完整设置 (synthetic)：检查代码 + 估计耗时
.venv\Scripts\python.exe extensions\e3_leakage\run.py --data synthetic

# 3) 正式研究运行 (真实数据，写入真实 archive: experiments/, registry/, reports/)
.venv\Scripts\python.exe extensions\e3_leakage\run.py --author <你的 id>

# 4) 交了 LB 之后：只读 registry 补 LB 列，不重新计算
.venv\Scripts\python.exe extensions\e3_leakage\run.py --refresh-lb <run_id>
```

- `--quick`：`k_feat {5, 20, 150} x k_fold {3, 5}`，3 个 split seed，null 2 x 2。数字**不能**进报告；报告用表图只写到 `<run>/report_preview/`（与 E1、E2 一样），不会覆盖 `reports/tables|figures`。但 run 本身仍会进入所选档案的 registry，所以 **`--quick` 只和 `--data synthetic` 一起用**。
- `--data` 不写时取 `$SSC_DATA`，再不写就是 raw（与 `scripts/` 一致）。
- 完整设置实测耗时（本机，`--n-jobs 1`，synthetic）：grid 39.5 s + null 29.5 s，总计约 **72 s**（含 import 约 77 s）。真实数据规模相同，耗时相同。
- `--n-jobs`：logreg 每次 fit 只要几毫秒，joblib 进程开销大于收益（quick 下 `--n-jobs 2` 反而从 4 s 变成 9 s）；只有把 `pipeline.model` 换成 rf/lgbm 等重模型时才值得开。结果与 `n_jobs` 无关（已验证一致）。
- 开发测试时设置 `SSC_ARCHIVE_ROOT` 到私人临时目录；**正式运行不要设**。

### Leaderboard 列怎么填

运行结束时若 registry 里没有该 run 的提交，脚本会打印步骤：

1. `.venv\Scripts\python.exe scripts\make_submission.py --run <run_id> --desc "E3 final model corr_topk(20)" --expected-public <你上传前相信的 public 分数，例如 honest CV 均值>`（core pipeline 的提交脚本：用 `test_pred.npy` 生成 CSV 并写 `registry/submissions.csv`；`--desc` 和 `--expected-public` 必填；synthetic 演练时加 `--data synthetic`）；
2. 上传（网页，或 `scripts\kaggle_submit.py --sub S###`），然后立即记录：`.venv\Scripts\python.exe scripts\record_lb.py --sub S### --submitted-at now --public <分数>`（用 kaggle_submit.py 上传时时间已自动记录，去掉 `--submitted-at now`）。不要直接调用 Python API 改 registry（`docs/ARCHIVE_RULES.md` §1.6；`record_lb.py` 负责 UTC 时间和校验）；
3. `run.py --refresh-lb <run_id>` -> 重写 `reports/tables/e3_leaderboard.csv`（quick run 写到它的 `report_preview/`；加上 `public_lb`、Hanley-McNeil 近似 SE `public_lb_se_approx`、`leaky_minus_public`、`honest_minus_public`；截止后有 private 也会一并读入）。

最终模型对两种方法是同一个，所以**一次提交**就够（每天限 5 次，别浪费）。

---

## 4. 输出 (Outputs) 与报告结构的对应

**运行目录** `experiments/<run_id>/`（原始结果，不可修改）：

| 文件 | 内容 |
|---|---|
| `config.yaml`, `meta.json`, `notes.md`, `summary.json` | 解析后的配置、环境、假设、汇总（含 sanity 结果和耗时） |
| `fold_scores_grid.csv` | 每个 (procedure, k_feat, k_fold, repeat, fold) 的 AUC |
| `repeat_scores_grid.csv` | 每个 split seed 的 leaky / honest CV 估计和 gap |
| `cells.csv` | 每格汇总（同 `e3_grid.csv`） |
| `null_fold_scores.csv`, `null_repeat_scores.csv`, `null_cells.csv` | 置换 label 实验 |
| `leaderboard.csv` | LB 表的 CV 部分（`--refresh-lb` 以它为底） |
| `final_features.csv` | 最终模型（全数据）选中的特征及 \|corr\|（synthetic 上有 `is_true`） |
| `test_pred.npy` | 最终模型的 test 预测，供 `scripts/make_submission.py --run <run_id>` |
| `*.png` | 三张图的副本 |

registry 行：`auc_mean` = 参考格子 (`k_feat=20, 5-fold`) 的 **honest** CV；`model/selector` 列描述 `test_pred.npy` 背后的模型（含 `corr_topk(k=20)`），运行目录里的 `config.yaml` 保留不含 selector 的 base pipeline。

**报告用输出** (`reports/tables/`, `reports/figures/`，每次非 quick 运行覆盖；`e3_manifest.json` 记录来自哪个 run_id)：

| 报告部分 | 用什么 |
|---|---|
| **Question** | 第 1 节的问题；明确写出 k 的两个含义都覆盖了 |
| **Protocol** | 第 2 节清单写成一段话：数据 (250 x 150)、filter、两个 procedure、网格、20 个 split seed、identical splits、固定模型 seed、模型、null experiment、LB 对比；引用 Ambroise & McLachlan (2002)、Hastie et al. (ESL, 2nd ed., Sec. 7.10.2)、Nadeau & Bengio (2003)、Hanley & McNeil (1982) |
| **Results**（必须有表或图） | 主图 `e3_gap_vs_kfeat.png`；表：`e3_grid.csv` 的 5-fold 行（leaky、honest、gap +/- SD、p）+ `e3_leaderboard.csv`；`e3_null.png` 或 `e3_null.csv` 一句话 + 小表；`e3_cv_vs_kfeat.png` 可选 |
| **Conclusion** | 偏高多少（在 k_feat=20 时多少 AUC、gap 最大在哪个 k_feat）；随 k_feat 和 k_fold 如何变化及原因（第 5 节）；与 LB 比较时哪个 CV 更接近（注意 public LB 的 SE 约 0.02-0.03）；对本队 validation protocol 的含义：**任何用到 label 的步骤都必须在 fold 内** |

---

## 5. 为什么 gap 依赖 k_feat 和 k_fold（报告 Conclusion 的推理依据）

**记号**：n = 250，K = k_fold。某个 noise feature（与 y 独立）在全部行、training part (T)、validation part (V) 上的样本相关系数分别为 r_all、r_T、r_V。近似有
r_all ≈ ((K-1)/K) r_T + (1/K) r_V，且 r_T ~ N(0, K/((K-1)n))，r_V ~ N(0, K/n) 相互独立。

模型在 T 上训练，给特征的权重大致与 r_T 同号；它在 V 上"蒙对"的程度取决于 **alignment** r_T · r_V。

- **honest**：按 r_T 选，而 r_V 与 r_T 独立 -> 被选中的 noise feature 期望 alignment = 0。
- **leaky**：按 |r_all| 选。由二元正态条件期望可推出

  **E[r_T · r_V | r_all = s] = s^2 - 1/n**，与 K 无关。

  即：全数据上 |r| 大于其噪声水平 1/sqrt(n) ≈ 0.063 的 noise feature，在 T 和 V 上"看起来同向"，像一个真实特征；|r| 更小的则是负 alignment。

### 随 k_feat：非单调

- **k_feat = 150**：没有选择，两个 procedure 是同一个模型 -> gap **恰好为 0**（sanity check；期望意义上也有 Σ_all (s^2 - 1/n) = 0）。
- **很小的 k_feat**：如果最强的几个特征是真实信号，两种方法都会选中它们（honest 的 fold 内选择也能找回来），真实特征的 alignment 不是泄漏 -> gap 小。若信号很弱，小 k 会选到 |r| 最极端的 noise feature，此时 gap 也可能大（null 实验中 k=5 的 gap 就不小）。
- **中间的 k_feat**：选进了大量 noise feature，每个都是因为（部分地）与 validation rows 的 label 碰巧相关才入选；累计泄漏 Σ_{top-k}(s^2 - 1/n) 随 k 先增后减，纯噪声时在 |z| = 1 处（约 32% 的 noise features，即 150 个中约 48 个）达到最大，k -> 150 时回到 0。AUC 的峰值会比 48 更靠左，因为特征越多模型分数里的噪声方差越大。
- synthetic（10 个真实特征）完整运行：5-fold gap = 0.040 (k=5)、0.032 (10)、0.069 (20)、**0.080 (50)**、0.050 (100)、0 (150)；k=10 处 gap 下降正是因为 10 个真实特征占满了名额。null 下 gap 峰值在 k=20 (0.203)，k=50 次之 (0.194)，与上面的推导一致。

### 随 k_fold：一阶上不变，让数据决定

直觉是"折数越多，每个 validation part 在选特征用的行里所占比例 1/K 越小，泄漏应该被稀释"。但这只说对了一半：validation part 越小，它自己的相关系数 r_V 越极端（方差 K/n），这两个效应**恰好抵消**——上面的条件期望 s^2 - 1/n 不含 K。所以一阶上 leaky 的偏差与 K 无关。

二阶效应（方向不确定，看数据）：

- K 越大，honest 的 training part 越大（167 -> 200 -> 225 行），fold 内选特征更准、模型更好，honest CV 上升（k-fold 的悲观偏差减小）；leaky 的训练集也变大；
- validation part 越小（83 -> 50 -> 25 行），fold AUC 噪声越大，AUC 是有界非线性函数，近似不再精确。

synthetic 完整运行：k_feat = 20 时 gap = 0.068 / 0.069 / 0.072（3 / 5 / 10-fold），差异远小于 gap 的 SD（约 0.01-0.02）——基本是平的，与推导一致。真实数据上请看 `e3_gap_vs_kfeat.png` 中三条线是否重合再下结论。

---

## 6. 常见陷阱 (Pitfalls)

1. **两个 k 的歧义**：报告里必须写明 k_feat 和 k_fold 都测了，各自的结论分开写。
2. **identical splits**：比较 leaky vs honest 时不要改 `cv.seed`；`split_sig` 列可核对。不同 `k_fold` 之间 splits 自然不同，不能做 paired 比较。
3. **split seed 的 SD 不是 standard error**：20 个 repeat 共用同一批 250 行，不独立。不要对 repeat 做普通 t-test；用 corrected t (Nadeau-Bengio)，并强调 spread。
4. **scaler 也要在 fold 内**：本脚本两种方法的 scaler 都在 fold 内，gap 只来自 selection。若手写代码时把 scaler 放到 CV 外面，会混入另一个（较小的）泄漏。
5. **LB 噪声**：public LB 只有约 395 行，AUC 的 SE 约 0.02-0.03（`public_lb_se_approx`）。一个 LB 分数无法区分小于约 0.05 的差别；结论应是"honest 与 LB 的差在 SE 范围内 / leaky 超出多少个 SE"，而不是只看一个数。
6. **public vs private**：截止后把 private LB 补进 registry 再 `--refresh-lb`，private 有约 19,355 行，才是可靠的对照。
7. **reports/tables 会被覆盖**：每次正式运行后在下面的结果记录里写下 run_id；提交 LB 后记得 `--refresh-lb`。
8. **--quick 和 synthetic 的数字不是证据**：synthetic 只检查代码是否按预期工作，不能当作对真实数据的结论。
9. **null 实验的 honest 约 0.5 而非恰好 0.5**：有限样本下会有 +/- 几个百分点的波动；看的是 leaky 与 honest 的差，而不是 honest 是否等于 0.5。
10. **换模型**：`pipeline.model` 可改（如 L1 logreg：`l1_ratio: 1.0`）。sklearn 1.9 中 `penalty` 已弃用，必须用 `l1_ratio`。若模型有随机性，务必固定 `random_state`，否则 k_feat=150 的 gap 不再恰好为 0。

---

## 7. 参考：synthetic 完整运行的预期输出（用于核对脚本是否正常）

`--data synthetic`（完整设置）关键数字：

| k_feat (5-fold) | leaky | honest | gap +/- SD |
|---|---|---|---|
| 5 | 0.881 | 0.841 | 0.040 +/- 0.011 |
| 20 | 0.917 | 0.848 | 0.069 +/- 0.012 |
| 50 | 0.896 | 0.817 | 0.080 +/- 0.009 |
| 150 | 0.794 | 0.794 | 0 (exact) |

Null（5-fold）：leaky 0.61-0.72，honest 0.50-0.52。LB 对照（k_feat=20）：leaky 0.917，honest 0.848，最终模型真实 public AUC 0.872 / private 0.877——honest 略悲观（训练集少 20%），leaky 明显过高。四个 sanity check 全部 PASS。

---

## 8. 结果记录 (Results log)

每次**完整**运行（真实数据）后由 owner 追加一行。run_id 对应 `experiments/<run_id>/`；结论写完后也要写进该运行目录的 `notes.md` 的 Conclusion 部分。

| 日期 | run_id | data | 配置改动（相对 config.yaml） | 关键数字（k_feat=20, 5-fold: leaky / honest / gap；gap 最大处；null） | public LB (sub_id) | 结论 | author |
|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | | raw | 无 | | | | |
| 2026-10-22 | 20261022-063908_e3_leakage | raw | config.yaml 已改（ext34，正式 run 前）：k_feat 加 2/7/13/30（7 = 强 \|corr\| 特征组大小，13 = 无标签高方差块大小，2/30 补曲线）；k_fold 加 2；null 置换 10 -> 30，null 的 k_fold [5] -> [3,5,10]（检验 k_fold 一阶无关的论断）。模型不变（L2 logreg C=0.1，即 brief 的线性模型）。Colab job J20261022-064045-pro1-rext34（--no-sync，VM 上代码 = bundle b413e6aa3bae4443） | leaky 0.867 / honest 0.793 / gap 0.073（SD over 20 split seeds 0.014；Nadeau–Bengio corrected SE 0.020，p 0.0003；`cells.csv` gap_std 0.0143，corrected_t 3.72）*（更新 2026-10-22；原写 "gap 0.073 +/- 0.014 (p 0.0003)"，± 是 split seed 间的 SD，不是 SE）*；gap 最大 0.090 @ k_feat 30-50（5-fold），0.115 @ 50 / 2-fold；k_feat=7 gap 0.001；null：honest 0.49-0.51，leaky 最高 0.71 (k_feat 30)，gap 不随 k_fold 降 (0.201/0.206/0.210) | **S004，public 0.79136**（2026-10-22 16:08 JST 上传，`registry/submissions.csv`；test_pred.npy, k_feat=20）：leaky − public +0.075（3.2 LB SE，SE 0.023），honest − public +0.002（`reports/tables/e3_leaderboard.csv`）；预登记 B005：proxy test AUC 0.820（R321 20261022-072942，GMM 伪标签 19,750 行），预期 public 0.820 +/- 0.021，90% 0.785–0.854，public 落在区间内 *（更新 2026-10-22；原写"待提交"）* | leaky 明显偏高、honest 略悲观；gap 在 k_feat 上非单调（7 个强特征被每折重新选中时为 0）；真实数据 gap 随 k_fold 下降是二阶效应（honest 变好），null 下平。*（更新 2026-10-22：在 public 上 honest 只高 0.002，部分是运气（LB SE 0.023）；"honest 略悲观"是相对 proxy 而言（−0.026）；见 `reports/extensions/e3_leakage.md`）* | ext34 (AI) |

**run.py 修改记录（ext34, 2026-10-22）**：(1) `SERIES` 加第 4 个颜色并按 `len(SERIES)` 取模——4 个 k_fold 时第 1、4 条线原来同色（正式 run 前修，已在 20261022-063908 的图里生效）；(2) `fig_null` 每个 panel 的长标题在 null 用多个 k_fold 时互相重叠，改为短标题 + suptitle（正式 run **之后**修，20261022-063908 的 `e3_null.png` 仍是重叠版本，报告不用这张图；数值不受影响）。
