# E2 Learning curve（学习曲线）

负责人：`TODO_NAME`（报告页眉必须写成员姓名）　｜　脚本：`run.py`　｜　参数：`config.yaml`

## 1. Question（要回答的问题）

> 用 250 行中大小为 50, 100, 150, 200, 250 的子集训练，AUC 随训练行数如何变化？
> 每个大小抽 10 个不同的随机子集，**平均分数是多少、子集之间差多少**？
> 把曲线外推到 **500 行训练数据**，预期 AUC 是多少？每个大小上的 spread 允许我们对这个估计有多大信心？

为什么重要：它回答“模型是不是还缺数据”（曲线是否已平），也给最终模型的期望分数一个参照
（报告 Conclusion 可以联系 core task：在这个 n 上更灵活的模型是否值得）。

## 2. Protocol checklist（brief 要求 -> run.py / config.yaml 中的具体实现）

| # | Brief 要求 | 实现（参数 / 步骤） |
|---|---|---|
| 1 | 子集大小 50, 100, 150, 200, 250 | `sizes: [50, 100, 150, 200, 250]` |
| 2 | 每个大小抽多个不同随机子集（10 个） | `n_subsets: 10`；子集 i（大小 n）用 `np.random.default_rng([subset_seed, n, i])` 分层无放回抽取（`stratified_subset()`，正类比例与全数据一致）；不同大小的子集是独立抽取，不是嵌套 |
| 3 | 每个子集分别训练、打分 | `score_subset()`：在子集**内部**做 repeated stratified k-fold CV（`cv`: 5-fold x 4，`ssc.validation.make_splits` + `cross_validate`），子集 i 的 CV 种子 = `cv.seed + i * cv_seed_stride`；模型 seed（`pipeline.random_state`）固定 |
| 4 | 记录有效训练行数 | 每个 CV 分数背后的模型只用了 n*(k-1)/k 行：`n_train_eff`, `x` 列（40, 80, 120, 160, 200） |
| 5 | 第二视角（n < 250） | `out_of_subset: true`：用整个子集训练，在不在子集里的 250−n 行上打分（`oos_auc`, `n_oos_rows`）；n=200 时只有 50 行打分，自身噪声大 |
| 6 | 报告每个大小的 mean 与差异程度 | `e2_learning_curve_table`: `cv_mean, cv_std, cv_sem, cv_min, cv_max`（以及 out-of-subset 的同样统计量）；图 `e2_learning_curve.png`：10 个单点 + mean ± std |
| 7 | 单个子集无法区分趋势与子集间波动 | 每个大小 10 个点全部画出；`spread_source` 列注明来源 |
| 8 | n = 250 时子集都等于全数据 | 只改变 CV 种子；表中 `spread_source = "CV seed only"`，图上注明 “spread = CV-seed noise only”，即**只有 evaluation noise** |
| 9 | 外推到 500 行 | inverse power law `AUC(x) = a − b·x^(−c)`（`scipy.optimize.curve_fit`，bounds `a ∈ [0.5, 1]`，`b ≥ 0`，`c ∈ [0.01, 3]`），拟合每个大小的均值，权重 σ = SEM = std/√10（`weight: sem`，下限 `sigma_floor`）；`fit.x_axis: effective` → **x 是有效训练行数**，`target_n: 500` 表示“用 500 行训练的模型”；`predict_at: [250, 500]` |
| 10 | 说明对外推有多大信心 | bootstrap：每个大小内对 10 个子集分数有放回重抽 → 重新拟合 → 收集 AUC(500)，`n_boot: 2000`，报告 median 和 90% 区间（`interval: 0.90`）；另报 250 → 500 的预测增益及 P(gain > 0) |
| 11 | 简单替代模型做 sensitivity check | `log_linear`：`AUC = α + β·ln x`（同样权重、同一 bootstrap）；另有 `sensitivity`: `power_law_unweighted`（不加权）、`power_law_drop_50`（去掉最小的 size） |
| 12 | 归档 | `ssc.tracking.Run(kind="extension", name="e2_learning_curve")`，原始结果存入 run 文件夹 |

**x 轴约定（必须在报告里写清）**：默认 `effective`，所有点画在“该分数背后的训练行数”上：
CV 分数在 n*(k−1)/k，out-of-subset 分数在 n（它的模型用了整个子集）。外推目标 500 也是训练行数。
若改成 `x_axis: nominal`，x = 子集大小 n，`target_n: 500` 就变成“在 500 行数据上做 5-fold CV”（模型实际用 400 行训练）。

## 3. How to run（从项目根目录运行）

```powershell
# 开发 / 测试时：把归档重定向到私人临时目录
$env:SSC_ARCHIVE_ROOT = "D:\tmp\<your-dir>\e2"

# quick smoke test（sizes 50/100/150/250，每个 3 个子集，CV 5x1，bootstrap 100；约 10 s）
.venv\Scripts\python.exe extensions\e2_learning_curve\run.py --data synthetic --quick

# full 设置，合成数据（附带 synthetic ground truth 检查）
.venv\Scripts\python.exe extensions\e2_learning_curve\run.py --data synthetic --n-jobs 8

# 正式：真实数据（先 Remove-Item Env:SSC_ARCHIVE_ROOT）
.venv\Scripts\python.exe extensions\e2_learning_curve\run.py --n-jobs 8 --author <id>
```

参数：`--data {raw,synthetic}`（默认 `$SSC_DATA`，再不写就是 raw，与 `scripts/` 一致）、`--config`、`--quick`、`--author`、`--n-jobs`、
`--replot RUN_ID`（2026-11-01 加：只用该 run 文件夹里已存的表重画报告图，不拟合模型、不建新 run、不写 registry；文件名前缀取自命名该 run 的 `reports/tables/e2*_manifest.json`；`--replot-out DIR` 写到别处预览），
（并行的是子集打分；bootstrap 是串行的，占大部分时间；结果与 n_jobs 无关）。

运行时间（本机，合成数据，full 设置）：约 65–80 s（`--n-jobs 8` 与串行几乎相同：CV 部分约 9 s，2000 + 2 x 500 次 bootstrap 拟合约 55–70 s，取决于机器负载）；`--quick` 约 10 s。
总拟合次数：50 个子集 x 20 折 + 40 次 out-of-subset = 1,040 次（合成数据另加 100 次 ground-truth 拟合）。

## 4. Outputs（输出文件）与报告结构的对应

run 文件夹 `experiments/<run_id>/`（raw）或 `sandbox/experiments/<run_id>/`（synthetic）：

| 文件 | 内容 |
|---|---|
| `subset_scores.csv` | 每个 (size, subset) 一行：`cv_auc`、`cv_fold_std`、`oos_auc`、`frac_constant_folds`、种子、`subset_hash`、split signature |
| `fold_scores_all.csv` | 所有 fold 的 AUC |
| `subset_membership.npy` | bool 矩阵 (50, 250)：每个子集用了哪些行（行顺序同 `subset_scores.csv`） |
| `bootstrap.csv` | 每个 bootstrap replicate 的参数 (a, b, c / α, β) 与预测值 |
| `ground_truth.csv` | 仅 synthetic：用 mock test 的新行训练、在 5,000 行上打分的“真实”学习曲线 |
| `learning_curve_table.csv`, `extrapolation.csv`, `*.png` | 报告输出的副本 |

报告输出（full 运行写入 `reports/tables/`、`reports/figures/`；`--quick` 只写到 `<run>/report_preview/`；`.md` 只含报告用的列，`.csv` 含全部列）：

| 文件 | 用在报告哪一段 |
|---|---|
| `e2_learning_curve_table.{csv,md}` | **Results** 表：每个 size 的 mean / std / min / max（CV 与 out-of-subset）、`frac_constant_folds` |
| `e2_extrapolation.{csv,md}` | **Results / Conclusion**：AUC(250)、AUC(500) 的点估计、bootstrap median 与 90% 区间；各 sensitivity 行；`envelope` 行 = 所有区间的并集（报告里给的区间至少用它）；synthetic 时还有 `true_auc_synthetic` 与 `covered` |
| `e2_ground_truth.{csv,md}` | 仅 synthetic（流程检查，不进报告） |
| `e2_learning_curve.png` | **Results** 主图：单点、mean ± std、power-law 拟合（实线 = 观测范围，虚线 = 外推）、nominal 90% band（名义区间；复刻上只覆盖 21%，R942）、log-linear、sensitivity 点、AUC(500) 区间。2026-11-01 起按印刷尺寸画（4.40 x 2.49 in，报告里 70% 宽度），300 dpi |
| `e2_bootstrap_target.png` | **Conclusion** 辅图：AUC(500) 的 bootstrap 分布（power law vs log-linear，标出 AUC = 1） |
| `e2_manifest.json` | 这些文件来自哪个 run_id |

两页报告建议：
- **Question**：第 1 节。
- **Protocol**：sizes、10 个分层子集、子集内 5-fold x 4 CV、n = 250 只变 CV seed、模型及固定 seed、x 轴约定、power law + 权重 + bootstrap + log-linear（第 2 节）。引用 scipy `curve_fit`、学习曲线的 inverse power law 形式（例如 Figueroa et al. 2012, BMC Med Inform Decis Mak）。
- **Results**：`e2_learning_curve.png` + `e2_learning_curve_table.md`（选列）+ `e2_extrapolation.md` 的两三行。
- **Conclusion**：AUC(500) 的估计和区间；250 → 500 的增益是否大于区间宽度；不同曲线形式之间的差异（model-form uncertainty）；局限（第 5 节第 1、4 条）。

## 5. Pitfalls（常见陷阱）

1. **子集来自同一批 250 行，彼此高度重叠**（两个 200 行子集至少共享 150 行）。所以大 n 处的 spread 低估了“换一批 n 行新数据”的真实波动，整条曲线也只是“这 250 行”条件下的曲线；n = 250 处只有 CV 噪声。bootstrap 把 10 个分数当成独立样本，因此 **90% 区间只反映子集间噪声，是不确定性的下限**。
2. **合成数据上的教训（2026-10-22 脚手架检查）**：主拟合 AUC(500) = 0.943 [0.931, 0.953]，而 ground truth = 0.917，区间**没有覆盖**真值；连**短外推 x = 250** 也没覆盖：0.910 [0.906, 0.913] vs 真值 0.903；预测增益 250→500 +0.033 [+0.025, +0.040]（P(gain > 0) = 1.00），真实增益只有 +0.015。原因：所有子集都来自同一批 250 行（这批行的 CV 0.892 比 200 行训练的真实 AUC 0.871 高，属于 parent-sample 效应，子集重抽样看不到），而 SEM 加权（下限 0.002）把曲线钉在只含 CV 噪声的 n = 250 点上。去掉 n = 50 的 sensitivity 拟合给出 0.919 [0.910, 0.939]，覆盖了真值；log-linear 给出 1.01（不可能 > 1）。即 **曲线形式、最小 size 和 parent sample 带来的不确定性远大于 bootstrap 区间**。`e2_extrapolation` 里的 `envelope` 行（主拟合与各 sensitivity 区间的并集）是报告里**最少**要给的范围，并写明它仍不含 parent-sample 噪声；不要只报一个窄区间。（对 250 行做外层 bootstrap 再重跑整条流程不可行：有放回抽样产生的重复行会同时落在 CV 的训练和验证折里，本身就是泄漏。）
3. **固定超参数的模型在小 n 上会退化**：L1 logreg C=0.1 在 40 行训练时常把所有系数压成 0，预测为常数 → AUC = 0.5（合成数据 n = 50 时 `frac_constant_folds` ≈ 0.53）。曲线左端因此反映的是正则化强度，而不只是数据量；这会让 power law 的拟合偏陡、外推偏乐观。liblinear 的 C 乘的是损失之和，n 越大有效正则越弱。可考虑：去掉退化的 size（`drop_smallest`），或换成在子集内调参的模型（更贵）。
4. **外推距离**：观测的最大训练行数是 200，外推到 500 是 2.5 倍；三参数模型只拟合 5 个点（2 个自由度）。看 `at_bounds`：若 `a` 落在上界 1，渐近线没被数据确定。log-linear 无上界，`frac_boot_above_1` > 0 说明它只是粗略的敏感性检查。
5. **CV 在小子集里的噪声**：n = 50 时每个验证 fold 只有 10 行（约 4–5 个正类），单折 AUC 极不稳定；这就是 n = 50 的 std 很大的原因，也是用 SEM 加权的理由。`sigma_floor` 防止 n = 250（只有 CV 噪声，std 很小）的点权重过大；可用 `weight: none` 做对比（表中 `power_law_unweighted`）。
6. **两种视角的含义不同**：CV 分数对应 n*(k−1)/k 行训练的模型；out-of-subset 对应 n 行训练的模型，但打分行数 250−n 随 n 变小（n = 200 时只有 50 行）。两者画在同一条“训练行数”轴上才可比。
7. **“500 行”是假设情景**：比赛不允许外部数据、也没有测试集标签，所以外推只是在估计“更多数据值多少”，不能用来训练。LB 分数（public 只有约 395 行，SE ≈ 0.02）不能用来验证外推。
8. **不要看了结果再改 sizes / 权重 / bounds 去追一个好看的数字**（forking paths）；改动请写进 results log 并说明理由。
9. **归档分离**：不设 `SSC_ARCHIVE_ROOT` 时 `--data raw` 写真实 archive；测试一律用 synthetic 或重定向；`--quick` 不会写 `reports/`。

## 6. Results log（每次 full 运行后由负责人填写）

| 日期 | run_id | data | 设置变化 | 关键数字 | 结论 / 下一步 | 作者 |
|---|---|---|---|---|---|---|
| 2026-10-22 | （临时 archive，仅流程检查） | synthetic | 默认 config，`--n-jobs 8`，65–80 s | CV mean 0.566 / 0.722 / 0.835 / 0.874 / 0.892，std 0.109 / 0.087 / 0.018 / 0.008 / 0.003（n = 50…250）；AUC(500) power law 0.943 [0.931, 0.953]，drop_50 0.919 [0.910, 0.939]，log-linear 1.01；truth 0.917；预测增益 250→500 +0.033 [+0.025, +0.040]，真实增益 +0.015 | 曲线单调上升、spread 随 n 缩小，流程正确；主区间过窄、未覆盖真值（见第 5 节第 2 条）。合成结果不是对真实问题的证据 | scaffold |
| 2026-10-22 | `20261022-070248_e2_learning_curve` | raw | (a) 默认 config = 监督基线 L1 logreg C=0.1、全部 150 个特征（= R003 sup_l1_all / S001）；Colab job `J20261022-070426-pro1-rext12`（8 核，`research/r307_e2_all_models.py --n-jobs 8`），49 s | CV mean 0.659 / 0.735 / 0.810 / 0.830 / 0.847，std 0.100 / 0.040 / 0.028 / 0.021 / 0.006（n = 50…250）；n = 50 时 24.5% 的折是常数预测；AUC(500) power law 0.906 [0.881, 0.916]，drop_50 0.869，log-linear 0.945，envelope [0.857, 0.974]；增益 250→500 +0.040 [+0.022, +0.047] | 曲线还在上升（缺数据），但外推值只能说“约 0.87–0.95”；曲线形式的不确定性是 bootstrap 区间的 3 倍 | ext12 (AI) |
| 2026-10-22 | `20261022-070339_e2_learning_curve_qda_block` | raw | (b) 监督 QDA reg 0.1、13 列 block（`research/r302_e2_qda_block.yaml`，经 `research/r301_run_extension.py`，prefix `e2_qda_`），同一 job | CV mean 0.973 / 0.990 / 0.992 / 0.995 / 0.996，std 0.022 / 0.005 / 0.003 / 0.0014 / 0.0005；AUC(500) 0.9969 [0.9952, 0.9986]，envelope [0.9952, 1.0]；增益 +0.0008 | QDA 在 40 行训练时已有 0.973，约 100 行后几乎平；再多数据几乎没有收益 | ext12 (AI) |
| 2026-10-22 | `20261022-070445_e2_learning_curve_gmm` | raw | (c) TransductiveGMM K=2、block（`research/r303_e2_gmm.yaml`，prefix `e2_gmm_`），同一 job | 所有 size、所有子集的 CV 与 out-of-subset AUC 都是 1.0000（std 0）；AUC(500) = 1.0 | 无标签结构把标签需求降到几十行以下；`research/r304_label_requirement.py`（run `20261022-070600_r304_label_requirement`）：2 个标签（每类 1 个）时 50 组全是 1.0 | ext12 (AI) |

**模型选择理由（ext12）**：brief 只要求一条曲线；为了回答“无标签结构如何改变标签需求”，在**同一组子集、同一 CV 种子、同一拟合设置**下跑三个模型：(a) 监督基线（本目录 `config.yaml`，未改模型），(b) 监督 QDA（R003 最好的监督模型，0.9958），(c) 团队的最终模型 TransductiveGMM（R003 / S003）。(b)(c) 不能由 `ssc.models.build_pipeline` 构造，所以用 `research/r301_run_extension.py` 把 `research/lib_ext12.build_pipeline` 换进未修改的 `run.py`，并换报告前缀，避免覆盖 (a) 的 `e2_*` 文件。三条曲线的对照：`reports/figures/r307_e2_model_comparison.png`、`reports/tables/r307_e2_model_comparison.md`（run `20261022-070518_r307_e2_model_comparison`）。
注意第 5 节第 3 条在真数据上同样成立：L1 C=0.1 在 n = 50 时有 24.5% 的折把系数全压成 0；而且 C 固定时有效正则随 n 变弱——R300 显示同样 250 行，C = 0.2 的官方 CV 已有 0.873（`20261022-063711_r300_l1_c02`），接近 power law 在 x = 250 的预测 0.865。也就是说，曲线外推出来的“多给数据的收益”有一部分只要放松正则就能拿到。

结论写法提示：先给 AUC(500) 的点估计与 `envelope` 范围（并说明它只是不确定性的下限），再给 250 → 500 的增益
（P(gain > 0) 同样只反映子集噪声），然后用 sensitivity 行说明曲线形式带来的额外不确定性，最后说明对 core task 的含义。
