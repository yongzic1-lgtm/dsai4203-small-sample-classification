# E1 Validation protocol design（验证方案设计）

负责人：`TODO_NAME`（报告页眉必须写成员姓名）　｜　脚本：`run.py`　｜　参数：`config.yaml`

## 1. Question（要回答的问题）

> 用 single hold-out 和 k-fold CV（k = 3, 5, 10，以及 repeated k-fold）评估同一个模型时，
> **只改变 split seed**（决定哪些行进 training / validation 的随机种子），估计值会波动多少？
> **如果两个模型的 AUC 相差 0.01，你会信任哪一种方案来做选择？**

为什么重要：团队每次在两个候选模型之间做决定，看到的 CV 差距往往就是 0.01 量级；
public LB 只有约 395 行，`ssc.validation.auc_standard_error(0.8, 178, 217)` ≈ 0.023，比 0.01 还大，
所以“选哪个模型”只能靠验证方案本身的精度。

## 2. Protocol checklist（brief 要求 -> run.py / config.yaml 中的具体实现）

| # | Brief 要求 | 实现（参数 / 步骤） |
|---|---|---|
| 1 | single hold-out vs k-fold CV | `schemes`: `holdout20`（`kind: holdout, test_size: 0.2`，即 StratifiedShuffleSplit），可选 `holdout30`；k-fold 方案 `kind: kfold` |
| 2 | 改变 k（3, 5, 10） | `schemes`: `3fold`, `5fold`, `10fold`（`n_repeats: 1`） |
| 3 | 每个方案重复多次，**只改变 split seed** | `n_split_seeds: 50`；第 s 次执行的种子 = `seed_base + s * seed_stride (+ r)`，由 `run_execution()` 构造 `Scheme(seed=...)`；模型、数据、其余参数完全不变 |
| 4 | sklearn k-fold 默认不 shuffle | 所有 k-fold 方案都用 `ssc.validation.make_splits`（`StratifiedKFold(shuffle=True, random_state=seed+r)`）；另设 `5fold_noshuffle`（`shuffle: false`）单独一行演示：每个 seed 都是同一个 split，spread = 0；`n_unique_splits` 列给出证据 |
| 5 | 模型若有随机性，固定其 seed | `models.A/B.pipeline.random_state: 0`，所有 split 共用，绝不从 split seed 派生 |
| 6 | 报告 spread 而不只是 mean | `e1_scheme_spread`: `mean, std, iqr, min, max`（每个方案 50 个 estimate），以及 `fits_per_estimate`（成本）；图 `e1_estimates_by_scheme.png`（box + strip） |
| 7 | 一个 estimate 的定义 | hold-out：那一个 validation AUC；k-fold：k 折 AUC 的平均；repeated k-fold（`5foldx10`, `5foldx20` = project.yaml 的 official_cv）：全部 k x R 折的平均 |
| 8 | 0.01 AUC 差距该信任哪个方案 | 模型 A（in-fold corr top-30 + L2 logreg C=0.05）与 B（corr top-50 + 同一 L2 logreg）：两个**相邻候选**，合成数据上 CV 差距约 0.01，代表团队真实会做的决定；两者在**同一组 splits** 上评估（`run_execution()` 对两个模型用同一个 `splits` 列表）；`e1_paired_difference`: `std_diff`、`mdd_2sd = 2 x std_diff` 及其 95% CI（`mdd_lo`, `mdd_hi`）、`sign_agree_ref`、`p_correct_at_target`、`frac_nb_p_lt_0.05`；图 `e1_diff_std_by_scheme.png`（0.01 水平线，误差棒 = std 的 95% CI）。真数据正式 run 前把 A/B 换成团队自己的相邻候选（`config.yaml` 的 `models`） |
| 9 | 参照（reference）符号 | `reference`: 5-fold x 50，种子 900000 起，与所有执行的种子不重叠（`load_config()` 会检查）；输出其 A−B 与 SE，`sign_reliable` = \|diff\| > 2 SE |
| 10 | 全部 split 分层 | `Scheme.stratify` 默认 True（StratifiedKFold / StratifiedShuffleSplit） |
| 11 | 归档 | `ssc.tracking.Run(kind="extension", name="e1_validation_protocol")`，原始逐次结果存入 run 文件夹 |

### 指标定义（写 Protocol 段时直接引用）

- **estimate**：一次执行方案后实践者会报告的数字（见第 7 行）。
- **std / IQR / min / max**：50 个 estimate（50 个 split seed）之间的离散度，只反映 split 带来的噪声。
- **paired difference**：同一 split 上 `estimate(A) − estimate(B)`。
- **MDD（minimal detectable difference）** = 2 x std(paired difference)。std 只来自 50 次执行，本身有抽样误差（50 次时 95% CI 约为 x0.84 到 x1.25），所以 `mdd_lo` / `mdd_hi` 给出 chi-square 95% CI；`note` 只有整个 CI 低于 0.01 时才写 `MDD < target`，整个 CI 高于 0.01 时写 `MDD > target`，否则写 `inconclusive`（此时可加大 `n_split_seeds`）。
- **sign_agree_ref**：A−B 的符号与 reference 符号一致的 seed 比例。
- **p_correct_at_target** = Φ(0.01 / std_diff)：若真实差距正好是 0.01，该方案一次执行排对顺序的概率（正态近似）。
- **frac_nb_p_lt_0.05**：Nadeau-Bengio corrected resampled t-test（`ssc.validation.paired_compare`）在该次执行中判为显著的比例（hold-out 只有 1 折，不适用）。
- **mean_within_fold_se / mean_within_rep_se**（`e1_scheme_spread.csv`）：一次执行内部就能算的误差条，分别是 “fold std / √(折数)”（naive）和 “各 repeat 均值的 std / √R”（仅 R > 1）；与真实的 seed-to-seed std 对比，看单次 run 能否预报自己的 split 噪声。`e1_paired_difference` 里的 `mean_within_rep_se` 是配对差的同一指标。
- **std_diff_unpaired / corr_AB**：若 A、B 在互相独立的 splits 上评估，差值的 std 会是 √(sA² + sB²)；`corr_AB` 是两模型 estimate 跨 seed 的相关系数。std_diff 与 std_diff_unpaired 的差距就是配对带来的收益。
- **n_unique_splits**：50 次执行里不同 split signature 的个数（noshuffle = 1）。
- **cost**：`fits_per_estimate`（每个模型）与 `fits_per_decision`（A 和 B 合计）。

## 3. How to run（从项目根目录运行）

```powershell
# 开发 / 测试时：先把归档重定向到私人临时目录，绝不写入真实 archive
$env:SSC_ARCHIVE_ROOT = "D:\tmp\<your-dir>\e1"

# quick smoke test（4 个 seed，repeats 截到 2，reference 5x4；约 10 s）
.venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --data synthetic --quick

# full 设置，合成数据（检查流程）
.venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --data synthetic --n-jobs 8

# 正式：真实数据（先 Remove-Item Env:SSC_ARCHIVE_ROOT，结果写入 experiments/ 与 reports/）
.venv\Scripts\python.exe extensions\e1_validation_protocol\run.py --n-jobs 8 --author <id>
```

参数：`--data {raw,synthetic}`（默认 raw）、`--config`（默认本目录 config.yaml）、`--quick`、
`--author`、`--n-jobs`（joblib 并行；结果与 n_jobs 无关，种子是确定的）。

运行时间（本机 32 核，合成数据，full 设置，含 5foldx20）：`--n-jobs 8` 约 46 s；`--quick` 约 10 s。
总拟合次数 = 2 x 50 x (1+1+3+5+10+50+100+5) + 2 x 250 = 18,000 次（每次含 fold 内的 |corr| 选择 + logreg）。换成 GBM 等慢模型时成本按比例增加。

`--data` 不写时取 `$SSC_DATA`，再不写就是 raw（与 `scripts/` 一致）。

与 `registry/decisions.md` D002（official CV = 5-fold x 20）和 `docs/VALIDATION_PROTOCOL.md` 的关系：
E1 的 `5foldx20` 行就是 official scheme 的 split-seed 噪声；它的 `std_diff` / MDD 用来确认或修订 D002（§2 与 §5.3 的 `TODO(E1)`）。

## 4. Outputs（输出文件）与报告结构的对应

run 文件夹 `experiments/<run_id>/`（raw）或 `sandbox/experiments/<run_id>/`（synthetic）：

| 文件 | 内容 |
|---|---|
| `estimates.csv` | 每个 (scheme, execution, model) 一行：estimate、pooled OOF AUC、fold std、split signature |
| `paired_estimates.csv` | 每个 (scheme, execution) 一行：A−B、Nadeau-Bengio p |
| `fold_scores_all.csv` | 所有 fold 的 AUC（可重算任何统计量） |
| `reference_fold_scores.csv` | reference 的 fold AUC |
| `scheme_spread.csv`, `paired_difference.csv`, `*.png` | 报告输出的副本 |
| `config.yaml`, `meta.json`, `summary.json`, `notes.md` | ssc.tracking 标准内容（含实际生效的 config 与 `_runtime`） |

报告输出（full 运行写入 `reports/tables/` 与 `reports/figures/`；`--quick` 只写到 `<run>/report_preview/`，不会覆盖报告文件；`.md` 只含报告用的列，`.csv` 含全部列）：

| 文件 | 用在报告哪一段 |
|---|---|
| `e1_scheme_spread.{csv,md}` | **Results**：表 1 “每个方案 estimate 的 mean / std / IQR / min / max / cost” |
| `e1_paired_difference.{csv,md}` | **Results** 表 2 与 **Conclusion** 的依据：std_diff、MDD、sign agreement、p_correct_at_target |
| `e1_estimates_by_scheme.png` | **Results** 图：spread 的分布（(a) 模型 A 的 estimate，(b) A−B） |
| `e1_diff_std_by_scheme.png` | **Conclusion** 图：哪个方案的 std 低于 0.01 / 0.005 线 |
| `e1_manifest.json` | 记录这些文件来自哪个 run_id（写进报告脚注） |

两页报告建议：
- **Question**：第 1 节的问题 + 为什么 0.01 重要（LB 噪声）。
- **Protocol**：方案列表、50 个 split seed、模型 seed 固定、A/B 配对、reference、指标定义（第 2 节）。引用 sklearn 与 Nadeau & Bengio (2003)。
- **Results**：一张图（`e1_estimates_by_scheme.png` 或 `e1_diff_std_by_scheme.png`）+ 一张表（`e1_paired_difference.md` 选几列）。
- **Conclusion**：成本最低且 MDD 的 95% CI 整体低于 0.01 的方案（写明用的是哪一对 A/B）；hold-out 的 spread 是多少倍；noshuffle 的假精度；局限（见第 5 节第 1、6 条）。

## 5. Pitfalls（常见陷阱）

1. **split-seed spread 不是全部不确定性。** 它是在“这 250 行”条件下的噪声；换一批 250 行（或 private LB）带来的 data-sampling 噪声不包含在内，repeat 再多也消除不了。MDD 因此是**下限**；`frac_nb_p_lt_0.05`（Nadeau-Bengio 校正）是更保守的补充视角。
2. **shuffle=False 的假精度。** `StratifiedKFold(5)`（也是 `cross_val_score(cv=5)` 的默认）不打乱，split 只由行顺序决定；sklearn 甚至在 shuffle=False 时传 random_state 会报 ValueError。“重复运行”得到完全一样的数字，std = 0 不代表精确。若数据按某种顺序排列，fold 还会有偏。
3. **必须配对，但配对不是万能的。** A 在 split 1、B 在 split 2 上比较会把两份 split 噪声都加进差值；两者必须用同一组 splits（同一个 split signature）。配对只消除两模型**共有**的 split 噪声：std(A−B) = √(sA² + sB² − 2ρ·sA·sB)。合成数据上默认的相邻候选（top-30 vs top-50）ρ ≈ 0.6–0.85，配对把 std 降低约 40–60%（`std_diff` vs `std_diff_unpaired`）；换成 L1 vs L2 这类差异大的一对时 ρ 只有 0.2–0.5，配对只降低 10–25%，配对差的 std 甚至大于单个模型的 std。所以判断 0.01 要看**配对差**的 std，不能拿单个模型的 std 代替。
4. **hold-out 的验证集很小**：20% = 50 行（约 22 个正类），单个 AUC 的 SE 就有约 0.06。
5. **reference 也是估计值。** 看 `sign_reliable`：若 \|A−B\| < 2 SE，`sign_agree_ref` 失去意义，只看 MDD 和 `p_correct_at_target`。
6. **MDD 取决于这一对模型，不只取决于方案。** 由第 3 条的公式，MDD 随两模型 estimate 的相关性 ρ 变化：相似的两个模型（相邻超参数、相邻 k）ρ 高、MDD 小；差异大的两个模型 ρ 低、MDD 大。旧默认 L1 C=0.1 vs L2 C=0.05（合成数据上差距约 0.08，ρ 低）得到的 MDD 明显更大（5-fold x10 约 0.010–0.011，只有 5x20 低于 0.01），而默认的相邻候选（差距约 0.01）在 5x10 就已低于 0.01。所以结论必须用**代表真实决定**的一对模型得出，并在报告里写明是哪一对；真数据上换成团队自己的相邻候选再跑。
7. **fold-mean AUC vs pooled OOF AUC**：estimate 按惯例取 fold AUC 的平均；pooled OOF AUC 也存在 `estimates.csv`，两者在小 fold 上会略有不同，报告里说明用的是哪一个。
8. **naive 误差条**：`mean_within_fold_se`（fold std / √k）常被当成 CV 的误差条，但 fold 之间不独立，k-fold 方差没有无偏估计（Bengio & Grandvalet 2004）；而且它想估计的是数据抽样的不确定性，不是 split 噪声。对比它、`mean_within_rep_se` 和 seed-to-seed std 即可看出各自含义。
9. **winner's curse**：用噪声大的方案在很多模型里挑最高分，挑出的分数会偏高；E1 的结论应同时约束“多少个候选”。
10. **归档分离**：不设 `SSC_ARCHIVE_ROOT` 时，`--data raw` 会写入真实 `experiments/`、`registry/`、`reports/`。测试一律用 synthetic 或重定向 archive；`--quick` 不会写 `reports/`。
11. 换模型（如 rf / lgbm）时保持 `random_state` 固定；否则 spread 里混入模型随机性，与 brief 要求不符。

## 6. Results log（每次 full 运行后由负责人填写）

| 日期 | run_id | data | 设置变化 | 关键数字 | 结论 / 下一步 | 作者 |
|---|---|---|---|---|---|---|
| 2026-10-22 | （临时 archive，仅流程检查） | synthetic | 默认 config，`--n-jobs 8` | 见下方 “synthetic sanity check” | 流程正确；合成数据结果不是对真实问题的证据 | scaffold |
| 2026-10-22 | `20261022-064054_e1_validation_protocol` | raw | 默认 config（A/B 仍是 top-30 vs top-50，理由见下）；Colab job `J20261022-064231-pro1-rext12`（8 核 High-RAM，`--n-jobs 8`，50 s） | std(A−B)：hold-out 20% 0.0287，5-fold 0.0149，10-fold 0.0149，5x10 0.0042，5x20 0.0031；MDD：5x20 0.0061 [0.0051, 0.0076]（唯一整个 CI < 0.01），5x10 0.0083 [0.0070, 0.0104]（inconclusive）；reference 5x50 A−B = −0.0029（SE 0.0014） | 0.01 的决定（相似的线性模型，AUC ≈ 0.80）只能信 5x20（D002 成立）；单次 k-fold 的 MDD ≈ 0.03，hold-out ≈ 0.06 | ext12 (AI) |
| 2026-10-22 | `20261022-064148_e1_validation_protocol_qda_pair` | raw | 第二对：QDA reg 0.1 vs 0.3（`research/r305_e1_qda_pair.yaml`，经 `research/r301_run_extension.py`，prefix `e1_qda_`），同一 job，144 s | std(A−B)：hold-out 20% 0.0106，3-fold 0.0029，5-fold 0.0026，5x20 0.0005；MDD：3-fold 0.0057 [0.0048, 0.0072] 已 < 0.01；hold-out 0.021 / 0.016 仍 > 0.01；reference A−B = +0.0122（SE 0.0003） | 接近天花板时 split 噪声小 4.5–6 倍，3-fold 就够；所以结论依赖于这一对模型所在的 AUC 区间 | ext12 (AI) |

**真数据 A/B 的选择（2026-10-22，ext12）**：直推式 GMM 在官方 CV 上饱和（AUC 1.0，R003），不能用来回答 0.01 的问题，所以从监督模型里选相邻候选。
`research/r300_e1_pair_screen.py`（官方 5-fold x 20，每个候选一个 experiment run）按事先写好的规则（线性模型中 |差距| 最接近 0.01 的相邻一对；另取 QDA reg_param 中最接近 0.01 的一对）筛了 11 对：
corr top-30 + L2（0.7910，`20261022-063705_r300_corr30_l2`）vs top-50 + L2（0.8011，`20261022-063706_r300_corr50_l2`），配对差 −0.0101（se_split 0.0023，repeat 均值相关 0.76）；
QDA reg 0.1（0.9958，`20261022-063715_r300_qda_r01`）vs reg 0.3（0.9832，`20261022-063717_r300_qda_r03`），+0.0126。表：`reports/tables/r300_pair_screen.csv`。
**教训**：E1 的 reference（5x50，种子 900000）显示线性这一对的真实 CV 差距只有 −0.003，50 次 5x20 执行的平均是 −0.0044；官方种子给出的 −0.0101 偏离约 2 个 sd。按“差距最接近 0.01”从 11 对里挑，本身就会挑中被 split 噪声放大的一对（winner's curse，第 5 节第 9 条）。这不影响 MDD 的结论（MDD 只取决于配对差的离散度），但说明单个 5x20 的点估计也有约 ±0.006（2 sd）的 split 噪声。
两对的对照：`research/r306_e1_both_pairs.py` → `reports/tables/r306_e1_pair_comparison.md`、`reports/figures/r306_e1_pair_comparison.png`（run `20261022-064412_r306_e1_pair_comparison`）。

**synthetic sanity check（2026-10-22，脚手架验证，full 设置，默认相邻候选 top-30 vs top-50，`--n-jobs 8` 约 46 s）**
（合成数据只检验代码是否做了它声称的事，不是关于真实问题的证据）：

| scheme | std(A) | std(A−B) | std(A−B) 不配对 | MDD [95% CI] | p_correct_at_0.01 | sign_agree_ref | fits/decision |
|---|---|---|---|---|---|---|---|
| hold-out 20% | 0.0499 | 0.0290 | 0.0729 | 0.058 [0.048, 0.072] | 0.63 | 0.64 | 2 |
| hold-out 30% | 0.0371 | 0.0235 | 0.0565 | 0.047 [0.039, 0.059] | 0.66 | 0.72 | 2 |
| 3-fold | 0.0190 | 0.0113 | 0.0269 | 0.023 [0.019, 0.028] | 0.81 | 0.88 | 6 |
| 5-fold | 0.0159 | 0.0118 | 0.0234 | 0.024 [0.020, 0.030] | 0.80 | 0.84 | 10 |
| 10-fold | 0.0139 | 0.0123 | 0.0191 | 0.025 [0.021, 0.031] | 0.79 | 0.60 | 20 |
| 5-fold x10 | 0.0045 | 0.0037 | 0.0066 | 0.0074 [0.0062, 0.0092] | 0.997 | 1.00 | 100 |
| 5-fold x20（official） | 0.0027 | 0.0026 | 0.0042 | 0.0052 [0.0044, 0.0065] | 0.9999 | 1.00 | 200 |
| 5-fold no shuffle | 0（1 个 unique split） | 0 | — | 退化 | — | — | 10 |

- 与理论一致：hold-out 的 spread 约为 5x20 的 18 倍；spread 随 k 和 repeats 下降；noshuffle 50 个 seed 只有 1 个 split。单次 k-fold（3/5/10）的 MDD 都在 0.02 以上（整个 CI 高于 0.01）；5x10 与 5x20 的整个 CI 都低于 0.01。
- 单次 repeated run 内部的误差条（`mean_within_rep_se` = 各 repeat 均值的 std / √R）与真实的 seed-to-seed std 一致（A−B：5x10 0.0037 vs 0.0037；5x20 0.0027 vs 0.0026）。所以 `compare_runs.py` 的 split SE（`std_diff_repeat / √R`）就是报告的配对均值差的 split 噪声；而单个 repeat 的 `std_diff_repeat` 大约是它的 √R 倍，不能直接当成门槛。这也回答了 `docs/VALIDATION_PROTOCOL.md` §5.3：比较模型时看**配对差**的 split SE，而不是单个模型的 `auc_std_repeat`。
- reference 5x50：A−B = +0.0103（SE 0.0017）；synthetic ground truth（250 行训练，19,750 行隐藏标签）：A−B = +0.016，符号一致。10-fold 的平均差只有 +0.004（每折训练 225 行，两模型差距随训练行数变化），所以它的 sign agreement 只有 0.60。
- 旧默认（L1 C=0.1 vs L2 C=0.05，差距约 0.08、ρ 低）的同一设置下 MDD 更大：5-fold x10 约 0.011、5x20 约 0.0085。对比两组数字就是第 5 节第 6 条的证据。

结论写法提示：先写“在 MDD 的整个 95% CI 低于 0.01 的方案中成本最低的是 …（A/B = 哪一对）”，再给 hold-out 的 MDD 是 0.01 的几倍，
最后说明局限（第 5 节第 1、5、6 条）。
