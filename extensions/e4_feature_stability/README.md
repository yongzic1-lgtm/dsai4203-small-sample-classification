# E4 特征排序的稳定性 (Stability of the feature ranking)

脚本：`extensions/e4_feature_stability/run.py`，参数：`extensions/e4_feature_stability/config.yaml`。
报告部分 (<= 2 页，英文，标题写 owner 姓名) 结构固定为 **Question / Protocol / Results / Conclusion**，见第 4 节。

---

## 1. 问题 (Question)

在大量 bootstrap resample（250 行有放回抽 250 行，至少几百次）上按 |corr(feature, target)| 给 150 个特征排序，**这一步不拟合任何模型**。记录每个特征进入 top-k 的频率：有些几乎每次都在，有些约为 chance rate k/150。

1. 把 150 个特征的 selection frequency 从高到低画出来，对 2-3 个 k 重复；
2. 频率是否在某处**急剧下降 (sharp drop)**？下降位置是否依赖 k？
3. 下降点以上的特征是不是"有用"的：只用它们拟合模型 vs 用全部 150 个；
4. 这说明"被稳定选中 (stably selected)"和"值得保留 (worth keeping)"之间是什么关系？

---

## 2. 协议清单 (Protocol checklist)

| Brief 要求 | run.py 中的实现 | config.yaml 参数 |
|---|---|---|
| bootstrap：250 行有放回抽 250 行，至少几百次 | `ssc.features.bootstrap_topk_frequency(X, y, k, n_boot, random_state=seed)`；所有 k 用**同一组** resample（同一 seed）；另用同一 RNG 约定计算完整的 \|corr\| 矩阵 (B x 150) 得到 rank 分布，并验证两者给出的频率完全一致 (`ssc_matches_local_k*`) | `bootstrap.n_boot: 1000`, `bootstrap.seed: 2026` |
| 只按 \|corr\| 排序，不拟合模型 | `ssc.features.abs_correlation` + `top_k_indices`（并列取较小列号） | `bootstrap.method: pearson` |
| 记录每个特征进入 top-k 的频率 | `e4_frequency.csv`：150 个特征的 `freq_k5/k10/k20`、全数据 \|corr\| 与 rank、bootstrap rank 的均值和 5%/95% 分位数 | `bootstrap.ks: [5, 10, 20]` |
| sanity：频率之和 = k | 每个 resample 恰好选 k 个 -> Σ_j freq_j = k (`freq_sum_equals_k*`) | – |
| 画 150 个特征按频率降序的曲线，2-3 个 k | `e4_frequency_sorted.png`（2026-11-01 起是一行四个 panel、全文宽、rank 轴取 log、300 dpi；每个 k 一个 panel：观测曲线、chance line k/150、二项分布上界、置换 label 的 null band、m*(k) 竖线；synthetic 上标出真实特征）+ `e4_frequency_overlay.png`（所有 k 叠在一张图，log x） | – |
| 客观地检测 sharp drop | 四个规则，`e4_drop.csv`：**gap** = 相邻排序频率的最大差；**half** = freq >= 0.5；**binom** = 超过 Bin(B, k/150)/B 的 Bonferroni 95% 上界；**perm** = 观测曲线在置换 label null 曲线的 95% 分位数之上的前导长度。m*(k) = primary rule | `drop.*`，`drop.primary_rule: perm` |
| 下降位置是否依赖 k | 比较各 k 的 m*(k)；`e4_overlap.csv`（各规则下 stable set 在不同 k 之间的交集、Jaccard、包含关系），`e4_stable_sets.csv`（具体特征列表） | – |
| 检验下降点以上的特征是否有用：只用它们 vs 全部 150 | repeated CV，**完全相同**的 splits（`make_splits` 只调用一次，所有变体 split signature 一致，否则报错），`paired_compare` 与 `all150` 配对比较 -> `e4_usefulness.csv`, `e4_paired.csv`, `e4_usefulness.png` | `cv`（= project.yaml `official_cv`: 5-fold x 20, seed 2026），`pipeline` |
| (a) naive / leaky | `naive_stable`：stable set 在全部 250 行上选出，再 `FixedColumns` 做 CV（有偏，同 E3） | `usefulness.threshold: 0.5` |
| (b) honest | `honest_stable`：`StabilityTopK(k, n_boot_fold, threshold)` 放进 Pipeline，每个 fold 只用 training part 重做 bootstrap 选择；fold 内选了哪些特征另存 `infold_selection.csv` | `usefulness.n_boot_fold: 200`, `usefulness.fold_seed: 0` |
| 区分 "stable" 和 "top by correlation" | 同样数量 m = \|全数据 stable set\| 的普通 top-m by \|corr\|：`naive_corr_top_m`（全数据选，FixedColumns）和 `honest_corr_top_m`（fold 内 `CorrelationTopK(m)`）；配对比较 `honest_stable vs honest_corr_top_m` | – |
| (c) **下降点以上的集合本身** | brief 问的是 "features above that point"：`naive_drop` = primary rule 的 m*(k) 集合（全数据选，FixedColumns，有偏）；`honest_drop_top_m` = fold 内 `CorrelationTopK(m*(k))`（m*(k) 来自全数据，只泄漏"大小"）。两者都与 all150 配对比较，另有 `naive_drop vs honest_drop_top_m`。集合为空时（下降点以上没有特征）退回 top-1 特征并在 `drop_set_fallback_top1` 标记 | `drop.primary_rule` |
| synthetic 上用 truth.csv 做 sanity check，真实数据上不依赖 truth.csv | 若 `data/<source>/truth.csv` 存在：每个规则的 precision / recall、true vs noise 的 feature-level AUC、fold 内选择的 precision / recall；不存在时这些列和检查自动跳过 | – |

说明：`*_stable` 变体用 **threshold 规则 (freq >= 0.5)**，因为 `StabilityTopK` 实现的正是这个规则，可以原样放进每个 fold；"下降点以上的集合"（primary rule）由 `*_drop*` 变体直接检验，所以两者不一致的 k 也有答案（synthetic quick：k = 5 时 threshold 集合 m = 4，下降点以上 m* = 7，Jaccard 0.571）。`e4_usefulness.csv` 的 `stable_set_size_all_rows`、`drop_set_size_all_rows`、`jaccard_stable_vs_drop_set` 记录两个集合的大小与重合度。

---

## 3. 运行方式 (How to run)

在项目根目录运行：

```powershell
# 1) 冒烟测试 (synthetic，写入 sandbox/)，约 6-10 s
.venv\Scripts\python.exe extensions\e4_feature_stability\run.py --data synthetic --quick

# 2) 完整设置 (synthetic)：检查代码 + 估计耗时
.venv\Scripts\python.exe extensions\e4_feature_stability\run.py --data synthetic

# 3) 正式研究运行 (真实数据，写入真实 archive)
.venv\Scripts\python.exe extensions\e4_feature_stability\run.py --author <你的 id>
```

- `--quick`：B = 200，k = {5, 10}，3 次置换，fold 内 50 个 resample，3 个 split seed。数字**不能**进报告（会打印 "cv differs from official_cv"，属正常）；报告用表图只写到 `<run>/report_preview/`，不覆盖 `reports/tables|figures`。run 仍会进入所选档案的 registry，所以 **`--quick` 只和 `--data synthetic` 一起用**。
- `--data` 不写时取 `$SSC_DATA`，再不写就是 raw（与 `scripts/` 一致）。
- 完整设置实测耗时（本机，`--n-jobs 1`，synthetic）：bootstrap 0.9 s + null 5.3 s + usefulness 36.5 s，总计约 **46 s**（含 import 约 52 s）。usefulness 的主要开销是 fold 内的 `StabilityTopK`（3 个 k x 100 个 fold x 200 个 resample，诊断时再算一遍）。
- `--n-jobs`：对 logreg 无益（进程开销大于收益），换重模型时再开；结果与 `n_jobs` 无关。
- `--replot RUN_ID`（2026-11-01 加）：只用该 run 文件夹里已存的表重画报告图（`e4_frequency_sorted.png` 等），不拟合模型、不建新 run、不写 registry；`--replot-out DIR` 写到别处预览。例：`run.py --replot 20261022-071414_e4_feature_stability`。
- 开发测试时设置 `SSC_ARCHIVE_ROOT` 到私人临时目录；**正式运行不要设**。

---

## 4. 输出 (Outputs) 与报告结构的对应

**运行目录** `experiments/<run_id>/`（原始结果，不可修改）：

| 文件 | 内容 |
|---|---|
| `config.yaml`, `meta.json`, `notes.md`, `summary.json` | 配置、环境、假设、汇总（m*(k)、sanity、耗时） |
| `frequency.csv` | 150 个特征的频率表（同 `e4_frequency.csv`） |
| `boot_abs_corr.npy` | B x 150 的 bootstrap \|corr\| 矩阵 (float32)，可离线算任意 k 的频率 |
| `null_sorted_curves.csv` | 每次置换、每个 k 的排序频率曲线（150 个位置） |
| `drop.csv`, `overlap.csv`, `stable_sets.csv` | 下降点、跨 k 重合度、各规则下的 stable set |
| `usefulness_fold_scores.csv`, `usefulness.csv`, `paired.csv` | 每个 (k, variant, repeat, fold) 的 AUC、汇总、配对比较 |
| `infold_selection.csv` | honest 变体每个 fold 选中的特征数、与全数据集合的 Jaccard、precision / recall (synthetic) |
| `*.png` | 三张图的副本 |

**报告用输出** (`reports/tables/e4_*.csv`, `reports/figures/e4_*.png`，每次非 quick 运行覆盖；`e4_manifest.json` 记录来自哪个 run_id)：

| 报告部分 | 用什么 |
|---|---|
| **Question** | 第 1 节 |
| **Protocol** | 第 2 节清单写成一段话：B = 1000、k = 5/10/20、只用 \|corr\|、四个 drop 规则（说明 primary 为什么选 perm）、usefulness 的七个变体（含下降点以上集合的 naive / honest 版本）、official CV (5-fold x 20) + identical splits + paired corrected t-test、模型；引用 Efron & Tibshirani (1993) bootstrap、Meinshausen & Bühlmann (2010) stability selection、Nadeau & Bengio (2003)、Ambroise & McLachlan (2002)（naive 版本的选择偏差） |
| **Results**（必须有表或图） | 主图 `e4_frequency_sorted.png`；表：`e4_drop.csv` 的 m_gap / m_half / m_perm（+ overlap 一句话）；`e4_usefulness.png` 或 `e4_usefulness.csv` + `e4_paired.csv` 的关键行（各变体 vs all150、naive vs honest、stable vs top-m） |
| **Conclusion** | 有没有 sharp drop、m*(k) 是否随 k 变；stable 特征是否有用（以 honest 结果为准）；naive 高估多少；"stable" 是否带来超出 top-\|corr\| 的信息；回答 stably selected 与 worth keeping 的关系（第 5 节） |

---

## 5. 解读要点（报告 Conclusion 的推理依据）

1. **bootstrap 频率本质上是"这个样本里 |corr| 排名"的函数。** resample 围绕观测样本波动，一个特征进入 top-k 的频率主要取决于它的观测 |r| 比第 k 名高出多少个标准误 (约 1/sqrt(250) ≈ 0.063)。所以 stability 不是独立于 |corr| 的新信息：synthetic 上 stable set 与同样大小的 top-m \|corr\| 集合完全相同（`jaccard_stable_vs_top_m = 1`，naive 两个变体的 AUC 一模一样）。
2. **chance rate k/150 只是平均值，不是 noise feature 的零分布。** 纯 noise 的样本 |r| 也有大有小，观测 |r| 偏大的 noise feature 在 bootstrap 里同样"稳定"。置换 label 的 null band 显示：没有任何信号时，排名第一的 noise feature 的频率 95% 分位数在 k = 5 / 10 / 20 时分别约为 0.59 / 0.74 / 0.86。因此 binomial 规则（把每次选择当作概率 k/150 的独立试验）严重偏宽：synthetic 上 m_binom = 7 / 21 / 34，后两者大部分是 noise（precision 0.43 / 0.26）。perm 规则才是"比噪声更稳定"的正确参照。
3. **下降位置是否随 k 移动是关键诊断。** 若存在一组强特征（数量 m），当 k >= m 时它们几乎每次都在，剩下 k - m 个名额由大量 noise feature 分享，曲线在 m 处断崖式下降，m* 不随 k 变；当 k < m 时 top-k 名额在强特征之间争夺，频率被摊薄，gap 规则会把这组特征切开。synthetic（10 个真实特征，其中 7 个样本 |r| 明显大）：m_perm = 7 / 7 / 7，m_half = 4 / 7 / 7，m_gap = 2 / 7 / 7。若真实数据上 m* 大致跟着 k 走（m* ≈ k），说明没有天然分组，"drop" 只是 top-k 截断本身造成的。
4. **"有用"只能用 honest 结果判断。** naive 版本在全部 250 行上选 stable set，validation rows 参与了选择，和 E3 的 leaky 一样偏高：synthetic 上 k = 10 时 naive_stable 0.916，honest_stable 0.883，差 +0.033；k = 5 时差 +0.057 (p = 0.02)。
5. **fold 内的 stability selection 更保守。** 每个 fold 只有 200 行，频率整体变低，阈值 0.5 选出的特征更少（k = 10：平均 6.2 个 vs 全数据 7 个），且 k = 20 时 fold 内 precision 只有 0.82（阈值 0.5 对 k = 20 太低，见第 2 点）。synthetic 上 honest_stable 始终略差于同样大小的 honest_corr_top_m（-0.012 到 -0.020，不显著）：stability 没有带来额外收益。
6. **stably selected 与 worth keeping 的关系**：强的真实特征两者兼有；但 stable ≠ useful（观测 |r| 偏大的 noise feature 也会稳定，见 null band），useful 也不一定 stable（synthetic 中特征 86 的真实权重 0.93，但这个样本里 |r| 只有 0.025，排名 110，几乎从不入选）。stability 衡量的是**在这个样本内**排名的可重复性，usefulness 是**样本外**的预测贡献，只能用 honest CV（以及 LB）来回答。是否删掉其他特征有利，还取决于模型：L2 logreg 在全部 150 个特征上被 noise 拖累 (0.794)，而 L1 / 强正则模型自带选择，差距可能小得多。

---

## 6. 常见陷阱 (Pitfalls)

1. **naive 结果不能当作"stable 特征有用"的证据**，报告里必须并列 honest 结果并说明差异来源（E3）。
2. **阈值应随 k 调整**：null 下 top noise feature 的频率随 k 增大（约 0.59 -> 0.86），固定阈值 0.5 在大 k 时会放进 noise。可在 `usefulness.threshold` 试 0.7 等，但每次改动都要在结果记录里写明（避免挑阈值造成新的选择偏差）。
3. **gap 规则在 k 小于强特征组大小时会切开该组**；报告 m*(k) 时把四个规则都列出来，不要只挑一个好看的。
4. **binomial 规则的零假设是错的**（见第 5 节第 2 点），只作为反例展示。
5. **honest_corr_top_m 的 m 来自全部 250 行的 stable set 大小，honest_drop_top_m 的 m*(k) 也来自全部 250 行**：这只是一个超参数（大小）的轻微信息泄漏，对结论影响可忽略，但应在 Protocol 中写一句。
6. **identical splits**：所有变体共用 official CV 的 splits（`split_sig` 列）；不要单独改 `cv.seed`。split seed 间的 SD 不是 standard error（20 个 repeat 共用同一批行），用 corrected t-test。
7. **模型选择影响 usefulness 的结论**：默认是 L2 logreg (C = 0.1)。用团队最终模型时，改 `pipeline`（不能含 selector，脚本会报错）；sklearn 1.9 中用 `l1_ratio` 而不是 `penalty`；固定 `random_state`。
8. **bootstrap 重复行**：每个 resample 约有 37% 的原始行缺席、部分行重复，这是 bootstrap 的正常行为；fold 内 StabilityTopK 只在 training part 上重抽样。
9. **synthetic 只是代码检查**：真实数据可能没有清晰的强特征组，结论以真实数据运行为准；`reports/tables` 会被每次运行覆盖，务必记录 run_id。

---

## 7. 参考：synthetic 完整运行的预期输出（用于核对脚本是否正常）

`--data synthetic`（完整设置，B = 1000）：

| k | m_gap | m_half | m_binom | m_perm | precision / recall (perm) |
|---|---|---|---|---|---|
| 5 | 2 | 4 | 7 | 7 | 1.00 / 0.70 |
| 10 | 7 | 7 | 21 | 7 | 1.00 / 0.70 |
| 20 | 7 | 7 | 34 | 7 | 1.00 / 0.70 |

Usefulness（5-fold x 20，k = 10）：all150 0.794；naive_stable 0.916 (7 个特征)；honest_stable 0.883 (fold 内平均 6.2 个)；honest_corr_top_m 0.902 (7 个)。honest_stable - all150 = +0.088 (corrected p = 0.002)；naive - honest = +0.033。九个 sanity check 全部 PASS。

---

## 8. 结果记录 (Results log)

每次**完整**运行（真实数据）后由 owner 追加一行；结论也写进 `experiments/<run_id>/notes.md` 的 Conclusion 部分。

| 日期 | run_id | data | 配置改动（相对 config.yaml） | m*(k)（gap / half / perm，各 k） | usefulness 关键数字（honest_stable vs all150；naive - honest；stable vs top-m） | 结论 | author |
|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | | raw | 无 | | | | |
| 2026-10-22 | 20261022-071414_e4_feature_stability | raw | config.yaml 已改（ext34，正式 run 前）：B 1000 -> 2000；ks 加 13（无标签高方差块大小）；null 置换 20 -> 50（envelope 更稳）；n_boot_fold 200 -> 500（fold 内选择噪声更小）；threshold 0.5、模型 (L2 logreg C=0.1) 不变。Colab job J20261022-071554-pro1-rext34 | k=5/10/13/20：gap 7/8/8/8，half 7/8/8/9，perm 7/8/9/17（binom 7/17/22/28）；7 个强特征 (98,17,100,112,80,77,46) 在 k>=10 时频率 >= 0.986 | honest_stable(k=10) vs all150 +0.063 (p 0.038)；naive-honest -0.001 (k=10) / +0.034 (k=5) / naive_drop-honest_drop +0.069 (k=20)；stable vs top-m 从不显著更好；stable 集 = top-m |corr| (Jaccard 1)；R320 (20261022-074624)：块 > stable（L2 0.862 vs 0.832，QDA 0.996 vs 0.961，GMM 1.000 vs 0.983） | 有 sharp drop（7-8 个特征，k>=8 时位置不变）；stable 相对 150 有用，但不是值得保留的集合：6 个块特征从不 stable 却靠方差携带信号；17 稳定但是块的线性组合（冗余） | ext34 (AI) |

**run.py 修改记录（ext34, 2026-10-22，正式 run 前）**：`SERIES` 加第 4 个颜色并按 `len(SERIES)` 取模——4 个 k 时 overlay 图里 top-5 与 top-20 原来同色。

**与无标签高方差块的比较**（不在 run.py 内，见 `research/r320_e4_stable_vs_block.py`，run 20261022-074624_r320_e4_stable_vs_block；proxy test 见 `research/r321_e3e4_proxy_test.py`，run 20261022-072942_r321_e3e4_proxy_test）。
