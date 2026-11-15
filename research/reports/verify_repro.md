# verify_repro：可复现性与提交文件完整性核查（Verify 第 1 轮，lens = reproducibility / submission integrity）

作者：AI 核查代理 "repro"（Claude Code，Claude Opus 5.5），2026-10-22（JST）。
Colab：账号 pro1，会话 vrepro（CPU Standard，2 核；用完已 `down`）。
新代码：`research/r910_repro_rerun.py`（Colab，重建 3 个模型）、`research/r911_repro_compare.py`（本地，只读文件比对）、
`research/r912_submission_precision.py`、`research/r913_submission_precision_qda.py`（Colab，复刻数据上的提交精度检验）。
没有提交 Kaggle，没有改别人的文件；只在自己 run 的 `notes.md` 里写了 Conclusion。

## 中文摘要（给队员）

**一句话结论**：三个被核查的模型都能**逐位**复现；S001–S003 的提交文件格式正确、和登记的 sha256 一致；
包版本没有任何不一致。
- 一个看起来严重、但实际无害的问题：提交文件只保留 8 位小数，再加上 GMM 后验在 float64 里饱和，S003 文件里 77.6% 的测试行挤在两个并列值上。在 60 个复刻上量化，平均 AUC 损失为 0（第 3 点），**不用重交**。
- 真正需要修的是档案记录：47 个 Colab run 没有记录代码包；10 个早期 run（包括 S001–S003 的来源 run）连计算环境都没记录；config.yaml 也不足以单独重建模型。

**1. 重跑结果（R910 + R911）：三个模型全部逐位复现**

| 模型 | 存档 run | 重跑 run（今天的代码，新 VM） | 100 折 AUC | oof | test_pred |
|---|---|---|---|---|---|
| S003 GMM K=2 | `20261022-050708_r003_gmm2_hv` | `20261022-091211_r910_s003_gmm2_hv` | 全部相同 | 5,000 个值全部相同 | 19,750 个值全部相同（sha `71fb85e7168cde88`） |
| E3 最终模型 | `20261022-063908_e3_leakage` | `20261022-091223_r910_e3_final_k20` | honest 100 折全部相同（0.793441）；leaky 格子也相同（0.866781） | 存档没有 oof | 全部相同（sha `eadb9ff18d9c6cec`） |
| r201 QDA 块 | `20261022-071234_r201_qda_block` | `20261022-091224_r910_r201_qda_block` | 全部相同（0.999903） | 全部相同 | 全部相同（sha `0d7e9b64d238586a`）；每折选中的 reg_param、折内块、内层 AUC 也都相同 |

- 存档的 E3 和 r201 是在 8 核 VM 上跑的，这次在 2 核 VM 上跑，结果仍然逐位相同。把 BLAS 限制成 1 个线程、joblib 改成单进程再跑一遍，结果也逐位相同。所以在 Colab 上，**线程数不是误差来源，随机种子也都固定住了**。
- **但在 Windows 笔记本上不能逐位复现**（本地冒烟测试，写进 scratch 档案）：S003 的 19,750 个测试分数全部不同，最大差 4.4e-11；提交文件（8 位小数）里有 1 个值变了；E3 最大差 3.3e-16。折 AUC 仍然完全相同。原因是不同 CPU 上 OpenBLAS 选的计算核不同（Colab 是 Haswell 核），求和顺序不同。`colab/README.md` 说"本地与 Colab 的结果可以互相复现"，这句话只在分数层面成立，逐位层面不成立。

**2. 代码包核查**

- **S003**：meta.json 里没有代码包（这是 D005 之前的 run）。D005 说是 `b8534fcd0d68fc89`。我解开核对过：其中的 `research/r003_transductive_vs_supervised.py`、`src/ssc/transductive.py`、`validation.py`、`data.py`、`models.py` 和今天的文件逐字节相同；只有 `tracking.py`（HKT 时钟、没有 compute 块）、`remote.py`、`project.yaml` 不同。这和 S003 的 meta.json 对得上（created_at 是 +08:00，没有 compute 块），但没法证明 VM 上跑的就是这个包。
  - 一个小疑点：这个包里的 `remote.py` 还没有"保存代码包"的功能，所以这个文件不可能是当时那次同步自己存下来的，应该是事后补存的。文件名和内容的 sha256 一致。
- **E3**：meta.json 写的是 `unsynced (--no-sync)`，没有包 hash；notes 里说是 `b413e6aa3bae4443`，但日志里没有能证明这一点的记录。作业开始前生成的 5 个包里，E3 的 `run.py` 都是同一个版本，`src/ssc` 也和今天相同。今天的 `run.py` 只改了 null 图的标题。加上重跑逐位相同，代码问题可以视为已排除。
- **r201**：代码包 `f84e098deaf7a1bf` 里的 r201 脚本和今天相同。`lib_sup.py` 的差别只有两处：一个默认值为 `roc_auc`、实际没用到的 `scoring` 参数，以及新加的画图代码。

**3. 提交文件（S001–S003）**

格式全部正确：
- 列名是 `id,target`，19,750 行，id 和 `sample_submission.csv` 的顺序完全一致；
- 没有 NaN 或 inf，值都在 [0, 1] 内；
- sha256 前缀和 registry 一致；
- 文件里的值恰好等于 `test_pred.npy` 按 `%.8f` 格式化后的结果（最大差 5.0e-9），score_transform 为 none，与登记一致；
- cv_auc 和 run 的 summary 一致，数据指纹和今天的数据一致。

**但是**：`make_submission.py` 写文件时只保留 8 位小数。
- S003 的 `test_pred.npy` 本来就只有 7,503 个不同值，因为 GMM 的后验概率在 float64 里饱和成 0/1：最小值上并列 7,907 行，最大值上并列 3,434 行。写成文件后只剩 2,897 个不同值：最小值并列 8,918 行，最大值并列 6,416 行，文件和 test_pred 的 Spearman 只有 0.969。
- S002（QDA）文件里有 760 行恰好是 0、718 行恰好是 1。
- 换句话说，Kaggle 打分用的排序，不是复刻研究（R153，"离贝叶斯上限只差约 2e-7"）评估过的那个排序。
- 这对 AUC 有没有影响，我在复刻数据上量化了（R912 GMM、R913 QDA，各 30 个 make_classification 复刻，有已知的测试标签）：**平均没有影响**。
  - 没有标签噪声时：GMM 的 8 位小数文件与原始分数的 private AUC 在 30 个复刻里完全相同；QDA 的差小于 1e-9。
  - flip_y = 0.005（约 50 个翻转标签）时：文件 − 原始分数 = GMM +5.1e-6 ± 9.5e-6（SE）、QDA +5.1e-6 ± 2.4e-5，都不显著。单个复刻上会差到约 ±1.5e-4（GMM）或 ±3e-4（QDA）。
  - 改用不饱和的 log-odds 排序也没有好处（GMM −2.1e-6 ± 3.1e-5）。
  - 原因：被并列的都是后验离 0/1 不到 1e-8 的行；翻转标签和特征无关，排不排序期望都只得 1/2。
  - **结论：现有的 S001–S003 文件不需要重交**。"S003 离贝叶斯约 2e-7"这个说法对上传的文件也成立（期望意义上）。以后的提交可以改成写秩，消掉每个文件约 1e-4 的额外噪声，但不是必须的。

**4. make_submission 预演（只读代码 + 在临时目录模拟，没有写 submissions/ 和 registry）**

- E3 最终模型和 r201_qda_block 都能生成合法文件（`validate_file` 没有报问题；下一个编号会是 S004）。
- E3 的 summary.json 没有 `official_cv: true`，所以 make_submission 会打印"不是官方 CV"的警告。其实它用的就是官方 split（split_sig `5708ce5c3828`），这个警告可以忽略。
- r201 的文件会从 18,332 个不同值压缩成 7,724 个，最大并列 5,157 行。
- `make_submission.py` 没有 `--dry-run` 选项，只能直接写文件。

**5. 环境**

- 168 个 run 的 meta.json 里，numpy 2.4.6 / pandas 3.0.6 / scipy 1.17.1 / sklearn 1.9.1 / lightgbm 4.7.0 / xgboost 3.2.0 全部和 `requirements-lock.txt`、`colab/requirements-colab.txt` 一致，**0 个不一致**。
- 这次的 VM 上，25 个固定版本的包全部装对了。
- 110 个 run 的 env_hash = 今天 `requirements-colab.txt` 的 sha（`8871117790f585a7`）。
- Python 版本不统一，因为 Python 的补丁版本没有固定（`uv venv --python 3.11`）：CPU VM 是 3.11.17，L4 GPU 的 25 个 FM run 是 3.11.16，笔记本是 3.11.5。
- meta.json 只记录 6 个包，没有记录 joblib、threadpoolctl、BLAS 版本和计算核、CPU 型号；torch、tabpfn 的版本只写在 FM run 的 config 里。

**6. 需要主会话 / 代码维护者处理（我不能改 scripts/、src/、colab/）**

1. （可选，低优先级）提交文件改用全精度或秩：`ssc.submission.create` 用 `float_format="%.17g"`，或者直接写秩（AUC 只看排序）。R912/R913 显示期望损失为 0，所以只是消掉每个文件约 1e-4 的额外噪声。S001–S003 不用重交。
2. `colab/remote.py`：加了 `--no-sync` 时，也要读 VM 上的 `.sync_code` 标记，写进 `compute.code_bundle`。现在有 47 个 run 没有记录代码包。
3. `ssc.tracking`：config 里存 `estimator.get_params(deep=True)`；meta 里加上 threadpoolctl 信息、CPU 型号、joblib 版本。原因是现在 S003 的 config.yaml 里没有任何 GMM 超参数（n_init=5、reg_covar=1e-6、tol=1e-3、random_state=0 都是代码里的默认值）。
4. `make_submission.py` 加 `--dry-run`；在 `create()` 里断言 `ds.ids_test == sample ids`。现在之所以没出错，只是因为 test.csv 的顺序恰好和 sample 一样（已核实）。
5. 时钟：笔记本比 Colab VM 快约 100 秒（51 个作业的中位数是 −99 s）。结果是 VM 上建的 run_id 比它自己的 job_id 还早，`submitted_at_utc` 也是按笔记本时间记的。建议同步一下 Windows 时间。

## Details (English)

### Scope and method

Lens: reproducibility and submission integrity. Checks made:
1. Code bundles. I extracted the bundles to the scratchpad and diffed them file by file against today's tree.
2. Re-runs. On a fresh Colab VM (pro1/vrepro, CPU Standard, 2 cores, job `J20261022-091347-pro1-vrepro`, bundle `df6dea90c123b9f4`, env `8871117790f585a7`), `research/r910_repro_rerun.py` re-created the three models with today's code and re-ran each with the official CV (5x20, seed 2026, signature `5708ce5c3828`). Each re-run is an experiment run with `official_cv: true`.
3. Comparison. A local pure-file comparison, `research/r911_repro_compare.py` (run `20261022-091852_r911_repro_compare`), compared the re-runs with the archived runs.
4. Submission files and environment. The same r911 run checked S001-S003, a dry run of the `make_submission` mapping (temporary folder only) and the package versions.
5. Submission precision. Replica studies R912 and R913 estimated what the submission format costs in AUC.

Smoke tests and the Windows cross-platform re-run went into the scratch archive only. After the Colab job I added an `--only` option to r910; the exact code that ran is bundle `df6dea90c123b9f4`.

### A. Archived vs re-run (r911 table `A_archived_vs_rerun.csv`)

| model | what | n | bit-identical | max abs diff | Spearman |
|---|---|---|---|---|---|
| S003 gmm2_hv | fold AUC (val_id matched) | 100 | yes | 0 | - |
| | oof.npy | 5,000 | yes | 0 | 1.0 |
| | test_pred.npy | 19,750 | yes | 0 | 1.0 |
| E3 final | honest fold AUC (k_feat 20, 5-fold cell of fold_scores_grid.csv) | 100 | yes | 0 | - |
| | leaky fold AUC (same cell) | 100 | yes | 0 | - |
| | test_pred.npy (refit on 250 rows) | 19,750 | yes | 0 | 1.0 |
| | archived config `pipeline`/`selection`/`leaderboard` == fields used | - | yes | - | - |
| r201 qda_block | fold AUC | 100 | yes | 0 | - |
| | oof.npy | 5,000 | yes | 0 | 1.0 |
| | test_pred.npy | 19,750 | yes | 0 | 1.0 |
| | per-fold best reg_param / in-fold block support / best inner AUC | 100 | yes / yes / yes | 0 | - |

- BLAS-thread sensitivity (run `20261022-091403_r910_repro_env`, `thread_sensitivity.csv`): each model was run again under `threadpool_limits(1)` with `n_jobs=1`. All three are bit-identical to the default-thread re-run (0 differing values).
- Environment of that VM: OpenBLAS 0.3.30 (numpy) and 0.3.31 (scipy), both with Haswell kernels, on an "Intel Xeon @ 2.20GHz". All 25/25 pins installed at their pinned versions.
- Cross-platform: a local Windows smoke re-run (scratch archive, Python 3.11.5, 32-core Intel family 6 model 183) does **not** reproduce bit for bit.
  - S003 test_pred: all 19,750 values differ, max abs diff 4.4e-11 (relative 1.5e-10), Spearman 0.99999992 (exact float ties of the archived array are broken), 1 value of the `%.8f` file changes.
  - E3: 6,295 values differ, by at most 3.3e-16.
  - Fold AUCs are identical.
  - Cause: different OpenBLAS kernels change the floating-point summation order inside GaussianMixture (EM covariance updates). The local synthetic smoke test also showed a 1e-13 thread sensitivity of the GMM on the laptop.
- Verdict: the claim "every run can be re-created from the archive" is **confirmed on Colab**. "Local and Colab results reproduce each other" (`colab/README.md`) is **overstated**: it holds only up to ~1e-11.

### B. Code bundles

| run | meta.json `compute.code_bundle` | what I found |
|---|---|---|
| S003 `20261022-050708_r003_gmm2_hv` | absent (pre-D005; no `compute` block at all) | D005 names `b8534fcd0d68fc89` (file mtime 06:01:44 JST, before job `J20261022-060326`). The bundle's r003 script and `src/ssc/{transductive,validation,data,models,features}.py` are byte-identical to today's. `tracking.py` differs (HKT clock, no `compute_info`), which matches the run's `+08:00` created_at and missing compute block. The bundle's own `colab/remote.py` has no bundle-keeping code, so the file must have been saved later (by a newer remote.py or by hand). Its name equals its sha256. **Consistent, not provable.** |
| E3 `20261022-063908_e3_leakage` | `unsynced (--no-sync)`, env_hash empty | notes.md names `b413e6aa3bae4443` (crashed `prepare()` at 06:35); no log line records the VM's `.sync_code`. All five bundles written before the job (b413e6aa, b229bfbf, 612a81a2, 01b75ead, f8ebcc69) share E3 `run.py` md5 eb1e2452 and today's `src/ssc`. Today's run.py (995ea4d2) differs only in the null-figure title/suptitle. **Unverified which bundle; functionally irrelevant (bit-identical re-run).** |
| r201 `20261022-071234_r201_qda_block` | `f84e098deaf7a1bf` | r201 script byte-identical to today. `lib_sup.py` differs by a new `Tuned(scoring="roc_auc")` argument (same default as the hard-coded value before) and new plotting code. **Confirmed.** |

Archive-wide (r911 D summary):
- 47 Colab runs from 16 jobs recorded `unsynced (--no-sync)` with an empty env_hash: E3 (1), 13 runs of `J20261022-064602-pro1-rmix`, 8 runs of 6 other rmix jobs, and all 25 rfm runs (8 jobs, L4 GPU; these ran in separate venvs built by `research/lib_fm.py`).
- 10 pre-D005 runs (R001, R002, the 8 R003 runs including the S001-S003 sources) have no `compute` block.
- For the unsynced runs only notes.md, if anything, says which code ran.

### C. Submission files S001-S003 (r911 `B_submissions.csv`)

| check | S001 | S002 | S003 |
|---|---|---|---|
| file = `submissions/S###_<run_id>.csv`, columns `id,target`, 19,750 rows | ok | ok | ok |
| ids = sample_submission ids, same order, unique | ok | ok | ok |
| NaN / inf / outside [0,1] | 0 / 0 / no | 0 / 0 / no | 0 / 0 / no |
| `ssc.submission.validate_file` | no problems | no problems | no problems |
| sha256[:16] (file = registry) | 315d234378010a1e | 8a5107656eb6d906 | 3aae140e5837753f |
| score_transform recomputed = registry | none | none | none |
| values == `"%.8f" % test_pred` exactly | yes | yes | yes |
| max abs diff file vs test_pred | 5.0e-9 | 5.0e-9 | 5.0e-9 |
| Spearman(file, test_pred) | 1.000000 | 0.999947 | **0.968931** |
| distinct values test_pred -> file | 19,750 -> 19,743 | 19,651 -> 16,284 | **7,503 -> 2,897** |
| rows in tie groups test_pred -> file | 0 -> 14 | 113 -> 4,073 | **12,462 -> 17,144** |
| largest tie in file | 2 | 760 (value 0.0; 718 at 1.0) | **8,918 (0.00607153); 6,416 at 0.986397** |
| cv_auc_mean registry = run summary; data fingerprint = today's data | ok | ok | ok |
| pre-registered before upload (created_at < submitted_at) | 06:12:55 < 06:14:41 JST | 06:12:59 < 06:15:16 | 06:13:03 < 06:16:07 |

The raw data still match `data/raw/SHA256SUMS.txt` (train, test, sample_submission). test.csv ids are in sample_submission order: `create()` relies on this without asserting it.

**Precision finding.**
- Two things coarsen the ordering of S003:
  - Float64 saturation: `GaussianMixture.predict_proba` underflows to exact 0/1, so 7,907 + 3,434 test rows already tie in `test_pred.npy`.
  - `float_format="%.8f"` in `ssc.submission.create`: this merges another ~4,700 rows into the two extreme groups.
- Kaggle therefore scores a file in which 77.6% of the rows sit in two tie groups.
- The R153 statement that "gmm2_hv is Bayes-optimal within 3e-7" was computed on raw float scores, not on the uploaded file.
- Replica quantification. Both runs use 30 replicas per noise level and score the private subset (19,355 rows). The four orderings are the same fitted model's float64 score (`raw`), its `%.8f` file (`csv8`), the unsaturated log-odds, and the true Bayes log-odds.
  - R912, run `20261022-092445_r912_submission_precision`: make_classification, class_sep 0.71, hamming 4, S003's GMM. The replicas reproduce the real tie structure: on average 8,450 / 8,560 rows in the two extreme file groups, Spearman(file, raw) 0.974 (real S003: 8,918 / 6,416, 0.969).
  - R913, run `20261022-093023_r913_submission_precision_qda`: in-fold HighVarColumns + QDA, reg 0.003 and 0.1.

| model | flip_y | AUC raw | csv8 − raw (mean ± SE; range) | log-odds − raw (mean ± SE) |
|---|---|---|---|---|
| GMM (S003) | 0 | 0.999703 | 0 exactly in 30/30 | 0 exactly in 30/30 |
| GMM (S003) | 0.005 | 0.997201 | +5.1e-6 ± 9.5e-6; −1.3e-4 .. +1.5e-4 | −2.1e-6 ± 3.1e-5 |
| QDA reg 0.003 (r201-like) | 0 | 0.999293 | −1.8e-10 ± 1.8e-10 | 0 exactly |
| QDA reg 0.003 | 0.005 | 0.996719 | +5.1e-6 ± 2.4e-5; −3.2e-4 .. +3.1e-4 | −1.9e-5 ± 1.7e-5 |
| QDA reg 0.1 (S002-like) | 0 | 0.994426 | +8.9e-10 ± 5.8e-10 | 0 exactly |
| QDA reg 0.1 | 0.005 | 0.991682 | +3.6e-6 ± 4.2e-6 | −5.1e-7 ± 3.8e-7 |

  - All differences have |t| < 1.6. The format costs nothing in expectation. It adds per-file noise of at most ~1.5e-4 (GMM) / ~3e-4 (QDA) on the private AUC, an order of magnitude below the public-LB SD (0.002–0.004).
  - Mechanism: only rows whose posterior is within ~5e-9 of 0 or 1 are merged. Without noise their labels agree with the posterior. With noise the flipped labels are independent of x and earn 1/2 in expectation whether ordered or tied.
  - Verdict: **harmless; no resubmission needed**. Writing ranks (or `%.17g`) would remove the extra per-file noise, but this is optional.
- Side observation (not this lens; for the gen verifier). On these generator-level replicas (a new random make_classification per seed, not R153's parametric replicas of the real data), S003's GMM is within 1e-6 of Bayes in 27/30 seeds but misses by 6.3e-4, 7.5e-4 and 7.0e-3 in seeds 19, 28 and 14 (flip 0; highvar block 15, 11 and 17 columns; 11,659–19,041 distinct scores). The "within 3e-7" statement is therefore conditional on replicas built from the real data's fitted structure. It is not a property of every make_classification draw.

### D. make_submission dry run (r911 `C_make_submission_dryrun.csv`; code read, nothing written to submissions/ or the registry)

- I ran `--help` only; the script has no dry-run mode. `create()` does `to_unit_interval` -> sample ids -> `to_csv(float_format="%.8f")` -> `validate_file` -> registry row. I reproduced these steps into a temporary folder.

| | E3 final `20261022-063908_e3_leakage` | r201 `20261022-071234_r201_qda_block` |
|---|---|---|
| status / has test_pred / 19,750 finite values | done / yes / yes | done / yes / yes |
| score_transform | none (0.00132 .. 0.99572) | none (0.0 .. 1.0) |
| validate_file | no problems | no problems |
| would be | `submissions/S004_20261022-063908_e3_leakage.csv` (sha16 d47d3f3909b1eba4 in the dry run) | `submissions/S004_20261022-071234_r201_qda_block.csv` (3be09c0723286457) |
| `summary.official_cv` | **missing** -> make_submission prints the "not validated on the official CV" banner although split_signature = 5708ce5c3828 | true |
| distinct values test_pred -> file; largest tie | 19,750 -> 19,747; 2 | 18,332 -> 7,724; **5,157** |
| registry cv_auc_mean would be | 0.79344 (honest CV; test_pred is the refit on 250 rows) | 0.99990 |

- The sub_id is assigned at creation time, so S004 is only correct if nothing is created first.
- `DAILY_LIMIT`/`daily_submission_limit` is 5 (the Kaggle rule). The team rule of at most 3 uploads per day is not enforced; make_submission only warns at 5.

### E. Environment (r911 `D_run_packages.csv`, `D_pins.csv`; r910 `pinned_vs_installed.csv`)

| item | finding |
|---|---|
| meta.json packages of all 168 runs (numpy, pandas, scipy, sklearn, lightgbm, xgboost) | 0 mismatches against requirements-lock.txt and colab/requirements-colab.txt (2.4.6 / 3.0.6 / 1.17.1 / 1.9.1 / 4.7.0 / 3.2.0) |
| lock vs colab pins | identical for all 24 shared packages; colab uses `xgboost-cpu==3.2.0` for lock `xgboost==3.2.0` (same code, documented) |
| local .venv vs lock | identical for all checked packages (incl. joblib 1.6.0, threadpoolctl 3.7.0) |
| VM installed vs pins (r910) | 25/25 match |
| Python | 3.11.17 (142 runs, CPU VMs), 3.11.16 (25 runs, L4 GPU runtime, FM track), 3.11.5 (laptop); patch level not pinned (`uv venv --python 3.11`) |
| env_hash | `8871117790f585a7` (= today's sha256 of requirements-colab.txt) on 110 runs; empty on the 47 unsynced runs; absent on the 10 pre-D005 runs |
| not recorded anywhere | joblib/threadpoolctl versions, BLAS library and kernel, CPU model; torch 2.14.1 / tabpfn / tabicl only in FM configs |

### F. Other integrity observations

1. Clock skew. Over 51 Colab jobs the earliest run_id of a job precedes its job_id by a median of 99 s (IQR −101 to −94 s). The job_id comes from the laptop clock, the run_id from the VM clock. Either the laptop runs ~100 s fast or the VM runs slow; Colab VMs are normally NTP-synced (not verified). Consequences:
   - local and Colab run_ids interleave out of order by ~1.5 min;
   - `submitted_at_utc` is recorded with the laptop clock, which matters only within ~100 s of the 09:00 JST reset.
2. The R003 run_ids and the S001-S003 file names use HKT (+08:00); already documented in D006, not renamed (append-only).
3. The archived `config.yaml` of research runs is not a recipe:
   - S003's config holds only a description and `model_code`; n_init, reg_covar, tol, random_state and alpha are the constructor defaults of `TransductiveGMM` at run time;
   - r201's config does not hold the inner-CV settings (5x2, seed 0) or the HighVarColumns ratio 1.5.
   I re-created the models from the scripts; a silent change of a default would not be detected by "re-run from config".
4. The registry author is `TODO_ID_1` on every row (see verify_rules).

### Verdicts on the claims under this lens

| claim | verdict | evidence |
|---|---|---|
| S003, the E3 final model and r201_qda_block can be re-created from the archive and give the archived numbers | confirmed (bit-identical on Colab) | r910 runs; r911 `A_archived_vs_rerun.csv` |
| The code bundles contain the code that produced the runs | confirmed for r201; consistent but not provable for S003 (b8534fcd) and E3 (b413e6aa, --no-sync) | section B |
| S001-S003 files are valid and equal the runs' test_pred | confirmed, with the precision caveat: equal to 8 decimals, ordering coarsened (S003 Spearman 0.969) | section C |
| make_submission would produce valid files for E3 / r201 | confirmed (spurious non-official warning for E3) | section D |
| Colab and local package versions are identical | confirmed for the pinned packages; Python patch level and BLAS kernel differ | section E |
| Local and Colab results reproduce each other | overstated: ~1e-11 for the GMM, 1 file value changes | section A |
| S003's "~2e-7 below Bayes" also holds for the uploaded (8-decimal, tied) file | confirmed in expectation (csv8 − raw = +5e-6 ± 1e-5 with noise, exactly 0 without); single-file noise ≤ ~1.5e-4 | R912 `20261022-092445_r912_submission_precision` |
| Supervised QDA files (S002 / r201) lose nothing by saturation + %.8f | confirmed in expectation (\|t\| < 1.6) | R913 `20261022-093023_r913_submission_precision_qda` |
