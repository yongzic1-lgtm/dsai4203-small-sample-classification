# notebooks/：只用于探索

## 命名

`NN_author_topic.ipynb`：NN 是两位序号，全队递增；author 是 `project.yaml` 中的成员 id；topic 用小写加下划线。例：`01_ab_eda.ipynb`、`02_cd_corr_noise_level.ipynb`。

## 规则

1. **notebook 只做探索**：看数据、试想法、画草图。notebook 的结果不是档案，不能直接进报告。
2. **要进报告的东西必须由脚本复现并归档**：写成 config，用 `scripts\run_experiment.py`（或 `extensions\<e#>\run.py`）在 clean commit 上跑，结果进 `experiments/` 和 `registry/`。报告用的图由脚本生成到 `reports/figures/`（前缀 `core_` 或 `e#_`）。
3. **不要从 notebook 写真档案**：不在 notebook 里创建 `ssc.tracking.Run`，也不调用 `ssc.submission.create` / `record`（meta.json 记录的命令行和 git 状态会失去意义）。**读**档案没问题：
   ```python
   from ssc import data
   from ssc.tracking import load_run, read_registry
   from ssc import paths
   ds = data.load("raw")                       # ds.X, ds.y, ds.X_test
   reg = read_registry(paths.experiments_csv("raw"))
   run = load_run("<run_id>", "raw")           # config, meta, summary, fold_scores, test_pred
   ```
4. **Kernel**：选择 **Python (ssc)**（即 `.venv`）。启动方式：`.venv\Scripts\python.exe -m jupyterlab`（不要用 `jupyter lab`，原因见 `docs/SETUP.md` §1 Jupyter）。
5. **commit 前清空输出**，避免把数据片段和大段输出带进 git：
   ```powershell
   .venv\Scripts\python.exe -m nbconvert --clear-output --inplace notebooks\01_ab_eda.ipynb
   ```
6. notebook 里得到的结论和想法，记到 `registry/journal.md` 和 `registry/ideas.md`。
7. 不发布公开 Kaggle notebook，不把 notebook 分享给队外的人（规则 5）。
