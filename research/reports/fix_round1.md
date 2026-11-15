# fix_round1：第 1 轮更正的终检（更正 1–29）

- 作者：AI 终检代理（Claude Code，Claude Opus 5.5），2026-10-22 20:30–21:30 JST。
- 输入：`research/reports/verify.md`；结构化汇总 `verify_synthesis.json`（字段 `corrections`，1–29 号）；4 个更正代理（fix_ext12、fix_ext34、fix_registry、fix_reports）和代码代理（fix_code）的结果；3 个第 2 轮研究代理（r2_inductive、r2_noise、r2_audit）的报告。
- 本文的编号一律是 synthesis 的 1–29。`verify.md` §4 的编号是 1–30，顺序不同，对照见英文表的第 2 列。
- 我只写了本文件。没有改其他文件，没有用 Colab，没有碰 Kaggle，没有删文件。临时文件都在 scratchpad：快照解压在 `D:\tmp\claude\D--polyu-class-ml\c9dcd3b2-a841-4b45-b8a8-1aabc1690c51\scratchpad\check_snapshot`，排版测量在 `D:\tmp\claude\D--polyu-class-ml\c9dcd3b2-a841-4b45-b8a8-1aabc1690c51\scratchpad\check_pages`。

## 中文摘要（给队友）

**一句话**：29 条更正里，25 条已完成，3 条部分完成（5、17、22），1 条按规则没有执行（29：文件不能由 AI 删，改用"不进代码包"缓解）。1–4 号里的决定都还要团队确认。只追加规则整体守住了：快照里的 162 个 run 文件夹逐字节没变，3 个 registry CSV 只在末尾追加，决定正文没有被改，只改了 Status 行。`archive_check.py` 0 个问题、4 个已知 warning；`pytest` 214 个全部通过。四份扩展草稿在约定的排版（XeLaTeX，Calibri 11 pt，A4，2.54 cm 页边距，单倍行距，段后 4 pt，图宽 60–70%）下都在 2 页以内，但余量很小；按 Word 默认段落间距（行距 1.08、段后 8 pt）排，E1、E2、E4 会超出（2.00–2.17 页）。三份 r2 报告的数字都能在档案里找到，我抽查了约 200 个，只发现几处末位舍入和两处措辞问题。

**已经修好的**
- **registry**：B006–B008 已登记，B004 没动（1）。D011 取代 D009 的措辞和证据（2）。D010 写下了最终选择规则，包括"不反推种子和隐藏标签"（3）。D012 补充了 D005 的代码来源和 D006 的时钟说明（4）。`ai_usage.md` 的错误句子已换成如实的披露段落，表格修好了，并加了 6 行（5）。`references.md` 加了 39 条，另有 9 条改为 used（6）。日志追加了一条更正记录（24、28）。
- **第 1 轮研究报告**：gen、mix、sup、fm、ext12、ext34 都在开头加了 Errata，并在原处改了句子（7–12）。三份核查报告 verify_leak、verify_rules、verify_cand 也是这样处理的（25–27）。
- **扩展草稿**：E1–E4 的数字都按档案改正了（13–16），并且都有了可见的 AI 说明、可见的引用和 References，篇幅也压到了 2 页以内（17）。四个标题保持不变。
- **代码**：`colab/README.md` 的复现说法已改正（18）。`--no-sync` 现在会记录 VM 上的代码包和环境 hash，meta.json 里新增了 BLAS 线程池、CPU 型号和估计器参数（19）。E3 的 `run.py` 会写入 `official_cv`（20）。`make_submission.py` 新增了 `--dry-run`、test id 检查和每天 3 次的上限（21）。新增 S003 的 config、`transductive_gmm` builder 和 README 里的复现路径（22），还新增了 `tests/test_transductive.py`（23）。

**档案规则检查**
- 快照 `backups/snapshot_20261022-084506.zip`（08:45 JST）里的 162 个 run 文件夹逐字节相同，`notes.md` 也一样。之后的 51 个 run 里，18:10 以前建立的 36 个，在 18:10 之后没有任何文件被改动。
- `registry/experiments.csv`、`compute_jobs.csv`、`submissions.csv`：快照的内容是现在文件的逐字节前缀。S001–S003 的提交文件没变。
- `decisions.md` 只改了 D005、D006、D009 的 Status 行；`beliefs.md` 只有追加；`ideas.md` 只改了 Status、Evidence、Reason 三列，这是该文件自己允许的。
- `journal.md` 第 54 行有一处旧条目的改动：快照文件名从 084443 改成了 084506。这是主会话在 08:45 打快照后立刻改的（会话记录里有 08:45:15 JST 的那条 sed 命令），不是本轮所为，属于同一分钟内改自己写的条目。以后这类改动请改用追加。
- `ai_usage.md` 改了 3 行旧记录的 Member 和 Verified-by 两格，并换掉了那句错误的草稿句。这两处改动是更正 5 明确要求的，旧句子原文保留在 :23 的更正说明里。
- fix_ext34 把 `research/reports/ext34.md` 和 `reports/extensions/e3_leakage.md` 的换行符从 LF 改成了 CRLF。这只影响外观，但第一次 commit 时这两个文件会显示为整份改动。
- 还没有任何 commit（D004），所以 git 给不出 diff。上面的检查都靠快照和各代理保存的改前副本。

**页数**（详见英文 §5）
- E1：1.94 / 1.96 / 1.97 页（图宽 60 / 65 / 70%）。
- E2：表格列宽按内容定时 1.89–1.97 页；按 verify_ext 的等宽、不可跨页表格是 2.04–2.07 页。
- E3：1.74 页（列宽按内容）或 1.91 页（等宽列，单元格折行）。E3 已经没有图。
- E4：1.79–1.94 页。
- 按 Word 默认段落间距排，会超出 2 页：E1 2.08–2.13、E2 2.08–2.17、E4 2.00–2.08；E3 是 1.96，单元格折行时 2.15。

**还没做的（谁来做）**
- **队员**：
  - 填四份草稿的作者行、`ai_usage.md` 的 Member 和 Verified by、披露段落里的 `[TODO]`。
  - 同步 Windows 时钟（D012）。
  - 确认 D007、D009–D012。
  - 存档比赛规则原文和比赛页面（public 395 行），并找讲师确认能否使用测试集特征。
  - 设置 git 身份后做第一次 commit（D004）。
  - 决定要不要删 `research/r935_r933_pooled.npz`、`r403_r402_pooled.npz`、`r424_r420_pooled.npz`（AI 不删文件）。
- **主会话**：
  - 上传 S005 和 S006，并按 r2_noise §1.5 登记 B006 和 B007 的 outcome（B006 的点估计偏高 1–3e-4）。
  - 新写一条决定（D013），内容包括：(a) D010 第 3 种情形具体化为"扩展网格 QDA（r930）+ r201"（r2_inductive）；(b) D009/D011 的证据 (1)、(2) 改为已复核：R441，以及 r404 的 0.804 vs 0.678（222 个复刻）；(c) 记录 fix_code 的代码改动，即 D005 的扩展：research 数据不进代码包，meta.json 新增字段。另外，D012 第 1 条说 `validation.py`、`models.py` 与 S003 的代码包逐字节相同，这一点现在已经不成立。
  - 在 `ai_usage.md` 里给 fix_ext12、fix_ext34、fix_reports、fix_code 和三个 r2 代理补行，并删掉 :21 那句已经过时的话（现在已经有 `tests/test_transductive.py`）。
  - 写一条日志，记录本轮更正和第 2 轮研究。
- **扩展负责人**：
  - E2 第 60 行的 "203 of 19,750 with a posterior of 0.1–0.9" 要改：按收敛后验算是 189 行，203 是用 S003 自己的分数算的（r2_noise §3.3）。
  - E3 的 `extensions/e3_leakage/README.md:181` 和 run 文件夹 notes.md 的 Conclusion 还写着"待提交"和"0.073 ± 0.014"。
  - E4 的 4 联图字太小；E2 的图例仍写着 "90% bootstrap band"，说明文字已经改了。
  - 导出时用单倍行距、段后不超过 6 pt、表格列宽自动适应，或者每份再删约 60–120 词，留出余量。
  - 统一四份草稿的 AI 说明：E1、E2 写的是 "checked against the archive"，最好像 E3、E4 那样写明检查者是另一轮 AI 核查。
- **研究报告的维护者**（主会话或下一轮更正）：
  - 按 r2_audit §5 改 sup.md（"no method exceeded 0.88"、"about 80 labels"、GPC 0.978 和 0.976）、fm.md（"best linear model"、0.9989、99.6%）、gen.md（"1 − R² ≈ 4e-14"）。
  - fm.md 的边界行结果现在可以写"已在 222 个复刻上确认"（r404）。
  - ext12.md 英文 "Text for the report"（:92）没有重写，引用时以 Errata 为准。
  - 可选：verify_leak.md:51、:147 和 verify_rules.md:87、:410。
- **代码维护者**：
  - `scripts/kaggle_submit.py` 仍只按 Kaggle 的 5 次限制停止，没用团队的 3 次。
  - `configs/_template.yaml` 的模型清单要加 `transductive_gmm`；ARCHIVE_RULES §2 要写进 meta.json 的新字段。
  - `research/r440_archive_pack.json.gz`（658 kB，不是 npz）仍然会进代码包。
  - 用 Colab 在真实数据上跑一次 `configs/final_s003_gmm2_hv.yaml`，确认 sha256 前缀 3aae140e5837753f。

---

## English

### 1. Method

- **Corrections.** For each correction I opened the target file(s) and located the new text by line. For numbers I recomputed the value from the cited archive table (`reports/tables/*.csv`, `experiments/<run_id>/summary.json`) with read-only scripts. Full-precision checks: E1 table vs `e1_paired_difference.csv` / `e1_qda_paired_difference.csv`; E2 table vs `e2_learning_curve_table.csv` and `r942_e2_coverage.csv`; E3 table vs `e3_grid.csv`, `e3_null.csv`, `r321_e3_proxy.csv`, `e3_leaderboard.csv`, `r944_e3_replica_decomp.csv`; E4 vs `e4_usefulness.csv` and `e4_paired.csv`; gen/mix vs `r920` summary.json, `r921_summary.csv`, `r922_posterior_by_prior.csv` and `r922_private_by_flip.csv`.
- **Append-only.**
  - I unzipped `backups/snapshot_20261022-084506.zip` (1,890 files) and compared it byte for byte with the archive.
  - For files edited after 18:10 JST (the start of the fix round), I used file modification times and the fixers' pre-edit copies (`scratchpad\fix_ext12\orig`, `scratchpad\fix_ext34\orig_*`, `scratchpad\fix_code\orig`). Each of these copies is byte-identical to the snapshot.
  - git cannot help: "your current branch 'main' does not have any commits yet".
- **Tests.** I ran `.venv\Scripts\python.exe scripts\archive_check.py` and `.venv\Scripts\python.exe -m pytest -q`. Neither wrote to the archive: the mtime of `registry/experiments.csv` stayed at 19:55:32.
- **Pages.** I wrote my own Markdown-to-XeLaTeX converter (`scratchpad\check_pages\md2tex_check.py`). It does not reuse either fixer's converter. Details are in §5.
- **r2 reports.** I read all three in full and recomputed about 200 of their numbers from the cited tables (§6).

### 2. Correction table

Status: **applied** / **partially** / **not applied** / **not needed**. "v#" = item number in `verify.md` §4.

| # | v# | File(s) | Status | Evidence (file:line) | Remaining / note |
|---|---|---|---|---|---|
| 1 | 1 | `registry/beliefs.md` | applied (main session) | B006 :21, B007 :22, B008 :23 (S003 private median ≈ 0.997, 90% 0.992–0.9996, P(≥ 0.999) ≈ 0.15). B004 :19 unchanged: the snapshot diff shows only appended lines 20–23 | B006's point 0.9988 is 1.2–2.8e-4 high (r2_noise §1.5: 0.99868 replicas, 0.99852 real features). Do not edit B006; use these values when recording its outcome |
| 2 | 2 | `registry/decisions.md` D009 | applied (superseding D011) | D011 :119–130 has the CV-resolution wording and cites `r936_private.csv`; D009 Status :99 "superseded by D011"; D009 body unchanged (snapshot diff: line 99 only) | Now outdated by round 2: evidence (1) is confirmed by R441 (r2_audit), evidence (2) by r404 on 222 replicas, 0.804 vs 0.678 (r2_inductive). r254's 0.825 / 0.702 is a mean over 3 replicas, not 6 (fix_reports). r2_audit gives the interval as ±1.6–5.1e-4. Needs a new decision (main session) |
| 3 | 3 | `registry/decisions.md` (new) | applied (main session) | D010 :105–117; default S003 + r201; switches; never FM or S004; item 6 :116 no seed or label recovery | Team confirmation TODO. r2_inductive proposes ext-grid QDA (r930) + r201 for switch 3; the D010 body cannot be edited, so this needs a new decision |
| 4 | 4 | `registry/decisions.md` D005, D006 | applied | D012 :132–141; D005 Status :62 and D006 Status :72 "accepted (amended by D012)"; bodies unchanged | A member must still sync the Windows clock. D012 item 1 says `src/ssc/{validation,models}.py` equal the S003 bundle; since fix_code (20:11) they no longer do. Record this in the new decision |
| 5 | 6 | `registry/ai_usage.md` | partially | Disclosure paragraph :19 (verify_rules §3 text with `[TODO]` clauses); correction note :23 quotes the old false sentence; table repaired; rows :7–15 (5 proposed rows + fix_registry); Operator filled as "AI (Claude Code) …; member: TODO" | Member and Verified-by still TODO (members only). :21 "no test imports `ssc.transductive`" is now stale (`tests/test_transductive.py`, fix_code). No rows yet for fix_ext12, fix_ext34, fix_reports, fix_code or the three r2 agents |
| 6 | 7 | `registry/references.md` | applied | 63 BibTeX keys (24 + 39); new sub-sections :66–89, :165–234; model-card and licence rows :240–242; 9 statuses set to used (e.g. efron1993 :108, ledoit2004 :137, figueroa2012 :157); dempster1977 :168, friedman1989 :175, fan2008 :177 | kohavi1995, bengio2004, ambroise2002 and meinshausen2010 stay candidate until each extension is submitted, as instructed. fix_ext12's note that dempster1977, friedman1989 and fan2008 are missing is stale |
| 7 | 8, 9 | `research/reports/gen.md` | applied | Errata :7–14; summary table :30–31; :42, :45; Text for the report :60, :62; Table G1 :71–72 | Values match `r922_posterior_by_prior.csv` (uniform_0_0.1: median 0.99733, q05 0.99230, q95 0.99959), `r921_summary.csv` (regret_eauc 2.57e-7, SE 1.8e-9, frac_pos 1.0) and r920 (MLE 0.703, LR 95% [0.639, 0.760]). Open (r2_audit item 6): "R² ≥ 1 − 4e-14" → "1 − R² ≈ 4e-14" |
| 8 | 10 | `research/reports/mix.md` | applied | Errata :6–14; :17, :25, :36 | Uses 0.76% (archived q95 0.01525 / 2), not the synthesis's 0.77%. States both cases: two orders of magnitude with clean labels, three to four at plausible noise. Checked against `r922_private_by_flip.csv` (sd 1.1e-5 at 0; 3.1e-4 at 0.003; 5.5e-4 at 0.01) |
| 9 | 11 | `research/reports/sup.md` | applied | Errata :10–17; :51, :61, :113, :115 (D009 wording; reg 0.001 = grid floor, r930 1e-4 / 3e-4; tn1 −6.3e-4; "0.9995 after model selection, r205"; r209 `*_r003` not nested-honest) | New, from r2_audit §5: "no method exceeded 0.88 on all 150 features" (:113) is refuted (TabICL v2 0.993); "about 80 labels" (:115) should read 80–120 (R441); GPC-ARD 0.978 vs 0.976. These are outside correction 9 |
| 10 | 12 | `research/reports/fm.md` | applied | Errata :5–12; :50, :65, :71, :83, :96–98 (prior-generated data; arXiv:2602.11139; CV fold-mean log-loss 0.0094 vs OOF 0.0071; boundary result marked unverified) | Boundary result is now confirmed on 222 replicas (r404: 0.814 vs 0.686 on r254's definition), so "unverified" can become "confirmed (r404)". Open from r2_audit: F21 "best linear model" → "on all 150 features"; F13 0.9990 → 0.9989; F27 99.5% → 99.6% |
| 11 | 13 | `research/reports/ext12.md:49` | applied | Errata :6–24 (11 items); old :49 corrected in place at :68 (R942: 21% coverage, bias −0.031; quote ~0.91–0.92 ± 0.04) | The English "Text for the report" (:92) still has the stale E2/E1 claims. The errata covers them, but anyone copying from that section must follow the errata |
| 12 | 14 | `research/reports/ext34.md:12, :51` | applied | Errata (勘误) :5–18; :26 (gap 0.073, SD 0.014, corrected SE 0.020, p = 0.0003); :31, :68 (S004 0.79136); :103 | File changed from LF to CRLF (cosmetic) |
| 13 | 15 | `reports/extensions/e3_leakage.md` | applied | Table :44–55, recomputed in full from `e3_grid.csv`, `e3_null.csv` and `r321_e3_proxy.csv` (all 90 cells match); null leaky 0.61–0.71 :59; LB :67–69; gap with SD and corrected SE :73–74; 61 / 48 features and plateau :84–85; R944 sentences :87–92, :96; model choice :98–99 | Uses LB SE 0.0233 (`e3_leaderboard.csv` public_lb_se_approx, at AUC 0.79), so the text says 3.2 and 1.2 SE where the synthesis had 3.6 and 1.4 (SE 0.021 at 0.82); the fixer's choice is the archived value. Proxy − public = +0.028 (0.8195 − 0.79136). `e3_leaderboard.csv` already holds S004. Still stale outside the draft: `extensions/e3_leakage/README.md:181` and the run's notes.md Conclusion (E3 owner) |
| 14 | 16 | `reports/extensions/e1_validation_protocol.md` | applied | Hold-out 30% CI [0.035, 0.053] :36; MDD = 2 sd ≈ 50% power, 80% needs 2.8 sd :29; pair ≈ 0.004 apart, SE 0.0019, 1.84 sd, normal-shift extrapolation :48; 4.5–6.1×, 80%-power MDD 0.0086 [0.0071, 0.0107], NB at most 1 of 50, data-sampling sd 0.0105 :56; visible references :58 | Table 1 re-checked against the CSVs: every MDD, CI and p_correct matches at the printed precision |
| 15 | 17 | `reports/extensions/e2_learning_curve.md` | applied | 0.695 :40; 0.785 :42; caption with nominal 90%, envelope [0.995, 1] and log-linear 1.008 :46; R942 bullet :49; caption bootstrap median 0.904 :54; replica-calibrated sentence :58; r321 boundary rows :60; "positive in every replicate" deleted; back-test argument dropped | The figure image still says "90% bootstrap band" (caption fixed). New, from r2_noise §3.3: the "203 … with a posterior of 0.1–0.9" at :60 uses S003's compressed score; the converged posterior gives 189 (`r422_real_boundary_counts.csv`) |
| 16 | 18 | `reports/extensions/e4_feature_stability.md` | applied | ≥ 98.5%, k = 10/13/20, perm m* 7/8/9/17, 1 of 50 null curves (p ≈ 0.02–0.04) :51–55; ±0.018 / ±0.005, perm-rule row 0.797 ± 0.015 :59–61; +0.063 (p 0.038), +0.028 (p 0.30), stability selection −0.033 to +0.002 :69–73; "never in the top-10 / freq ≥ 0.5 set" with perm exceptions :88–90; "above the best linear model in R300 (0.898)" :92–93; old Table 1 merged into the figure caption | All values match `e4_usefulness.csv` and `e4_paired.csv`. The 4-panel figure has about 3–4 pt text (fix_ext34) |
| 17 | 19 | all four drafts | partially | Visible AI line (E1/E2 :5, E3/E4 :5–6); visible References (E1 :58, E2 :62, E3 :102–105, E4 :98–101); prose cut; trial exports done | Author lines still "TODO". Fits 2 pages only with single spacing and content-fitted tables (§5); with Word's default paragraph spacing E1, E2 and E4 exceed 2 pages. The AI line wording differs between E1/E2 and E3/E4 |
| 18 | 20 | `colab/README.md` | applied | :19–24 (bit-identical Colab-to-Colab, R910; laptop about 1e-11, max 4.4e-11, 1 CSV value; OpenBLAS build) | – |
| 19 | 21 | `colab/remote.py`, `src/ssc/tracking.py` | applied (one documented deviation) | remote.py: `vm_markers` :351, `--no-sync` path :456, compute fields :602–605. tracking.py: joblib / threadpoolctl in PACKAGES :56, `cpu_model` :112, `threadpools` :134, meta :247, `record_estimator` / `save_cv(estimator)` :263–270, `estimator_params` :378. validation.py :185 keeps the template estimator | Estimator parameters go to `meta.json["estimator"]`, not `config.yaml`, because config.yaml must stay the exact input config (ARCHIVE_RULES §1.2). I agree with this. ARCHIVE_RULES §2 does not yet list the new meta fields |
| 20 | 22 | `extensions/e3_leakage/run.py` | applied | `official_signature` :113–126; `official_cv` / `official_split_signature` in summary.json :560–563 | Only future runs get the field. `20261022-063908_e3_leakage/summary.json` is unchanged (append-only) |
| 21 | 23 | `scripts/make_submission.py`, `src/ssc/submission.py` | applied | `TEAM_DAILY_LIMIT = 3`, `team_daily_limit` :40–52; `check_test_ids` :55–68; `create(dry_run=)` :128–174; CLI `--dry-run` / `--force-limit` :50–51; 6 new tests | `scripts/kaggle_submit.py` still stops only at Kaggle's 5 (code maintainers) |
| 22 | 24 | `README.md:104` | partially | Route :97–116; `configs/final_s003_gmm2_hv.yaml`; builder `transductive_gmm` in `src/ssc/models.py` :46–56, :86, :90; test that the config builds the r003 estimator bit-identically (synthetic data); read-only dry run of the archived S003 run printed 3aae140e5837753f (fix_code; I did not re-run it) | No real-data run of the config yet (Colab job, code maintainers). A "clean checkout" test is impossible until the first commit (D004) |
| 23 | 25 | `tests/` | applied | `tests/test_transductive.py` (9 tests); pytest 214 passed (§4) | Update `ai_usage.md:21` (see 5) |
| 24 | 26 | r300_block run folders | applied | `registry/journal.md:75` records the correction and group B; both run folders are byte-identical to the snapshot | The core report must count both runs as group B (report authors) |
| 25 | 28 | `research/reports/verify_leak.md` §4.4 | applied | Errata :8–12; :58 (Chinese), :154 (English): 0.0032 is conditional split noise; corrected SE 0.020, t 3.72, p 0.0003 | – |
| 26 | 29 | `research/reports/verify_rules.md` | applied | Errata :9–15; r210 re-run (r932, R900) :493; pages 2.16–2.71 and cuts :92, :129, :414 | The "150–250 words" phrase does not occur in the file (not needed); the correct cuts were added |
| 27 | 30 | `research/reports/verify_cand.md` §2 | applied | Errata :6–10; :25, :91 (0.9977 under R156's truncated grid; 0.9968 [0.9923, 0.9996] under a flat prior) | – |
| 28 | 5 | `registry/journal.md` | applied | New entry :58–87; correction :74 (flip_y support 0–0.015, MLE 0.003; S003 private ≈ 0.997) | – |
| 29 | 27 | `research/r935_r933_pooled.npz` | not applied (deletion not allowed for AI); mitigated | `colab/remote.py:66–67, :304–311, :344–348`: `research/*.npz|*.npy` > 1 MB now travel once as "research data", not in every code bundle (13.5 MB → 1.28 MB, fix_code) | Team decision to delete r935 (3.5 MB), r403 (5.7 MB) and r424 (3.0 MB) once r936/r404/r425/r426 are frozen. `research/r440_archive_pack.json.gz` (658 kB) still rides in the code bundle |

### 3. Append-only audit

| Area | Check | Result |
|---|---|---|
| `experiments/` (162 run folders in the snapshot) | every file byte-compared with the snapshot | identical, including notes.md; no file missing, none added |
| `experiments/` (51 later folders) | latest mtime of non-notes files vs run_id time; any change after 18:10 | non-notes files written 0–42 min after each run_id time (run plus Colab merge). None of the 36 runs created before 18:10 has a file changed after 18:10. The 15 new r2 runs carry their own Conclusions |
| `registry/experiments.csv`, `compute_jobs.csv`, `submissions.csv` | snapshot bytes are a prefix of the current file | yes for all three (69,806 → 93,525; 15,019 → 24,363; 1,233 → 2,347 bytes) |
| `submissions/` | S001–S003 byte-compared | identical; S004–S006 new |
| `decisions.md` | snapshot diff | only the Status lines of D005 :62, D006 :72 and D009 :99 changed; D010–D012 appended. D010 itself was written after the snapshot (main session), so it cannot be byte-checked; there is no sign of an edit |
| `beliefs.md` | snapshot diff | append only (:20–23) |
| `ideas.md` | snapshot diff | Status / Evidence / Reason of I001 and I003–I008 only (allowed by ideas.md :3); Idea text unchanged |
| `references.md` | snapshot diff | 16 `% status:` comment lines changed; new sub-sections inserted (:66–89, :165–234, :240–242); no BibTeX field changed |
| `journal.md` | snapshot diff | one appended entry (:58–87). Line 54 also differs (084443 → 084506). This edit is not from this round: the main session made it at 08:45:15 JST on 2026-10-22 with `sed`, right after the snapshot was taken (session transcript). It is a same-minute fix of its own entry, but strictly it breaks the append-only rule |
| `ai_usage.md` | snapshot diff | Member and Verified-by cells of the 3 old rows filled, and the false draft sentence replaced, both as correction 5 required; the old sentence is quoted at :23; What-for and Files cells unchanged |
| Line endings | byte count | fix_ext34 converted `research/reports/ext34.md` and `reports/extensions/e3_leakage.md` from LF to CRLF |
| Other files changed after 18:10 | full tree scan outside experiments / logs / tables / figures | only the files listed by the five fixers and three research agents, plus remote.py merges into `registry/*.csv`; nothing else |

### 4. archive_check and pytest

- `scripts\archive_check.py`: `runs=213, runs_without_conclusion=0, registry_rows=213, code_bundles_referenced=45, jobs=89, submissions=6, report_docs=20`. Result: **0 problems, 4 warnings**. All 4 are known: 3 failed runs whose reason is in notes.md (`20261022-065046_r152_calibration`, `20261022-071440_r201_qda_all`, `20261022-091033_r900_perm_null_a`), and the `r20x_candidates.csv` wildcard in sup.md.
- `python -m pytest -q`: **214 passed** in 35.8 s (195 before the fix round plus 19 new). The project's `addopts = "-q"` makes `-q` suppress the summary line; I read the count with `-rA`.

### 5. Page counts (XeLaTeX, Calibri 11 pt, A4, 2.54 cm margins)

Method:
- My own converter, `scratchpad\check_pages\md2tex_check.py`. HTML comments are dropped, the `##` headings become `\section*`, tables are set in `\small`, and the figure is set at the given share of the text width.
- Length = (pages − 1) + `\pagetotal` / `\pagegoal` on the last page, as in verify_ext.
- "plain" spacing = single line spacing with 4 pt between paragraphs (verify_ext's setting). "Word" spacing = line spacing 1.08 with 8 pt between paragraphs (Word's Normal style).
- Table layouts:
  - *equal*: tabularx with equal columns that cannot break across pages (verify_ext's exact method).
  - *dash*: columns sized by the separator dashes, as pandoc does; the table can break.
  - *content*: columns sized by the longest data cell, ragged-right; the table can break. This is closest to Word's autofit.
- Results: `scratchpad\check_pages\results_final.tsv` and `results_content.tsv`.

| Draft | Figure width | plain: content | plain: dash | plain: equal | Word: content | Word: dash | Word: equal |
|---|---|---|---|---|---|---|---|
| E1 | 60 / 65 / 70 % | 1.94 / 1.96 / 1.97 | 1.94 / 1.96 / 1.97 | 1.94 / 1.96 / 1.97 | 2.08 / 2.13 / 2.13 | 2.08 / 2.13 / 2.13 | 2.08 / 2.13 / 2.13 |
| E1 | 100 % | 2.07 | 2.07 | 2.07 | 2.22 | 2.22 | 2.22 |
| E2 | 60 / 65 / 70 % | 1.94 / 1.96 / 1.97 | 1.89 / 1.91 / 1.93 | 2.04 / 2.04 / 2.07 | 2.13 / 2.13 / 2.17 | 2.08 / 2.13 / 2.13 | 2.45 / 2.49 / 2.49 |
| E2 | 100 % | 2.07 | 2.04 | 2.16 | 2.27 | 2.25 | 2.89 |
| E3 (no figure) | – | 1.74 | 1.91 | 1.91 | 1.96 | 2.15 | 2.50 |
| E4 | 60 / 65 / 70 % | 1.79 / 1.81 / 1.81 | 1.84 / 1.88 / 1.88 | 1.94 / 1.94 / 1.94 | 2.00 / 2.00 / 2.04 | 2.06 / 2.06 / 2.08 | 2.04 / 2.04 / 2.04 |
| E4 | 100 % | 1.86 | 1.94 | 1.94 | 2.06 | 2.11 | 2.13 |

Before the fix round, verify_ext measured 2.16–2.71 pages.

Reading:
- With plain spacing and tables that can break, all four fit 2 pages at a figure width of 60–70%. This agrees with the fixers: fix_ext12 measured E1 1.94 / 1.96 / 1.97 and E2 1.96 / 1.97 / 1.99; fix_ext34 measured E3 1.74 and E4 1.89.
- With verify_ext's unbreakable equal-width tables, E2 stays over 2 pages (2.04–2.07), as fix_ext12 also reported. Its table cells wrap at equal widths.
- E3 at 1.91 vs 1.74 shows that its 9-column table must have content-fitted columns. Equal widths make each row wrap onto two lines.
- The margins are thin. Under Word's default paragraph spacing, E1, E2 and E4 run to 2.00–2.17 pages, and E3 to 1.96, or 2.15 if its table cells wrap.
- So the owners should do one of the following:
  - export with single spacing, at most 6 pt after paragraphs, autofit tables and a figure of at most 70% of the text width;
  - or cut about 100 words in E1 and E2 and about 50 in E4 for a safety margin.
- Missing glyphs: none reported in any log.

### 6. Traceability of the round-2 research reports

All three reports cite a run_id or table for their numbers. I recomputed about 200 values from the cited files.

- **`research/reports/r2_inductive.md`** (about 70 values). Sources checked:
  - `reports/tables/r404_private.csv`, `r404_pairs.csv`, `r404_by_level.csv`, `r404_boundary.csv`, `r404_public_abc.csv`, `r404_eb_kappa.csv`, `r404_reg_choice.csv`, `r404_weights.csv`, `r404_real_check.csv` and `r401_eb_kappa_frequency.csv`;
  - the tn1 rows of `r936_private.csv` and `r936_pairs.csv` (ext −4.72e-4; ext + r201 −1.45e-4);
  - the summary.json files of the 7 candidate runs (CV 0.99990, 0.99998, 0.99903, 0.99919 (24), 0.98631, 0.99981 (9));
  - `20261022-185320_r401_inductive_official_checks/summary.json` (1.1e-16, 1.3e-15).

  Everything matches except two last-digit roundings in Table A2: "(1.7e-6)" is 1.65e-6 in the table (rounds to 1.6e-6), and "[+2.05e-4]" is 2.045e-4 (rounds to 2.04e-4). Neither changes a conclusion.
- **`research/reports/r2_noise.md`** (about 90 values). Sources checked: `r425_levels.csv`, `r425_private.csv`, `r425_public_pred.csv`, `r425_check.csv`, `r425_r936_repro.csv`, `r421_public_pred.csv`, `r421_window_probs.csv`, `r421_support_and_private.csv`, `r426_designs.csv`, `r426_per_level.csv`, `r422_population.csv`, `r422_replica_summary.csv`, `r422_replicas.csv` (refit − true −0.57 ± 3.81) and `r422_real_boundary_counts.csv`. Everything matches at the printed precision, with these exceptions:
  - Four 5th-digit roundings, none material: §1.2 f = 0.015 "0.99239" (archived 0.992385); §2.2 k = 1, stratified 197, "2.63e-3" (2.6246e-3); §1.5 stratified-198 r209 upper "0.99939" (0.999385); default-aware r209 "0.99912" (0.999115).
  - §1.5 (R426) says "−5.1e-4 to −6.1e-4 at every level from 0.001 to 0.03". At 0.03 the value is −4.85e-4 (`r426_per_level.csv`).
  - R426's row "r936 design" gives 0.99882, while §1.1's exact reproduction of r936 gives 0.99884 (`r425_r936_repro.csv`). Both are archived, but the text should say that they use different level weights (posterior mean f 0.00421 vs 0.00437).
  - §3.3's S003 score range [0.0061, 0.9864] with 6,416 ties cites no table. I recomputed it from `submissions/S003_20261022-050708_r003_gmm2_hv.csv`: min 0.00607153, max 0.986397, 6,416 rows at the maximum, 204 rows in (0.1, 0.9). It is also in `verify_repro.md` §C (R912). Add the citation.
- **`research/reports/r2_audit.md`**. Sources checked:
  - `r440_core_numbers.csv`: 242 claims, 235 yes and 7 no; the per-section counts equal the report's table, and the 7 failures are exactly the ones listed;
  - `r440_label_budget_compare.csv`: 37 comparable cells, max |z| 1.984 for QDA-LW in-fold at 10 labels, 0.656 vs 0.624;
  - `r440_label_budget_first_hit.csv`;
  - `r441_label_budget.csv`: r201 at 40 labels undefined in 371 of 400 draws; at 80 labels mean 0.9842, median 0.9984, q10 0.975; 0.9993 at 120;
  - `r440_journey.csv`: surprises −0.72 / +0.55 / +0.14 / −1.35.

  All match. Caveat: the S005 and S006 rows of the journey table show r936's predictions (0.99884, 0.99913). r2_noise has since refined the S005 prediction to 0.99868 (replicas) and 0.99852 (real features). Refresh the table after the uploads, as r2_audit says.

Cross-report consistency:
- r2_inductive and r2_noise agree on S003's private AUC under the flat prior: 0.99680 (r404, binned weights) vs 0.99683 (R425).
- r2_audit's "±1.6–5.1e-4" refines the "±2–5e-4" that D011 has just adopted. A future decision should use the refined range.

### 7. Open items by owner

| Owner | Item | Source |
|---|---|---|
| Members | Fill author lines (E1–E4), Member / Verified by and the `[TODO]` clauses of the disclosure | corrections 5, 17 |
| Members | Sync the Windows clock; confirm D007 and D009–D012; archive the rules text and the competition page (public size 395); ask the instructor about test-feature use | D012, D010, r2_audit D5 |
| Members | First git commit after setting identities (D004); decide on deleting the pooled npz files | correction 29 |
| Main session | Upload S005 and S006 and record B006 and B007 against r2_noise §1.5 | r2_noise |
| Main session | New decision D013: (a) concrete switch 3 (ext-grid QDA r930 + r201); (b) D009/D011 evidence (1) and (2) confirmed by R441 and r404, interval ±1.6–5.1e-4; (c) code changes of fix_code, which extend D005 (research data channel, new meta.json fields), noting that D012 item 1's "byte-identical" no longer holds for `validation.py` and `models.py` | r2_inductive, r2_audit, fix_code |
| Main session | `ai_usage.md`: rows for fix_ext12, fix_ext34, fix_reports, fix_code and the three r2 agents; replace :21; journal entry for this round and round 2 | corrections 5, 23 |
| Extension owners | E2 :60 boundary-row wording (189 converged posterior; 203 is S003's score); E3 README :181 and run Conclusion; E2 figure legend "nominal"; E4 figure legibility; export settings or about 50–100 more words cut; one AI-line wording for all four | §5, r2_noise, fix_ext12, fix_ext34 |
| Research-report maintainers | sup.md (0.88, 80–120 labels, GPC 0.976 / 0.978), fm.md (best linear on all features, 0.9989, 99.6%, "confirmed (r404)"), gen.md (1 − R² ≈ 4e-14); optional verify_leak.md :51, :147 and verify_rules.md :87, :410; ext12.md English section :92 | r2_audit §5, r2_inductive §6, fix_reports |
| Code maintainers | `kaggle_submit.py` team limit 3; `configs/_template.yaml` adds `transductive_gmm`; ARCHIVE_RULES §2 lists the new meta fields; one Colab run of `configs/final_s003_gmm2_hv.yaml` on real data (expect AUC 1.0 in 100/100 folds, signature 5708ce5c3828, sha256 prefix 3aae140e5837753f); a clear error in `ssc.transductive` when no high-variance column is found (this needs a decision, since the file is part of S003's provenance) | fix_code |

### 8. Files

- Written: `research/reports/fix_round1.md` (this file).
- Scratch only:
  - `D:\tmp\claude\D--polyu-class-ml\c9dcd3b2-a841-4b45-b8a8-1aabc1690c51\scratchpad\check_snapshot` (unzipped snapshot);
  - `D:\tmp\claude\D--polyu-class-ml\c9dcd3b2-a841-4b45-b8a8-1aabc1690c51\scratchpad\check_pages`: converter `md2tex_check.py`, runners `run_all.sh`, `run_e12.sh` and `run_content.sh`, results `results_final.tsv` and `results_content.tsv`, PDFs under `out`, `out2` and `out3`.
