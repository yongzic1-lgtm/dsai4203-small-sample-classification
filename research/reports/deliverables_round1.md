# deliverables_round1：本轮交付物终检（2026-11-01）

- 作者：AI 终检代理（Claude Code，Claude Opus 5.5；工作流代理 final_check），2026-11-01 12:09–12:40 JST。
- 输入：本轮各代理的结果（core、builder、assembled、fixed、video、ext_figures、reflect_contrib、rules_docs）；快照 `backups/snapshot_20261022-213717.zip`（2026-10-22 21:37 JST，2,685 个文件）；项目里的当前文件。
- 我只写了本文件。没有改其他文件，没有用 Colab，没有碰 Kaggle，没有删文件，没有 commit。`build_report.py` 用 `--out` 和 `--tmp-dir` 写到 scratch，所以 `reports/report.pdf` 没被覆盖；pytest 加了 `-p no:cacheprovider`，并设了 `PYTHONDONTWRITEBYTECODE=1`。检查前后各做了一次项目的 SHA-256 清单（6,041 个文件，不含 `.git`、`.venv`），两次完全相同。
- 临时文件（快照解压、对比脚本和输出、试排的 PDF）：`D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\final_check`。

## 中文摘要（给队友）

**一句话**：AI 能做的部分基本做完了。核心报告、四份扩展、贡献表、`reports/report.pdf`（15 页，每部分都在页数限制内）和视频材料都已备好，只差只有队员能填的内容（队名、姓名、分工、讲师的答复、每人核查了什么），以及只有队员能做的事（Canvas、合队、录视频、反思、勾选最终提交、git commit）。本轮守住了只追加规则。

**四个命令的结果**
- `archive_check.py`：0 个问题，4 个 warning，都是 10-22 就有的（3 个 failed run，原因写在各自的 notes.md；`sup.md` 引用的 `r20x_candidates.csv` 是通配写法，不是文件）。214 个 run 对应 registry 的 214 行，6 个提交。
- `pytest -q`：245 个测试全部通过，用时 30 s。
- `build_report.py`：共 15 页。core 6 页（实际用了 5.77），E1 2（1.79），E2 2（1.74），E3 2（1.64），E4 2（1.85），贡献表 1（0.90）。有两个 warning：`project.yaml` 的 `report.extensions_owned` 还是空的，所以四个扩展都排了进去；还剩 TODO（core 8 处，每个扩展 2 处，贡献表 43 处）。现有的 `reports/report.pdf`（12:03 生成）和我 12:10 用当前文件重排的结果，文字逐行相同，所以它是最新的。
- `package_deliverables.py`（dry run）：缺队名，缺视频；report PDF、代码（180 个文件）、提交记录（6 个都有 public 分数，0 个标为 final）都 OK。另外查了一遍：要打进 zip 的文件里，只有 `project.yaml` 的 `compute.accounts` 里有两个个人 Gmail 地址。`--build` 之前要删掉（清单 E）。

**各交付物**（细节见英文 §2）
- 核心报告、E1–E4、贡献表、`report.pdf`、视频材料：**就绪，只差队员的字段**。
- 代码复现：S003 的复现路线已在 Colab 上验证过；第二个默认最终提交 S005 的复现命令还没写进 README；仓库还没有任何 git commit。
- 反思、Kaggle 最终选择、Canvas checkpoint、队伍 zip：**还没开始**。这些只有队员能做；最终选择按计划在 11-23 至 11-25 做。

**只追加规则**：本轮守住了。
- `experiments/` 的 2,015 个文件与快照逐个相同（大小和 CRC32），没有新增、删除或修改的 run。`registry/experiments.csv`、`compute_jobs.csv`、`submissions/`、`logs/` 也与快照相同。
- `decisions.md`：本轮只在末尾追加了 D014，并改了 D007、D010 的 Status 行，其余没动。
- 本轮改过的图表都由原来生成它们的脚本重新生成：E2、E4 的 9 张图由各自的 `run.py --replot` 根据已归档的 run 表重画，PNG 里记着来源 run；新的 `core_*` 由 `submission_history.py` 和 `make_core_figures.py` 写出，`core_manifest.json` 里的输入哈希与当前文件一致。
- 快照之后、本轮之前（10-22 22:23，上一个会话）还有 4 处改动。`submissions.csv` 给 S005、S006 填了上传时间和 public 分数，`beliefs.md` 给 B006、B007 填了 outcome 和 lesson，`journal.md` 追加了一条：这三处都是允许的。但 `ai_usage.md` 第 10 行的 "What for" 一栏被就地改写（"写这行时还没上传"改成了上传记录），严格说不是允许修改的字段（ARCHIVE_RULES §2 只允许改 Status / Outcome）。内容属实，建议在 journal 里记一句，不必回改。
- 缺口：本轮（11-01）AI 做的工作，`registry/ai_usage.md` 和 `registry/journal.md` 里都还没有记录。

**最要紧的五件事**（带日期的完整清单见英文 §5）
1. **11-01 – 11-03：有 NetID 的队员登录 Canvas**，查 checkpoint 截止日、反思 PDF 的截止日和篇幅、团队登记状态（10-16 截止）、视频语言和 zip 截止。checkpoint 如果很近，它排在所有事前面。
2. **定队伍，在 Kaggle 上合队**，要在 checkpoint 截止前做完，因为队伍在那一刻固定。做法：队友先加入比赛，再接受现有 Kaggle 队伍发出的邀请。然后填 `project.yaml` 的 `team` 和 `report.extensions_owned`，并在 Canvas checkpoint 里写明每人的 Kaggle display name。
3. **11-02 – 11-05：私下、笼统地问讲师**三件事：(a) 无监督步骤里能否用测试集的无标签特征（S003）；(b) 能否用算法给测试行打伪标签再训练（S006）；(c) 能否用预训练的表格基础模型。答复记成新决定，它决定 D010 走哪种情况。
4. **补记录**（A→T：队员让 AI 起草，再由队员核对）：在 `ai_usage.md` 加本轮的行，在 `journal.md` 加 11-01 的条目，写 D002 的冻结决定（原定 11-01）。
5. **后面几步**：11-23 – 11-25 勾选最终提交（默认 S003 + S005）；11-24 – 11-26 录视频；11-28 生成 PDF 和 zip 并上传。11-29 只做补救（截止 23:59 HKT = 11-30 00:59 JST）。

**剩下的 TODO 由谁填**（`reports/` 和 `video/` 里用 Select-String 搜到 106 行，完整清单见英文 §3）
- 全体队员（组队之后）：队名、Kaggle 队名、姓名、学号、Kaggle display name、分工。在核心报告第 3、5 行，贡献表第 52–76 行，视频讲稿和幻灯片里。
- 每个扩展的负责人：各扩展第 3 行的 Author 行。
- 只有队员能回答的：核心报告第 7 行（勾选之后删掉 TODO）、第 116 行（讲师的答复）、第 143 行（每人重跑、改写、核查了什么，确认了哪些决定）；视频语言；反思模板。
- AI 可以先做：贡献表图例里 ai_usage 的行号（要先加 ai_usage 的行）。
- 不用填：`reports/README.md` 里的检查命令、`FACTS.md` 里描述档案现状的句子、`core_submission_history.csv` 的 author 列（抄自只追加的 registry）、`reports/build/report.tex`（10:46 的旧排版副本，不进 git，也不进 zip）。

**另外注意**
- 页数余量不大：core 还剩约 0.23 页（第 6 页约 12 行），E4 约 0.15 页。改写之后用 `build_report.py --parts <部分> --out $env:TEMP\check.pdf` 重查。只用 `build_report.py` 排版：按 Word 默认段距（行距 1.08、段后 8 pt），E2、E4 会超过 2 页（ext_figures）。
- 本轮三份 AI 审稿（N1–N22、C1–C36、U1–U18）的原文没有存进仓库，只有 fixed 代理列出的未处理项。英文 §6 列出这些项，并写了我从文件里看到的具体差异。

## English

### 1. Command results

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe scripts\archive_check.py` | Exit 0. `runs=214, runs_without_conclusion=0, registry_rows=214, code_bundles_referenced=46, jobs=90, submissions=6, report_docs=21`. **0 problems, 4 warnings**, all known since 10-22: failed runs `20261022-065046_r152_calibration` (BrokenProcessPool), `20261022-071440_r201_qda_all` (ValueError) and `20261022-091033_r900_perm_null_a` (BrokenProcessPool), each with a written Conclusion; and `research/reports/sup.md` cites `r20x_candidates.csv`, a wildcard name, not a file. Re-run after writing this file: `report_docs=22`, still 0 problems and the same 4 warnings. |
| `.venv\Scripts\python.exe -m pytest -q` | Exit 0, **245 passed** in 30.25 s; none skipped, none failed. I added `-p no:cacheprovider` and set `PYTHONDONTWRITEBYTECODE=1` so that nothing was written to the project. `pyproject.toml` has `addopts = "-q"`, so the extra `-q` becomes `-qq` and hides the summary line (the run printed 245 dots and exited 0); I read the count from a second run with `-o addopts=""`. |
| `.venv\Scripts\python.exe scripts\build_report.py` (with `--out` and `--tmp-dir` in the scratch folder) | Exit 0, 9.5 s, **15 pages**: core 6 (5.77 used, limit 6, pp. 1–6); E1 2 (1.79, pp. 7–8); E2 2 (1.74, pp. 9–10); E3 2 (1.64, pp. 11–12); E4 2 (1.85, pp. 13–14); contribution table 1 (0.90, p. 15). Warnings: `report.extensions_owned` is empty, so all four extensions are included; TODO left: core 8, e1 2, e2 2, e3 2, e4 2, contrib 43. No margin overflow, missing figure, missing glyph or heading warning. The text of `reports/report.pdf` (written 12:03:36) is identical to the text of this build (pdftotext, 679 lines), so the PDF in `reports/` is current. |
| `.venv\Scripts\python.exe scripts\package_deliverables.py` | Exit 0, dry run, nothing written. `[MISSING]` team name (`team.name` is TODO; it blocks `--build`); `[MISSING]` video; `[OK]` report PDF (`reports/report.pdf`, 0.77 MB); `[CHECK]` contribution table (only PDFs are zipped; the table is p. 15 of `report.pdf`, so this is met); `[OK]` code, 180 files, 14.54 MB; `[OK]` submission history, 6 submissions, 6 with a public score, 0 final; `[CHECK]` `project.yaml`, 8 TODO values. Records: 1,645 files (registry 9, experiments 1,636 without `*.npy`), 38.51 MB; total 53.81 MB uncompressed. My extra scan: the only e-mail addresses in the files that would be zipped are the two personal Gmail addresses under `project.yaml` `compute.accounts`; no file contains the Windows user path. |

### 2. Deliverables

Status uses three levels: ready / ready except team fields / not started. "Team fields" are what only members can supply: names, IDs, Kaggle names, ownership, their answers and their own review.

| Deliverable | File(s) | Status | What exactly is missing | Who |
|---|---|---|---|---|
| Core report (max. 6 pages) | `reports/report_core.md` | Ready except team fields | :3 team name and Kaggle team name; :5 member names (drop the unused slots for 1 or 2 members); :7 "TODO: tick them on Kaggle, 23–25 Nov" (delete after ticking; change the second final if the instructor's answer changes the D010 case); :116 the instructor's ruling on D007, with date; :143 what each member re-ran, rewrote and checked, and which decisions they confirmed (U12: add the member-review sentence only when it is true). Every section needs a member lead who can explain it (rule 6). Space left: 0.23 page, about 12 lines on p. 6. | Members |
| E1 (max. 2 pages) | `reports/extensions/e1_validation_protocol.md` | Ready except team fields | :3 author and Kaggle name; an owner. Review items C2 (:5) and C32 (:45, :56), see §6. 1.79 pages. | E1 owner |
| E2 | `reports/extensions/e2_learning_curve.md` | Ready except team fields | :3; C2; C17 (:35); C31 (:59, :69). `extensions/e2_learning_curve/README.md` :77 should mention `--replot` and "nominal 90% band" (ext_figures). 1.74 pages. | E2 owner |
| E3 | `reports/extensions/e3_leakage.md` | Ready except team fields | :3; C2; C33 (:66–68). Leftover from `fix_round1.md` §7: `extensions/e3_leakage/README.md` :181 and the run Conclusion. 1.64 pages. | E3 owner |
| E4 | `reports/extensions/e4_feature_stability.md` | Ready except team fields | :3; C2; C18 (:97–98). `extensions/e4_feature_stability/README.md` :27 (log rank axis). 1.85 pages, about 7 lines spare. | E4 owner |
| Contribution table (1 page) | `reports/contribution_table.md` | Ready except team fields | 43 TODO: :52 team and Kaggle team; :56–58 member rows (name, ID, student ID, Kaggle display name, owned extension); :62–76 Lead and Contributors of 15 parts; :78 "row TODO" (the number of the new `ai_usage.md` row). Delete rows of absent members and unowned extensions. C3 (high): the E1–E4 rows (:68–71) say "sole author", but the legend (:78) says Claude Code did the operating work. C34: AI-use cells. Re-check one page after filling (0.90 now). | Members; :78 A→T |
| `reports/report.pdf` | `reports/report.pdf`, built from the parts and `project.yaml` `report.extensions_owned` | Ready except team fields | Current: same text as a fresh build at 12:10. It holds all four extensions and every placeholder. Rebuild after the fields are filled and `extensions_owned` lists only the owned extensions; the build should then print no TODO warning. | Members (run the script) |
| Video kit | `video/SCRIPT.md`, `video/slides.tex`, `video/slides.pdf`, `video/RECORDING.md`, `video/PLAN.md` | Ready except team fields; recording not started | Names in `SCRIPT.md` :74–77, :169–171, :181 and in the `EDIT HERE` block of `slides.tex` (:17–25); switch off the unowned extensions (`\Eonefalse` etc.); rebuild `slides.pdf` (`RECORDING.md` §2). The current `slides.pdf` (11:02) embeds the older `core_journey.png` (1386 × 561 px; the file regenerated at 11:47 is 1386 × 462). Language requirement: Canvas (`PLAN.md` :9). Then record with every member on camera, at most 5:00 and 300 MB, as exactly one file `video\<team>.mp4`. C36 (`PLAN.md`) is open. All 24 distinct spoken numbers that have a decimal point or at least 3 digits occur in the core report, the extensions or `reports/tables/*.md`. | Members |
| Reflections (one PDF per member, on Canvas) | `reports/reflections/TEMPLATE.md`, `reports/reflections/FACTS.md` | Not started (members only) | Each member writes their own, in English and in their own words. Members' own beliefs and journal entries do not exist yet: B001–B008 and the journal are AI-written (author TODO). Deadline and length unknown (Canvas). A missing reflection gives zero for that member's extension. C35 (`FACTS.md`) is open. | Each member |
| Code reproduction | `README.md` "Reproducing the final submission", `configs/final_s003_gmm2_hv.yaml`, `requirements-lock.txt` | S003: ready. S005 route: not started | S003: the config route was re-run on Colab (`20261022-212507_final_s003_gmm2_hv`, sha256 prefix `3aae140e5837753f`). S005 (r201, the second default final) has no README command, only the research script `research/r201_sup_quadratic.py` (bit-exact re-run `20261022-091224_r910_r201_qda_block`). The repository has no commit (`git log`: none; D004), so the "clean commit" condition of `docs/VALIDATION_PROTOCOL.md` §5.1 is open. The `.venv\Scripts\*.exe` launchers and the Jupyter kernel still point to the old folder (`docs/SETUP.md` §1). | A→T: S005 route, after a Colab run a member authorises. T: commit, private remote, kernel |
| Kaggle final selection | Kaggle Submissions tab; `registry/submissions.csv` (`selected_final`); `registry/decisions.md` | Not started (planned 11-23 to 11-25) | The rule is written: D010 + D013 (both "proposed"; team confirmation TODO). The D010 case depends on the instructor's answer. Then tick 2 submissions (confirm the number on the tab), run `record_lb.py --sub S### --final yes` for each, and append a decision "final = ..., by D010". Case 3 needs r930 pre-registered and uploaded first (prediction in D013). | Members (Kaggle team) |
| Canvas checkpoint | Canvas (no file) | Not started; date unknown | A NetID login to find the date. Team of 1, 2 or 3, fixed at its deadline, with the Kaggle merger done first (the Kaggle team has 1 member today); every member's Kaggle display name (rule 1). Team registration (deadline 10-16): status unknown. | Members |
| Team zip | `deliverables/<team.name>.zip` from `package_deliverables.py --build` | Not started | `team.name`; the video file; delete the two e-mail addresses in `project.yaml` `compute.accounts`; rebuild `report.pdf` first; unzip once and check. | Members |

### 3. Remaining TODO placeholders, by who fills them

`Select-String -Pattern TODO -CaseSensitive` over every text file in `reports/` and `video/` finds 106 lines in 15 files. Every line is in exactly one row below.

| Who fills it | File:line | What |
|---|---|---|
| All members, once the team is fixed | `reports/report_core.md`:3, :5 | team name, Kaggle team name, member names |
| | `reports/contribution_table.md`:52, :56–58, :62–76 | team and Kaggle team; member rows; Lead and Contributors of 15 parts (42 of the 43 occurrences) |
| | `video/SCRIPT.md`:74–77, :169–171, :181 | team, Kaggle team and speaker names (3-, 2- and 1-member versions) |
| | `video/slides.tex`:17–25 | `\TeamName`, `\KaggleTeam`, `\NameA`–`\NameC`, `\OwnerEone`–`\OwnerEfour`; rebuild `slides.pdf` afterwards |
| | `video/PLAN.md`:46, :57, :58; `video/RECORDING.md`:122, :128 | `video\TODO_TEAM_NAME.mp4` in the ffmpeg / ffprobe commands: use the team name when running them |
| Each extension owner | `reports/extensions/e1_validation_protocol.md`:3, `e2_learning_curve.md`:3, `e3_leakage.md`:3, `e4_feature_stability.md`:3 | author and Kaggle display name (only owned extensions go into the PDF) |
| The Kaggle team, after ticking | `reports/report_core.md`:7 | "TODO: tick them on Kaggle, 23–25 Nov" |
| The member who asks the instructor | `reports/report_core.md`:116 | date and answer of the instructor's ruling |
| Each member | `reports/report_core.md`:142 (comment), :143 | what each member re-ran, rewrote and checked; decisions confirmed |
| | `reports/reflections/TEMPLATE.md`:1, :3, :18, :25, :32, :41 | own copy `reports/reflections/<id>.md` (not in the zip) |
| A member with Canvas access | `reports/reflections/TEMPLATE.md`:9; `video/PLAN.md`:9; `video/RECORDING.md`:12 | reflection length and deadline; video language |
| AI on a member's request (A→T) | `reports/contribution_table.md`:78 | "row TODO" = the number of the new `ai_usage.md` row for the 2026-11-01 work (append that row first) |
| Nobody: not placeholders | `reports/README.md`:38, :40 | the TODO search command itself |
| | `reports/reflections/FACTS.md`:21, :28, :29, :31, :32, :34, :53, :73, :210 | statements that a field is still TODO in the archive; they go out of date when the fields are filled |
| | `video/SCRIPT.md`:203 | word-count note ("each TODO name counts as one word") |
| | `reports/tables/core_submission_history.csv`:2–7 | author `TODO_ID_1`, copied by `submission_history.py` from the append-only `registry/submissions.csv`; the report does not print this column |
| | `reports/build/report.tex` (27 lines) | a `--keep-tex` copy from 10:46, older than the current build; git-ignored, not zipped |

### 4. Append-only audit (snapshot `backups/snapshot_20261022-213717.zip`)

Method: every one of the 2,685 snapshot entries was compared with the current file by size and CRC32. Current files in the snapshot's folders that the snapshot lacks were listed. The snapshot copies of `registry/` and of all 30 changed files were extracted to `...\final_check\snapshot` and diffed by row (CSV) or by line (Markdown). Result: 2,655 files unchanged, 30 changed, 23 new, 0 deleted.

| Area | Check | Result |
|---|---|---|
| `experiments/` | 2,015 files, size + CRC32 | **All identical**, including every `notes.md`. No folder or file added or removed. 214 folders = 214 registry rows (archive_check). |
| `registry/experiments.csv`, `registry/compute_jobs.csv` | snapshot copy vs current | **Identical** (214 and 90 rows). |
| `registry/submissions.csv` | row diff | 6 rows. Only S005 and S006 changed: `submitted_at_utc` and `public_lb` filled at 2026-10-22 22:23 JST (previous session, after the snapshot). Both are allowed fields. Not touched this round. |
| `submissions/`, `logs/` | size + CRC32 | Identical (6 and 149 files). |
| `registry/decisions.md` | line diff | This round: D014 appended (:155–170); the Status lines of D007 (:81) and D010 (:107) amended. No other line changed. **Rule held.** |
| `registry/beliefs.md` | line diff | Outcome and lesson cells of B006 and B007 filled (10-22 22:23, previous session); the predictions are unchanged. Allowed. |
| `registry/journal.md` | line diff | One entry appended (10-22 22:23, previous session). Allowed. No entry for 2026-11-01. |
| `registry/ai_usage.md` | line diff | Row 10, "What for" cell, edited in place at 10-22 22:23 (previous session): "they were not uploaded when this row was written" became the upload record. The text is true, but this is not an allowed field (ARCHIVE_RULES §2: old entries only get Status / Outcome updates). Suggestion: note it in the journal; no rewrite. No row for the 2026-11-01 work. |
| `registry/ideas.md`, `registry/references.md` | line diff | Unchanged. |
| `reports/figures/`, `reports/tables/` | size + CRC32; manifests | 9 PNGs changed (`e2_*` 6, `e4_*` 3; 10:01–10:04): regenerated by the extensions' own `run.py --replot` from the archived run tables. Their PNG metadata names the source run; `e2_manifest.json`, `e2_qda_manifest.json`, `e2_gmm_manifest.json` and `e4_manifest.json` are unchanged. 11 new `core_*` files: `core_submission_history.*` (09:38, `scripts/submission_history.py`) and the rest (11:47, `scripts/make_core_figures.py`). The input hashes in `core_manifest.json` match the current files. No `r###_`, `e1_` or `e3_` file changed. |
| `configs/`, `research/`, `src/` | size + CRC32 | Unchanged (8, 109 and 10 files). |
| Code and docs | diff | Changed: `extensions/e2_learning_curve/run.py` and `extensions/e4_feature_stability/run.py` (`--replot`); new `scripts/build_report.py`, `scripts/make_core_figures.py`, `tests/test_build_report.py`; `project.yaml` gained only the `report:` block (`official_cv` unchanged); `.gitignore` gained `reports/build/`. Docs rewritten: README, `colab/README.md`, `colab/login_account.sh`, `docs/ASSIGNMENT_CHECKLIST.md`, `SETUP.md`, `TIMELINE.md`, new `docs/COMPETITION_RULES.md`. `docs/ARCHIVE_RULES.md` and `docs/VALIDATION_PROTOCOL.md` are unchanged, so no decision was needed. No formal run used the changed code. |
| Whole tree | mtime scan (without `.git`, `.venv`) | Files modified on 2026-11-01 are the ones above, the Markdown drafts in `reports/` and `video/`, `reports/report.pdf`, `video/slides.*`, `reports/build/` and `.pytest_cache/` (from the builder's test runs); nothing in `sandbox/`, `data/` or `submissions/`. A 0-byte `make_figures.py` in the project root dates from 10-24, is referenced nowhere and is not zipped; a member can delete it. |

### 5. Prioritised to-do list, 2026-11-01 to 2026-11-29

T = only members can do it; A→T = the AI can draft it when a member asks, then members check it and must be able to explain it. The Canvas checkpoint and the reflection deadline are unknown until a member logs in to Canvas; if either falls inside a week, it goes first.

| When | # | Task | Who |
|---|---|---|---|
| 11-01 – 11-03 | 1 | Log in to Canvas with a NetID. Record the checkpoint deadline and content, the reflection deadline and length, the team-registration status (deadline 10-16), the video language, any AI-disclosure format and the zip deadline in `docs/ASSIGNMENT_CHECKLIST.md`, `README.md` (key dates) and `docs/TIMELINE.md`. | T |
| Before the checkpoint deadline | 2 | Fix the team (1, 2 or 3 members). Teammates join the competition and accept its rules; the existing Kaggle team invites them on the Team tab; they accept. Share the repository, predictions and scores only after the merger (rule 5; Foundational 5b, 5c; D014). | T |
| Right after #2 | 3 | Fill the `project.yaml` `team` block (names, ids, student ids, Kaggle display names, one different extension each) and `report.extensions_owned`. | T |
| Before the checkpoint deadline | 4 | Submit the Canvas checkpoint with every member's Kaggle display name (rule 1). | T |
| 11-01 – 11-03 | 5 | Append `registry/ai_usage.md` rows and a `registry/journal.md` entry for the 2026-11-01 workflow (core report and its figures, `build_report.py`, the E2/E4 replot, contribution table, `FACTS.md`, video kit, rules archive D014 and docs, this check). Then put the row number into `reports/contribution_table.md` :78. | A→T |
| 11-02 – 11-05 | 6 | Ask the instructor, privately and in general terms: (a) unlabelled test features in unsupervised steps (S003); (b) retraining on algorithmic pseudo-labels of test rows (S006); (c) pretrained tabular foundation models. Record the answer as a new decision. It picks the D010 case. | T |
| 11-02 – 11-05 | 7 | Decision freezing D002 (target was 11-01); fill `TODO(E1)` and the `test_pred` mode in `docs/VALIDATION_PROTOCOL.md`; say there that §5 was replaced by D010. Name question (b) of #6 in `docs/ASSIGNMENT_CHECKLIST.md` :37 (C12). | A→T |
| 11-02 – 11-05 | 8 | Set git identities, make the first commit, create a private remote (D004); sync the Windows clock (D012); re-register the Jupyter kernel (`docs/SETUP.md` §1). | T |
| 11-02 – 11-08 | 9 | Read and confirm D007 and D009–D014 (D010 and D013 are "proposed"; D014 was accepted by the AI only); fill "Decided by" of D001–D003; save the Kaggle rules page as a PDF in `docs/` (steps in `docs/COMPETITION_RULES.md`). | T |
| 11-06 – 11-08 | 10 | Extension owners read their code and draft. Sunday 11-08: `archive_check.py --snapshot`. | T |
| 11-09 – 11-15 | 11 | Each owner rewrites their extension in their own words, fixes C2 and their items in §6, fills the author line, and checks pages with `build_report.py --parts e# --out $env:TEMP\e#.pdf`. A Colab re-run is optional. | T (A→T help) |
| 11-09 – 11-15 | 12 | Each member writes at least two own beliefs in `registry/beliefs.md` (for example S003's private score) and own difficulties in `registry/journal.md`: the material for reflection questions 2–4. | T |
| 11-09 – 11-15 | 13 | One member leads each core-report section, reviews the draft and cuts what nobody can explain. | T |
| 11-09 – 11-15 | 14 | README command for reproducing S005 (r201), after one Colab run that a member authorises. | A→T |
| 11-16 – 11-22 | 15 | Explanation session (rule 6): official CV, the basis of D007, GMM/EM and QDA, the generator identification, public-LB noise, each extension. | T |
| 11-16 – 11-22 | 16 | Finish the contribution table (fix "sole author", C3; AI-use cells, C34), the `ai_usage.md` Member and Verified-by cells, and core :143 (U12 only when true); build the whole PDF. Sunday 11-15 and 11-22: `archive_check.py`. | T |
| 11-16 – 11-22 | 17 | Video: each speaker rewrites their lines; fill `slides.tex` `EDIT HERE`, switch off unowned extensions, rebuild `slides.pdf`. | T (A→T help) |
| By 11-22 | 18 | If the instructor has answered: a decision fixing the D010 case. No rule changes after that. | T |
| 11-23 – 11-25 | 19 | Tick the 2 finals on the Kaggle Submissions tab by D010/D013 (default S003 + S005); `record_lb.py --sub S### --final yes`; decision "final = ..., by D010"; delete the TODO in core :7. Case 3 only: pre-register and upload r930 first. | T |
| 11-24 – 11-26 | 20 | Record the video: every member on camera while speaking, at most 5:00; compress to at most 300 MB (Clipchamp or HandBrake; ffmpeg is not installed); exactly one file `video\<team>.mp4`. | T |
| 11-26 – 11-27 | 21 | Final text; delete the two e-mail addresses in `project.yaml` `compute.accounts`; `package_deliverables.py` dry run without MISSING; each member writes and exports their reflection (earlier if Canvas says so). | T |
| 11-28 (Sat) | 22 | `build_report.py` to `reports\report.pdf` (only owned extensions, no TODO warning); `package_deliverables.py --build`; unzip and check; upload the zip to Canvas; each member uploads their reflection PDF. | T |
| 11-29 (Sun) | 23 | Buffer for repairs only; `archive_check.py`. Kaggle closes 23:59 HKT = 11-30 00:59 JST. | T |
| After the close | 24 | Record the private scores of all six submissions (`record_lb.py --private`) and the outcomes of B004 and B008; use them in reflection question 4 only if its deadline is later. | T or A→T |

### 6. Review items that were not applied this round

The fixed agent listed these as open because they sit in files it did not own. The review texts themselves are not in the repository. The third column is my own reading of the files, so the reviewer's exact point may differ. Saving the three reviews (for example as `research/reports/review_round1.md`) would let owners read the original wording.

| Item | File | What the files show | Who |
|---|---|---|---|
| C2 (high) | E1–E4 :5 | The AI-use line says only that "analysis code and this draft were prepared with an AI assistant (Claude Code) and checked against the archive". The core report (:143) says the AI designed, coded and ran all 214 runs and drafted the extensions. Make the four lines agree with it and say who checked. | Owners (A→T now) |
| C3 (high) | `reports/contribution_table.md` :68–71 | "sole author" in the Contributors cells, against the legend (:78). | Members |
| C10 | `registry/ai_usage.md` | No row for 2026-11-01 (see §5 #5). | A→T |
| C11 | `registry/decisions.md`, `docs/VALIDATION_PROTOCOL.md` | No D002 freeze decision; the protocol is still marked "draft" (§5 #7). | A→T |
| C12 | `docs/ASSIGNMENT_CHECKLIST.md` :37 | The instructor question names only the unsupervised use of test features, not S006's pseudo-labels (§5 #6). | A→T |
| C17 | E2 :35 | Calls QDA with `reg_param` 0.1 (S002's model, CV 0.9958) "the best supervised model"; S005 (nested-reg QDA, CV 0.9999) is better. | E2 owner |
| C18 | E4 :97–98 | "best linear model in R300 (L2 on the whole block, 0.898)"; the core report gives 0.932 for the best penalised linear model (R249) and 0.947 for the linear limit (R154). | E4 owner |
| C24 | `registry/decisions.md` (D007, D014 Status), `docs/COMPETITION_RULES.md` §4 item 1 | Status notes on the pending instructor answer. | A→T |
| C31 | E2 :59, :69 | "about 75 [labels] for QDA (median AUC 0.99)" and "189 of 19,750 with a posterior of 0.1–0.9; R422"; the core report uses R441 (label-only QDAs 120–160, S005 120, QDA at fixed reg 0.003: 40; mean AUC at least 0.99) and about 186 rows (R154). Different runs and definitions: name them or align. | E2 owner |
| C32 | E1 :45, :56 | "orders a 0.01 pair correctly only 64 % of the time" and "at least 0.93 for all 28 pairs" are stated without the core report's qualifier (:62: a normal approximation; the studied pair is about 0.004 apart, R940). | E1 owner |
| C33 | E3 :66–68 | E3 says the proxy expectation (0.820) is 0.028, "1.2 SE", above S004's public score; core Table 1 gives z = −1.35 for the same gap. The two use different SE conventions (core comment :74–77): name the one used. | E3 owner |
| C34 | `reports/contribution_table.md` AI-use cells; `registry/ai_usage.md` rows | AI-use pointers and rows for the new work. | Members + A→T |
| C35 | `reports/reflections/FACTS.md` | Details are not in the repository. | A→T |
| C36 | `video/PLAN.md` | Details are not in the repository. | A→T |
| U12 | `reports/report_core.md` :143 | Replace the closing sentence with a member-review claim only when it is true; the Verified-by cell of every `ai_usage.md` row still has a member TODO. | Members |
| C26, 2nd clause | `reports/report_core.md` | Rejected by the fixer: the 13-column block is not the 12 informative columns (R901). No action. | – |

### 7. Method and scratch files

All in `D:\tmp\claude\D--polyu-class------\76f1e57c-48e0-46a7-82bf-66f79a8ce473\scratchpad\final_check`:
- `manifest.py`: SHA-256 manifest of the project (without `.git`, `.venv`; large backups and raw data by size and mtime). `manifest_before.tsv` and `manifest_after_checks.tsv` are identical: the four commands changed no project file.
- `pytest.txt`, `pytest_summary.txt`: test output.
- `build\report_check.pdf`: the scratch build. `build\report_existing.txt` and `build\report_check.txt`: pdftotext of the existing and the fresh PDF, compared line by line.
- `snap_compare.py` and `snap_compare.tsv`: snapshot vs current, per file. `snapshot\`: extracted snapshot copies of `registry/` and of the 30 changed files. `registry_diff.py`, `line_diff.py` and `registry_diff.txt`: row and line diffs of the registry.
- `png_meta.py`: PNG metadata of the regenerated figures. `pdfimages -list` on a copy of `video/slides.pdf` gave the embedded image sizes.
- `num_cross.py`: spoken numbers of `video/SCRIPT.md` checked against the core report, the extensions and `reports/tables/*.md`.

## Update（主会话，2026-11-01 12:50 JST，在本报告之后完成；上面的内容保留原样）

- §6 的审稿意见：三份审稿原文和每条的处理结果已存档，见 `research/reports/review_round1.md`。C2、C3、C10、C11、C12、C17、C18、C24、C31–C36 都已改。U12 仍待成员（只有在成员真正复核后才能改）。C33 按审稿人的原意改了：落在预登记区间里的是 public 分数，不是 proxy。E3（第 37 行：按 public 分数算的 Hanley–McNeil SE）和核心报告 Table 1（按预期 AUC 算的 SE）各自写明了 SE 的算法，所以 1.2 和 1.35 两个数字都对。
- §5 #5：`registry/ai_usage.md` 追加了第 14–21 行，`registry/journal.md` 有 2026-11-01 条目。
- §5 #7：写了 D015（proposed：D002 冻结、不作修改；§5 的选择规则由 D010 取代；讲师问题的三问）。`docs/VALIDATION_PROTOCOL.md` 的 `TODO(E1)`、test_pred 模式和 §5 的说明都已补；清单里的讲师问题现在写明了 (a)(b)(c) 三问。
- §5 #8：Jupyter kernel 已重新注册。git 身份和第一次 commit、同步 Windows 时钟仍由队员做。
- §5 #14：S005 的 README 复现路线已写进 README，并在 Colab 上实测（按用户 10-22 对 Colab 的长期授权）。run `20261101-123515_r201_qda_block` 与原 run 逐位相同，提交文件的 sha256 前缀 3be09c0723286457 与 S005 一致。
- §5 #16：贡献表的 "sole author" 已去掉（C3），AI use 一栏已填（C34）。成员的 Member、Verified by 两栏仍是 TODO。
- §5 #17：讲稿和幻灯片里与审稿意见冲突的说法已改，`video/slides.pdf` 已重建（用的是 11:47 重画后的 core 图）。
- §5 #21：不用再手动删邮箱了。`scripts/package_deliverables.py` 写进 zip 的 `project.yaml` 会把邮箱换成 `<removed>`，预检时其他含邮箱的文件会标成 CHECK；有测试覆盖。
- 重建后的 `reports/report.pdf` 页数：core 5.77/6，E1 1.89，E2 1.80，E3 1.64，E4 1.88，贡献表 0.94。pytest 248 个全部通过；archive_check 0 个问题（215 个 run）。快照 `backups/snapshot_20261101-125005.zip`。
