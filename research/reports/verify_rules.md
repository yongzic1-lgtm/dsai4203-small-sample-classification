# verify_rules：规则合规与交付物准备情况核查（Verify 第 1 轮，lens = rules，第 2 版）

作者：AI 核查代理 "rules"（Claude Code，Claude Opus 5.5），2026-10-22 约 16:00–17:00 JST。
方法：只读档案文件。没有跑 Colab，没有新建 Run，也没有改别人的文件。唯一的代码是只读的引用清点脚本
`research/r945_citation_inventory.py`。它扫描代码，不读数据，也不写档案，所以不建 Run。本版修了脚本的自匹配问题，
并补上了写在字符串里的 import。提交前再跑一次：`.venv\Scripts\python.exe research\r945_citation_inventory.py`。
本文件替换 09:18 的第 1 版。第 1 版里有 4 处错误，已在英文部分 §0 更正。

## Errata (verify round 1, 2026-10-22)

以下更正已直接改进正文（中文摘要第 4 点、英文 §0、§4 表 4、§5）。依据：`research/reports/verify.md` §3 C3、C6，`research/reports/verify_ext.md` §1。

1. **r210 re-run**（§5 "Next upload"）。Old: "It is equally compliant but has not been independently re-run." → New: r210's test_pred was reproduced bit-identically by an independent re-implementation (r932) and its CV by R900. Sources: `experiments/20261022-092058_r932_real_repro_agreement/notes.md` Conclusion (S003, r209, r210, r201 bit-identical; nested reg choices of r201/r210 in 100/100 folds); `reports/tables/r900_perm_null_A_summary.csv` (r210 real AUC 0.9999838 = archived, difference 0).
2. **Extension page counts**（摘要第 4 点；§0 表；§4 表 4）。Old: "about 2.25–2.5 pages each" / "实测每份约 2.25–2.5 页" → New: 2.16–2.71 pages (E1 2.55–2.66, E2 2.57–2.71, E3 2.16–2.22, E4 2.19–2.23; figure at 60% / 100%). Source: `research/reports/verify_ext.md` §1 (XeLaTeX, Calibri 11 pt, A4, 2.54 cm margins).
3. **Cuts needed**（摘要第 4 点；§4 表 4）。The synthesis lists "150–250 words" as a misquote. That figure does not occur in this file, so there was nothing to replace. The correct cut sizes are added next to the page counts: about 400 words in E1/E2, about 250 in E3/E4, and E4 drops or merges one table (verify_ext §1).

## 中文摘要（给队员）

**一句话结论**：四次提交（S001–S004）都没有违反比赛规则。真正的风险有三类：
1. 有两条规则解释还没有被讲师或队员确认：用测试集特征做直推式建模，以及预训练基础模型。
2. "文书"不合格：引用缺 39 条；AI 披露里有一句话与事实不符；所有署名都是 TODO。
3. 交付物几乎都没开始：核心报告 0 字；4 份扩展草稿都超过 2 页；贡献表、视频、反思只有模板。

另外，全部 191 个 run 和全部决定都是 AI 做的。"每个成员都能解释每个细节"这一条，现在还做不到。

**1. 比赛规则（逐项结论，详见英文 §1 表 1；round1 的 53 个候选逐个归类）**

- **提交次数与预登记：合规。**
  - 2026-10-21 这个 Kaggle 日（UTC）交了 3 次（S001–S003）。
  - 2026-10-22 交了 1 次（S004，16:08 JST，public 0.79136）。
  - 上限是每天 5 次，队规是 3 次。每次上传前都写了 `expected_public`。B005 在上传前就已登记，我 16:05 读文件时它还是 pending。
- **严格监督的候选（不碰测试集特征）：合规。** 比如 r201_qda_block、r210_qda_block_nll、r255_bqda_eb_trainonly、E3 模型。
- **用测试集特征做直推式建模（S002、S003、所有 GMM / 自训练 / label spreading 候选）：按 D007 判为合规，但这是"未核实"的合规。**
  - 支持理由：规则 4 禁止的是"比赛之外的数据"，测试集特征属于比赛数据；无监督步骤不看标签；伪标签由算法产生，不是人工标注。
  - 缺口 1：比赛规则原文没有存进档案。D007 引用的 "Foundational Rules 4b" 在仓库里找不到出处。
  - 缺口 2：D007 写着"团队确认 TODO"。
  - S003 是主力提交，所以建议**私下、用笼统的问法**问讲师一句（不提分数和方法细节）。
  - 同时上传一个严格监督的候选（建议 r201_qda_block）作为"规则保险"，见第 5 点。
- **TabPFN v2 / TabICL v2：只作分析对照可以；不能进入提交或集成。** 讲师同意之前，一律按不合规处理。
  - "只用合成数据预训练"这句话在档案里**没有证据**。存档的两份模型卡都没写训练数据。
  - TabPFN 实际用的文件 `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`，模型卡说它是 `v2_default` 的别名。这一点第 1 版说得过重，已更正。
  - 但 `v2_default` 本身用什么数据训练，卡上没写。据我所知，Prior Labs 还发布过用真实数据继续预训练的版本（Real-TabPFN）。这一条来自我的背景知识，档案里没有。
  - 所以报告里**不要写 "exclusively synthetic"**，除非有人查实。
  - `fm.md` 把 TabICLv2 写成 "ICML 2026"，但模型卡只给了 arXiv 预印本，要改。
  - 已确认：FM 的预测没有进入任何提交或其他代码。
- **复刻数据 / `make_classification` 模拟：合规。** 它们只用于评估和比较流程，没有训练任何要提交的模型。
  - 代码里查不到"暴力搜索随机种子、复现隐藏标签"的尝试。gen 代理也明确说没做。
  - 建议写一条团队决定，明令禁止两件事：反推随机种子或隐藏标签；用复刻数据的行去训练要提交的模型。复刻库已经和 sklearn 逐位一致，技术上做这件事很容易。
- **数据和代码传到 Colab VM：不算"队外共享"（有条件）。**
  - 理由：VM 只有我们自己的账号能访问；Google 是算力提供方，不是接收方；§8 本来就允许私有网盘。
  - 45 个代码包逐个检查过，里面没有数据文件、token 或凭据。
  - 条件 1：pro2 账号必须是队员本人的账号，档案里没写，请确认。
  - 条件 2：不要分享 notebook 或 Drive。
  - Colab 自己的服务条款可能限制"多账号绕过用量限制"。这是我对 Colab FAQ 的记忆，没有核实；它不是比赛规则，但影响账号安全。
- **Kaggle 意义上的"队"：** 现在档案里只有一个 Kaggle 参赛者（用户本人），队友信息全是 TODO。
  - Kaggle 规则里的"队内"指 Kaggle 上合并过的队伍。队友在合队之前看到仓库、预测或分数，按 Kaggle 规则就算"队外私下共享"。
  - 所以要先合队，并查清合队截止日，再共享。
- **一处元数据错误**：`r300_block_l2_c1` / `_c01` 的 config 写着 `uses_unlabeled_test_features: false`。但它们用的固定 13 列，是在 train+test 特征上用方差规则选出来的。报告里不能把它们算作"严格监督"。

**2. 引用（`registry/references.md`，详见英文 §2）**

- 现在有 24 条，其中 8 条是 used。Nadeau–Bengio 和 Hanley–McNeil 两条都在，书目信息正确。
- **缺 39 条**：
  - 软件 7 条：TabPFN、TabICL、TabICLv2、PyTorch、NumPyro、Pyro、JAX；
  - 方法 32 条：EM、BIC、make_classification（Guyon 2003）、RDA、liblinear、SAGA、LIBSVM、Platt、kNN、RF、ExtraTrees、GP、self-training、pseudo-label、label spreading、label propagation、半监督 EM、DP 混合、Bishop、NIW、BDA、horseshoe、正则化 horseshoe、NUTS、R-hat、Gelman–Rubin、QR、Mardia、D'Agostino、ABC 两篇、Wilks。
  - 英文 §2 给了可以直接粘贴的 BibTeX。
- **13 条要从 candidate 改成 used**：LightGBM、XGBoost、Efron & Tibshirani、Hastie、Varma & Simon、Cawley & Talbot、Ledoit–Wolf、Tibshirani、Figueroa；另外 Kohavi、Bengio–Grandvalet、Ambroise、Meinshausen 在对应扩展交上去时改。
- PyMC 没用过，不要引用；seaborn 和 statsmodels 没有被导入；Kaggle API 只有代码，没有实际用来上传过（S001–S004 都是网页上传的）。
- **E1、E2 的引用几乎都在 HTML 注释里，导出后会消失。**
- 50 多条引用放不进核心报告 0.4 页的预算。建议的办法：
  - 经典方法引 Hastie 2009 一本教科书即可；
  - 探索性 run 统一写成一行 "software / methods used in exploratory runs"，但不能漏掉。

**3. 生成式 AI 披露（`registry/ai_usage.md`）：不够，而且有一句话不符合事实。**

- 草稿里那句 "All experiments, results, interpretations and the final text were produced and checked by the team" 和档案矛盾：191 个 run 全部由 AI 设计并运行，作者都是 `TODO_ID_1`。**这句必须重写。**
- 表格坏了：第 11 行的引用块紧贴第 12 行，第 2、3 行会被渲染进引用块，不成表格。
- "Member" 和 "Verified by" 两栏全是 TODO。
- **第 1 版说"上传 S001–S003 和扩展草稿没记"，这是错的**：第 2、3 行已经记了。真正漏记的是：
  - S004 的上传、B005、E3 的 LB 刷新；
  - AI 写了 D005–D009（包括规则解释 D007）、B001–B005、I010–I014 和 journal；
  - `scripts/archive_check.py` 和 Conclusion 的补写；
  - 本轮 verify（R900–R945 和 verify_*.md）；
  - 上传方式：AI 在队员已登录的 Chrome 里操作 Kaggle 网页。
- 核查手段也没写。注意：**单元测试没有覆盖最终模型**（`tests/` 里没有任何测试用到 `transductive.py`），所以披露里不能写"单元测试验证过最终模型"。
- 扩展是个人打分的，但 4 份草稿里的 "Drafted by Claude Code" 都写在注释里，导出后看不到。每份扩展要有一行可见的 AI 说明。
- 建议的条目和替换段落见英文 §3。

**4. 交付物准备情况（详见英文 §4 表 4 和带日期的 TODO）**

- 核心报告：0 字，全是 TODO。
- 扩展：verify_ext 实测每份 2.16–2.71 页（每份都超过 2 页；E1、E2 要删约 400 词，E3、E4 约 250 词，E4 还要删掉或合并一个表）；作者都是 TODO；E3 的 LB 结果已经有了（S004 0.79136），但正文还写着 pending。
- 贡献表、反思（只有模板）、视频（只有计划，ffmpeg 和 pandoc 都没装；xelatex 有）：都没开始。
- 扩展数量：每个成员只能负责 1 个扩展，2–3 人的队最多 2–3 个扩展计分。用户"四个都要"的想法，要和这条规则对一下。
- 反思 Q4（"被 LB 证伪的信念"）：现在 B001–B003、B005 全被证实，而且全是 AI 写的。B004（private ≥ 0.999）按 R922 只有 14–51% 的概率成立，很可能被证伪，但它不是队员自己的信念。每个队员要在下一次提交**之前**写下自己的 belief。
- 复现：
  - README 用的是 `configs\TODO_final.yaml`，但 `src/ssc/models.py` 不能从 config 构建 TransductiveGMM。S003 只能通过 `research/r003_*.py` 复现；R910 已证明这条路在 Colab 上能逐位复现。
  - 全部 191 个 run 都是 `no-commit`。按 VALIDATION_PROTOCOL §5.1，现在**没有任何合格的最终候选**。
- 打包隐私：zip 会收进 `project.yaml` 和 `colab/README.md`，里面有两个私人邮箱。打包前删掉。

**5. 提交建议（规则角度）**

- 尽快（按队规预登记）上传一个**严格监督**的候选，建议 `20261022-071234_r201_qda_block`。它的 CV 是 0.9999，R910 已逐位复现，完全不用测试集特征。这样最终两个选择可以是"S003 + 严格监督 QDA"：讲师怎么解释规则 4，都至少有一个合规的选择。
- 如果讲师说不能用测试集特征：S002、S003 都不能选，两个最终选择都从严格监督的候选里挑。
- TabPFN / TabICL 的 run 永远不选。

**最紧急的 6 件事（本周内，T = 只有队员能做）**：

1. 10-23（T）查 Canvas：团队登记（10-16 已截止）、checkpoint 日期和 Kaggle display name、反思截止和篇幅、视频语言；填 `project.yaml`。
2. 10-23（T）队友加入比赛并在 Kaggle 合队，查清合队截止日；合队之前不共享仓库和预测。
3. 10-24（T）把 Kaggle Rules 页、Data 页许可和 TabPFN LICENSE 的原文存进 `docs/`。私下问讲师两个问题（测试集特征、预训练模型），答复记为 D010。
4. 10-24（T）设置 git 身份，做第一次 commit，建私有远端；确认 pro2 账号属于谁。
5. 10-25（A→T）修 `ai_usage.md` 和 `references.md`（英文 §2、§3 有现成文本），确认 D007/D009，并加一条"禁止反推种子 / 复刻数据不进训练"的决定。
6. 10-25（主会话）把 E3 草稿里的 "pending" 换成 S004 的结果；预登记并上传 r201_qda_block。

---

## Details (English)

Scope: `round1_explore.json` (6 explorer entries, **53** candidate rows), S001–S004, and `docs/`, `registry/`, `reports/`, `research/`, `extensions/` and `src/ssc/` as of 2026-10-22 about 16:00–17:00 JST. Registry counts are "at time of reading", because other verifiers were still adding r9xx runs. No Colab job, no Run, no edit outside my own files.

### 0. Corrections to version 1 of this report (09:18 JST)

| v1 statement | Correct statement | Evidence |
|---|---|---|
| "All 68 rows of round1_explore.json" | 53 candidate rows: 8 + 1 + 20 + 2 + 8 + 14 | parsed `round1_explore.json` |
| ai_usage.md does not log the S001–S003 uploads (item 4a) or the extension drafts (4e) | Both are logged, coarsely: row 2 ("created, uploaded and recorded Kaggle submissions S001-S003 with the user's authorization") and row 3 ("full real-data runs of extensions E1-E4 and drafts of their 2-page reports"). Not logged: the upload channel, S004, and the other items in §3 | `registry/ai_usage.md:12-13` |
| TabPFN file name "finetuned" is suspicious | The archived card says `finetuned-*` files are "content-identical aliases" and that `...finetuned-zk73skhh.ckpt` = `v2_default`. The training data of `v2_default` is still not stated | `experiments/20261022-063927_r250_fm_licence_check/tabpfn_README.txt:57` |
| Extension pages estimated at 2.3–2.6 from word counts | Measured by verify_ext with XeLaTeX (Calibri 11, A4): 2.16–2.71 pages (E1 2.55–2.66, E2 2.57–2.71, E3 2.16–2.22, E4 2.19–2.23), all over 2 | `research/reports/verify_ext.md` §1 |
| 162/163 runs, 36 bundles, 52 jobs | 191 runs, 45 bundles, 69 Colab jobs | registry files at about 16:10 JST |

v1 of `r945_citation_inventory.py` matched its own pattern table (every method count was +1). It also missed the imports in `r250`, which sit inside a subprocess string, and it reported the Kaggle API as "used" only because the code exists. All three are fixed in v2.

### 1. Competition-rule compliance

**Basis.** The archive's rule text comes from three places:

- `docs/ASSIGNMENT_CHECKLIST.md:108-115` (§G): max 5/day; Kaggle display names at the checkpoint; "everything used must be cited"; "No external data for training"; "No private sharing of code / data / predictions / results outside the team"; GenAI allowed with disclosure, "members must be able to explain every detail";
- `research/reports/fm.md:61`, which quotes rule 4 as "You may not use data from outside this competition to train your model";
- D007, which cites "Foundational Rules 4b" (hand labelling).

**Neither the Kaggle Rules tab nor the course brief is archived.** "Foundational" occurs only in D007 and `src/ssc/transductive.py:8`. The user accepted the Kaggle rules on 2026-10-22 (session memory), but the text was not saved. Every verdict below therefore rests on paraphrases.

**Table 1: verdict per candidate.** All 53 rows of `round1_explore.json` plus S001–S004. "Test features" was checked against each run's `config.yaml` (`uses_unlabeled_test_features`) and the feature provenance.

| Group | Runs (run_id prefix 20261022-) | Ext. data (rule 4) | Pretrained weights | Test features | Verdict |
|---|---|---|---|---|---|
| **S001** | 050647_r003_sup_l1_all | no | no | no | **compliant** |
| **S002** | 050659_r003_sup_qda_hv | no | no | label-free column rule + scaling on train+test | **compliant under D007** (unverified rule text) |
| **S003** | 050708_r003_gmm2_hv | no | no | K=2 mixture fitted on 20,000 feature rows; fold labels only map components | **compliant under D007** (unverified; main residual risk) |
| **S004** | 063908_e3_leakage | no | no | no (in-fold corr top-20 + L2) | **compliant** |
| A. Strictly supervised (16) | 063711_r300_l1_c02, 063710_r300_l1_c01, 063706/063705/063700_r300_corr{50,30,10}_l2, 063908_e3_leakage, 070202_r255_bqda_eb_trainonly, 070206_r255_qda_trainonly, 073018_r253_hs_all, 075227_r257_hs_indep, 082047_r210_qda_block_nll, 071234_r201_qda_block, 072747_r205_select_supervised, 072331_r202_gpc_ard_block, 074613_r203_lgbm_block_nested, 065631_r200_l1_all | no | no | no (flag false; block re-derived in the training fold) | **compliant under any reading of rule 4** |
| B. Label-free test-feature use, supervised fit (18) | 063715/063717_r300_qda_r01/_r03, **063713_r300_block_l2_c1** (flag says false; see below), 070200_r255_bqda_eb_hv, 070204_r255_bqda_weak_hv, 070207_r255_gpc_ard_hv, 070536/070539/070542/070543/070545/070546_r256_* (6), 073222_r253_hs_quadhv, 071441_r201_qda_block_tt, 082044_r210_sup_qda_tt_lw, 080551/080546_r209_sup_qda_tt_r01/_r003, 074147_r207_l2_block_tt_wide | no | no | column rule / scaling only | **compliant under D007**, same residual risk as S002 |
| C. Transductive / semi-supervised on 20,000 rows (15) | 080453/080505_r155_gmm2_{inf,hv}_tight, 074857_r105_gmm2_block_mle, 064436/064447/064454/064735/064749/064738/064500_r102_* (7), 071054_r254_dpgmm_hv, 071450_r254_dpgmm_pca12, 080632_r209_selftrain_qda_lw, 082102_r210_selftrain_qda_nll, 080634_r209_labelspread_widegrid | no | no | yes; pseudo-labels are algorithmic, not hand labels | **compliant under D007**, same residual risk as S003 |
| D. Tabular foundation models (4) | 073743_r251_tabpfn2_all, 073935_r251_tabpfn2_hv, 074251_r251_tabicl_all, 074643_r251_tabicl_hv | **weights fitted outside the competition**; their pretraining data are not documented in the archive | TabPFN v2 (Prior Labs License = Apache-2.0 + attribution), TabICL v2 (BSD-3) | `_hv`: block rule only | **As an analysed comparison: compliant if cited and attributed. As a submission or ensemble input: questionable; treat as non-compliant without the instructor's written approval** |
| E. Simulation / analysis (not candidates) | R103, R150–R156, R254 replicas, R304, R308, R321 proxy, R900–R943 | synthetic replicas, used for evaluation and model comparison only | no | yes (analysis) | **compliant**. It becomes questionable if replica rows are ever added to the training set of a submitted model |

Supporting checks:

- **Limits and pre-registration (confirmed).**
  - Uploads in `registry/submissions.csv`: S001–S003 at 21:14:41, 21:15:16 and 21:16:07 UTC on 10-21 (3 in that Kaggle day); S004 at 07:08:41 UTC on 10-22 (1 so far that day). The limit is 5/day (`project.yaml:7`) and the team limit is 3/day (D008).
  - `expected_public` was written before each upload (created_at 06:12:55–06:13:03 JST and 16:05:37 JST).
  - B005 was still "pending" when I read `beliefs.md` at about 16:05 JST, before the S004 upload at 16:08:41 JST.
  - The order of B001–B003 against the uploads cannot be proven, because there is no git history.
  - `selected_final` is empty. If nothing is selected, Kaggle auto-picks by public score.
- **FM outputs are not reused (confirmed).** `tabpfn`, `tabicl` and `r251_` appear only in `research/lib_fm.py`, `r250`, `r251` (and r945). `submissions.csv` contains none of them.
- **"Pretrained exclusively on synthetic data" (unverified; overstated in fm.md:62 and :87).**
  - Neither archived model card states its training data. The TabPFN card's sentence "leverages prior-data based learning" (line 11) is not a statement about the data of the `v2_default` checkpoint.
  - From my background knowledge, not from the archive: Prior Labs has also released variants continued-pretrained on real datasets (Real-TabPFN, 2025). So the claim needs an archived source.
  - Fix: archive the pretraining-data statement for the exact checkpoints, or write "pretrained by their authors on prior-generated data [refs]; never trained on this competition's data".
- **TabICLv2 venue (overstated).** The card's BibTeX gives `journal={arXiv preprint arXiv:2602.11139}` (`tabicl_README.txt:32-35`). fm.md:62 and :74 say "ICML 2026".
- **TabPFN attribution.** The card confirms "Apache 2.0 with additional attribution requirement" (`tabpfn_README.txt:21`) and points to the LICENSE at commit `49394b05`. The phrase "Built with PriorLabs-TabPFN" appears only in AI-written text. The requirement is confirmed; the wording is unverified. Archive the LICENSE text.
- **No generator-seed or label recovery (confirmed by code search).**
  - The only `random_state=s` equality check is `research/r151_replica_validation.py:101-108`. It compares the team's `make_classification_with_truth` with sklearn's output for the same seed, never with the real data.
  - A grep for seed-search loops over the real data found nothing.
  - gen.md ("合规") states that this was deliberately not attempted.
  - Recommend a decision: "never attempt to recover the generator's random_state or the hidden labels; replica rows never enter the training set of a submitted model".
- **Private sharing via Colab: not sharing outside the team (compliant, conditional).**
  1. Only the team's own Google accounts can reach the VMs. Google is the compute provider, not a recipient who uses the content.
  2. `ARCHIVE_RULES.md` §8 already allows a "private cloud drive" from the same kind of provider.
  3. Kaggle's own notebooks run on Google infrastructure.
  4. Nothing is published: no public notebook, no Drive share. VMs are stopped with `remote.py down`.
  5. The data upload is the competition's own three CSVs (`colab/remote.py:60,343`). All **45/45** bundles in `logs/code_bundles/` contain no `.csv`, `.npy`, `kaggle.json`, token, `.env`, credential, key or `.pem` file (tar listing; 0 matches).
  6. The same reasoning covers the AI provider (Claude Code read the repository and results). GenAI use is explicitly allowed with disclosure.

  Conditions:
  - (a) **The pro2 account (`project.yaml:21`) must belong to a team member.** The address resembles the user's own, but the archive does not say so.
  - (b) Colab's terms on running several accounts. My recollection of the Colab FAQ is that it disallows using multiple accounts to work around usage limits. This is unverified, and it is a provider-ToS question, not a competition rule.

  Jobs: pro1 40, pro2 28, 1 without an account (J20261022-051802, before the column existed).
- **Kaggle teams (unverified).**
  - Kaggle's private-sharing rule is about the Kaggle team. The archive shows one Kaggle participant (the user, display name only in session memory). `project.yaml: team` is TODO.
  - Until teammates join the competition and merge on Kaggle, giving them the repository, predictions or scores is private sharing outside a Kaggle team.
  - The merger deadline (often about one week before the close) and the rule that a merged team's submission count must not exceed the allowance are not archived. Check the competition timeline.
- **Metadata error (new).** `experiments/20261022-063712_r300_block_l2_c01/config.yaml` and `063713_r300_block_l2_c1` use `fixed_columns` = `lib_ext12.BLOCK` (`research/lib_ext12.py:36`). That is the label-free high-variance rule computed on train+test features. Yet both say `uses_unlabeled_test_features: false`, while the sibling `r300_qda_r01/_r03`, which use the same block, say `true`. Do not count them as strictly supervised in the report. This is a provenance problem, not a label leak (verify_leak).
- **Own-process compliance.**
  - `ARCHIVE_RULES.md` §11 ("every substantive AI contribution goes into ai_usage.md") is not met (§3).
  - VALIDATION_PROTOCOL §5.1 (`docs/VALIDATION_PROTOCOL.md:65`) requires candidates "made from a clean git commit". All **191/191** rows of `registry/experiments.csv` have `git_commit = no-commit`, so no candidate is eligible. Either re-run the finalists on a commit, or amend §5.1 by a decision (D005 argues that code bundles suffice; verify_repro's R910 shows bit-exact Colab re-runs).
  - D001–D003 say "Decided by: TODO". D007 and D009 say "团队确认 TODO". All 14 ideas have proposer "AI (Claude Code)".

### 2. Citations

**Method.** `research/r945_citation_inventory.py` (v2): an AST walk plus a string-import regex over 99 code files in `research/`, `extensions/`, `src/ssc/`, `scripts/` and `colab/`, excluding itself. It matches 40 method patterns and compares them with the 24 BibTeX keys in `registry/references.md` (8 `used`, 16 `candidate`). I then confirmed the important hits by hand (file names are listed in the script output).

**Table 2: third-party modules actually imported.**

| Module | Files | references.md | Action |
|---|---:|---|---|
| sklearn, numpy, pandas, scipy, matplotlib | 52 / 67 / 70 / 35 / 21 | used | ok. Update versions in notes: matplotlib 3.11.2, lightgbm 4.7.0, xgboost 3.2.0 (`requirements-lock.txt`) |
| lightgbm, xgboost | 3 / 2 | candidate | **set used**: runs `r203_lgbm_*`, `r203_xgb_*`, `r205` are reported as negative results (sup.md) |
| tabpfn | 2 (r250, r251) | **missing** | add `hollmann2025`; tabpfn 9.2.0; checkpoint and licence in the note |
| tabicl | 2 | **missing** | add `qu2025` and `qu2026` (arXiv preprint); tabicl 2.2.0; checkpoint `tabicl-classifier-v2-20260212`; BSD-3 |
| torch | 2 | **missing** | add `pytorch`; 2.14.1 (r251 config `versions`) |
| numpyro, jax | 2 / 1 | **missing** | add `numpyro`, `bingham2019` (Pyro, which NumPyro asks to be cited too) and `jax`; numpyro 0.22.0, jax/jaxlib 0.10.2 (r253 config `versions`) |
| joblib, PyYAML, tabulate, threadpoolctl, huggingface_hub | 25 / 28 / 6 / 1 / 1 | software note only | list versions (1.6.0, 6.0.3, 0.10.0, 3.7.0, 2.2.0) in the report's software line, together with Google Colab (Colab CLI) as the compute platform |
| seaborn, statsmodels | 0 | candidate | not imported anywhere: delete or keep as candidate. Do not cite |
| kaggle (API) | code only (`scripts/kaggle_submit.py`, `src/ssc/submission.py`) | candidate | keep candidate. S001–S004 were uploaded through the web page |
| PyMC, Pyro (import) | 0 | — | not used. Cite Pyro only as NumPyro's companion reference |

**Table 3: methods used, with missing references** (all need a new entry with status `used`). Tier 1 = the core report will need it; tier 2 = needed when the exploratory or negative result is mentioned (it will be, in Analysis); tier 3 = analysis tools, cite if the statistic is reported.

| Tier | Key | Where used (evidence) |
|---|---|---|
| 1 | dempster1977 (EM) | `ssc.transductive.TransductiveGMM` = **S003**; `lib_mix.SemiSupEM` |
| 1 | schwarz1978 (BIC) | R101, R150, R254 (choice of K=2 / full covariance, D007) |
| 1 | guyon2003 (`make_classification` design) | the identified generator, R150–R153, `lib_gen_replica.py` |
| 1 | friedman1989 (RDA) | QDA `reg_param`: **S002**, r201, r210, r256 |
| 1 | fan2008 (LIBLINEAR) | liblinear logistic: **S001**, **S004**, E1 pairs |
| 2 | defazio2014 (SAGA) | r200 elastic net (`r200_sup_linear.py:66`) |
| 2 | chang2011 (LIBSVM), platt1999 | `SVC(probability=True)` in r201, r205 |
| 2 | cover1967 (kNN), breiman2001 (RF), geurts2006 (ExtraTrees) | r201 kNN; r203 RF and ET |
| 2 | rasmussen2006 (GP classification) | r202, r255_gpc_ard_hv |
| 2 | yarowsky1995 (self-training), lee2013 (pseudo-label) | r204, r206, r209, r210 self-training; r102_quadlogit_pseudo, r102_selftrain_qda99 |
| 2 | zhou2004 (label spreading), zhu2002 (label propagation) | r204, r206, r209_labelspread_widegrid |
| 2 | nigam2000 (semi-supervised EM) | r102_semisup_eps01, r105 |
| 2 | blei2006 (DP variational mixture) | R254 |
| 2 | bishop2006, murphy2007, gelman2013 | R255/R256 NIW Bayesian QDA, evidence maximisation, Student-t predictive |
| 2 | carvalho2010, piironen2017 (horseshoe, regularised horseshoe) | R253, R257 |
| 2 | hoffman2014 (NUTS), vehtari2021 (rank-normalised R-hat), gelman1992 | R253, R257 (`split_gelman_rubin`, `r257_hs_identifiability.py:69`) |
| 2 | hollmann2025, qu2025, qu2026, pytorch, numpyro, bingham2019, jax | R250, R251, R253, R257 |
| 3 | golub2013 (pivoted QR) | `lib_fm.py:568` (R257) |
| 3 | mardia1970; dagostino1973 | R150, R151, `lib_gen_replica`; `scipy.stats.normaltest` in R002, R150 |
| 3 | pritchard1999, beaumont2002 (rejection ABC) | R153, R156 |
| 3 | wilks1938 (LR χ²) | R100, R102, R150, R922 (`chi2.sf`) |
| optional | castelli1996; chapelle2006; mclachlan2000; scudder1965; massey1951 | "few labels suffice" (R206, R304); transductive framing (D007); original self-training; KS tests (R001, R100) |

**Status corrections (candidate → used).**

| Key | Evidence |
|---|---|
| efron1993 | R103, R153 parametric bootstrap |
| hastie2009 | E3 visible text "Sec. 7.10.2"; protocol §5; also covers LDA, Gaussian NB, polynomial logistic regression and PCA (r003, r200, r201, r205) |
| varma2006, cawley2010 | r205 nested selection; protocol |
| ledoit2004 | LDA shrinkage (r200, r003); QDA-LW (r206–r210) |
| tibshirani1996 | L1 logistic in **S001** |
| lightgbm, xgboost | r203, r205 |
| figueroa2012 | E2 visible text |
| kohavi1995, bengio2004 (E1); ambroise2002 (E3, E4); meinshausen2010 (E4) | set to used when that extension is submitted |

**Entries checked and correct:** nadeau2003 (Mach. Learn. 52(3):239–281) and hanley1982 (Radiology 143(1):29–36), both `used`. Also sklearn, numpy, pandas, scipy, matplotlib, efron1979 and the other 13 method entries (bibliographic data match the standard references).

**Proposed BibTeX** (paste under "## Methods"; versions from run configs):

```bibtex
@article{dempster1977, % status: used (EM; TransductiveGMM = S003)
  author={Dempster, A. P. and Laird, N. M. and Rubin, D. B.}, title={Maximum Likelihood from Incomplete Data via the {EM} Algorithm},
  journal={Journal of the Royal Statistical Society: Series B}, volume={39}, number={1}, pages={1--38}, year={1977}}
@article{schwarz1978, % status: used (BIC, R101/R150/R254)
  author={Schwarz, G.}, title={Estimating the Dimension of a Model}, journal={The Annals of Statistics}, volume={6}, number={2}, pages={461--464}, year={1978}}
@misc{guyon2003, % status: used (generator identified as sklearn make_classification, R150-R153)
  author={Guyon, I.}, title={Design of Experiments for the {NIPS} 2003 Variable Selection Benchmark}, year={2003}}
@article{friedman1989, % status: used (QDA reg_param: S002, r201, r210)
  author={Friedman, J. H.}, title={Regularized Discriminant Analysis}, journal={Journal of the American Statistical Association}, volume={84}, number={405}, pages={165--175}, year={1989}}
@article{fan2008, % status: used (liblinear: S001, S004)
  author={Fan, R.-E. and Chang, K.-W. and Hsieh, C.-J. and Wang, X.-R. and Lin, C.-J.}, title={{LIBLINEAR}: A Library for Large Linear Classification}, journal={Journal of Machine Learning Research}, volume={9}, pages={1871--1874}, year={2008}}
@inproceedings{defazio2014, % status: used (saga, r200)
  author={Defazio, A. and Bach, F. and Lacoste-Julien, S.}, title={{SAGA}: A Fast Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives}, booktitle={Advances in Neural Information Processing Systems 27}, pages={1646--1654}, year={2014}}
@article{chang2011, % status: used (SVC, r201/r205)
  author={Chang, C.-C. and Lin, C.-J.}, title={{LIBSVM}: A Library for Support Vector Machines}, journal={ACM Transactions on Intelligent Systems and Technology}, volume={2}, number={3}, pages={27:1--27:27}, year={2011}}
@incollection{platt1999, % status: used (SVC probability=True, r205)
  author={Platt, J. C.}, title={Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods}, booktitle={Advances in Large Margin Classifiers}, publisher={MIT Press}, pages={61--74}, year={1999}}
@article{cover1967, % status: used (kNN, r201)
  author={Cover, T. M. and Hart, P. E.}, title={Nearest Neighbor Pattern Classification}, journal={IEEE Transactions on Information Theory}, volume={13}, number={1}, pages={21--27}, year={1967}}
@article{breiman2001, % status: used (random forest, r203)
  author={Breiman, L.}, title={Random Forests}, journal={Machine Learning}, volume={45}, number={1}, pages={5--32}, year={2001}}
@article{geurts2006, % status: used (extra trees, r203)
  author={Geurts, P. and Ernst, D. and Wehenkel, L.}, title={Extremely Randomized Trees}, journal={Machine Learning}, volume={63}, number={1}, pages={3--42}, year={2006}}
@book{rasmussen2006, % status: used (GP classification, r202/r255)
  author={Rasmussen, C. E. and Williams, C. K. I.}, title={Gaussian Processes for Machine Learning}, publisher={MIT Press}, year={2006}}
@inproceedings{yarowsky1995, % status: used (self-training, r204/r209/r210)
  author={Yarowsky, D.}, title={Unsupervised Word Sense Disambiguation Rivaling Supervised Methods}, booktitle={Proceedings of the 33rd Annual Meeting of the Association for Computational Linguistics}, pages={189--196}, year={1995}}
@inproceedings{lee2013, % status: used (pseudo-labels, r102_quadlogit_pseudo)
  author={Lee, D.-H.}, title={Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks}, booktitle={ICML 2013 Workshop on Challenges in Representation Learning}, year={2013}}
@inproceedings{zhou2004, % status: used (LabelSpreading, r204/r209)
  author={Zhou, D. and Bousquet, O. and Lal, T. N. and Weston, J. and Sch{\"o}lkopf, B.}, title={Learning with Local and Global Consistency}, booktitle={Advances in Neural Information Processing Systems 16}, pages={321--328}, year={2004}}
@techreport{zhu2002, % status: used (LabelPropagation, r204)
  author={Zhu, X. and Ghahramani, Z.}, title={Learning from Labeled and Unlabeled Data with Label Propagation}, institution={Carnegie Mellon University}, number={CMU-CALD-02-107}, year={2002}}
@article{nigam2000, % status: used (semi-supervised EM, r102_semisup_eps01)
  author={Nigam, K. and McCallum, A. K. and Thrun, S. and Mitchell, T.}, title={Text Classification from Labeled and Unlabeled Documents using {EM}}, journal={Machine Learning}, volume={39}, number={2--3}, pages={103--134}, year={2000}}
@article{blei2006, % status: used (DP-GMM, R254)
  author={Blei, D. M. and Jordan, M. I.}, title={Variational Inference for {D}irichlet Process Mixtures}, journal={Bayesian Analysis}, volume={1}, number={1}, pages={121--143}, year={2006}}
@book{bishop2006, % status: used (evidence approximation, R254/R255)
  author={Bishop, C. M.}, title={Pattern Recognition and Machine Learning}, publisher={Springer}, year={2006}}
@techreport{murphy2007, % status: used (NIW conjugate Bayesian QDA, R255/R256)
  author={Murphy, K. P.}, title={Conjugate {B}ayesian Analysis of the {G}aussian Distribution}, institution={University of British Columbia}, year={2007}}
@book{gelman2013, % status: used (Student-t posterior predictive, R255)
  author={Gelman, A. and Carlin, J. B. and Stern, H. S. and Dunson, D. B. and Vehtari, A. and Rubin, D. B.}, title={Bayesian Data Analysis}, edition={3rd}, publisher={CRC Press}, year={2013}}
@article{carvalho2010, % status: used (horseshoe, R253/R257)
  author={Carvalho, C. M. and Polson, N. G. and Scott, J. G.}, title={The Horseshoe Estimator for Sparse Signals}, journal={Biometrika}, volume={97}, number={2}, pages={465--480}, year={2010}}
@article{piironen2017, % status: used (regularised horseshoe, R253/R257)
  author={Piironen, J. and Vehtari, A.}, title={Sparsity Information and Regularization in the Horseshoe and Other Shrinkage Priors}, journal={Electronic Journal of Statistics}, volume={11}, number={2}, pages={5018--5051}, year={2017}}
@article{hoffman2014, % status: used (NUTS, R253/R257)
  author={Hoffman, M. D. and Gelman, A.}, title={The {No-U-Turn} Sampler: Adaptively Setting Path Lengths in {H}amiltonian {M}onte {C}arlo}, journal={Journal of Machine Learning Research}, volume={15}, pages={1593--1623}, year={2014}}
@article{vehtari2021, % status: used (R-hat, R257)
  author={Vehtari, A. and Gelman, A. and Simpson, D. and Carpenter, B. and B{\"u}rkner, P.-C.}, title={Rank-Normalization, Folding, and Localization: An Improved $\widehat{R}$ for Assessing Convergence of {MCMC}}, journal={Bayesian Analysis}, volume={16}, number={2}, pages={667--718}, year={2021}}
@article{gelman1992, % status: used (split Gelman-Rubin, r257:69)
  author={Gelman, A. and Rubin, D. B.}, title={Inference from Iterative Simulation Using Multiple Sequences}, journal={Statistical Science}, volume={7}, number={4}, pages={457--472}, year={1992}}
@book{golub2013, % status: used (QR with column pivoting, lib_fm.py:568)
  author={Golub, G. H. and Van Loan, C. F.}, title={Matrix Computations}, edition={4th}, publisher={Johns Hopkins University Press}, year={2013}}
@article{mardia1970, % status: used (multivariate normality, R150/R151)
  author={Mardia, K. V.}, title={Measures of Multivariate Skewness and Kurtosis with Applications}, journal={Biometrika}, volume={57}, number={3}, pages={519--530}, year={1970}}
@article{dagostino1973, % status: used (scipy.stats.normaltest, R002/R150)
  author={D'Agostino, R. and Pearson, E. S.}, title={Tests for Departure from Normality. Empirical Results for the Distributions of $b_2$ and $\sqrt{b_1}$}, journal={Biometrika}, volume={60}, number={3}, pages={613--622}, year={1973}}
@article{pritchard1999, % status: used (rejection ABC, R153/R156)
  author={Pritchard, J. K. and Seielstad, M. T. and Perez-Lezaun, A. and Feldman, M. W.}, title={Population Growth of Human {Y} Chromosomes: A Study of {Y} Chromosome Microsatellites}, journal={Molecular Biology and Evolution}, volume={16}, number={12}, pages={1791--1798}, year={1999}}
@article{beaumont2002, % status: used (ABC, R156)
  author={Beaumont, M. A. and Zhang, W. and Balding, D. J.}, title={Approximate {B}ayesian Computation in Population Genetics}, journal={Genetics}, volume={162}, number={4}, pages={2025--2035}, year={2002}}
@article{wilks1938, % status: used (likelihood-ratio chi-square, R100/R150)
  author={Wilks, S. S.}, title={The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses}, journal={The Annals of Mathematical Statistics}, volume={9}, number={1}, pages={60--62}, year={1938}}
```

and under "## Software":

```bibtex
@article{hollmann2025, % status: used (analysed comparison only, R250/R251)
  author={Hollmann, N. and M{\"u}ller, S. and Purucker, L. and Krishnakumar, A. and K{\"o}rfer, M. and Hoo, S. B. and Schirrmeister, R. T. and Hutter, F.},
  title={Accurate Predictions on Small Data with a Tabular Foundation Model}, journal={Nature}, volume={637}, pages={319--326}, year={2025}, doi={10.1038/s41586-024-08328-6},
  note={tabpfn 9.2.0; weights tabpfn-v2-classifier-finetuned-zk73skhh.ckpt (= v2\_default), huggingface.co/Prior-Labs/TabPFN-v2-clf, accessed 2026-10-22; Prior Labs License (Apache-2.0 with attribution)}}
@inproceedings{qu2025, % status: used (R250/R251)
  author={Qu, J. and Holzm{\"u}ller, D. and Varoquaux, G. and Le Morvan, M.}, title={{TabICL}: A Tabular Foundation Model for In-Context Learning on Large Data}, booktitle={International Conference on Machine Learning}, year={2025}}
@misc{qu2026, % status: used (R250/R251)
  author={Qu, J. and Holzm{\"u}ller, D. and Varoquaux, G. and Le Morvan, M.}, title={{TabICLv2}: A Better, Faster, Scalable, and Open Tabular Foundation Model}, howpublished={arXiv preprint arXiv:2602.11139}, year={2026},
  note={tabicl 2.2.0, checkpoint tabicl-classifier-v2-20260212, huggingface.co/jingang/TabICL, BSD-3-Clause}}
@inproceedings{pytorch, % status: used (backend of TabPFN/TabICL)
  author={Paszke, A. and Gross, S. and Massa, F. and others}, title={{PyTorch}: An Imperative Style, High-Performance Deep Learning Library}, booktitle={Advances in Neural Information Processing Systems 32}, pages={8024--8035}, year={2019}, note={torch 2.14.1}}
@misc{numpyro, % status: used (horseshoe NUTS, R253/R257)
  author={Phan, D. and Pradhan, N. and Jankowiak, M.}, title={Composable Effects for Flexible and Accelerated Probabilistic Programming in {NumPyro}}, howpublished={arXiv:1912.11554}, year={2019}, note={numpyro 0.22.0}}
@article{bingham2019, % status: used (cited with NumPyro)
  author={Bingham, E. and Chen, J. P. and Jankowiak, M. and Obermeyer, F. and Pradhan, N. and Karaletsos, T. and Singh, R. and Szerlip, P. and Horsfall, P. and Goodman, N. D.}, title={Pyro: Deep Universal Probabilistic Programming}, journal={Journal of Machine Learning Research}, volume={20}, number={28}, pages={1--6}, year={2019}}
@misc{jax, % status: used (NumPyro backend)
  author={Bradbury, J. and Frostig, R. and Hawkins, P. and Johnson, M. J. and Leary, C. and Maclaurin, D. and Necula, G. and Paszke, A. and VanderPlas, J. and Wanderman-Milne, S. and Zhang, Q.}, title={{JAX}: Composable Transformations of {P}ython+{N}um{P}y Programs}, howpublished={\url{https://github.com/jax-ml/jax}}, year={2018}, note={jax/jaxlib 0.10.2}}
```

Also add the two model cards and the Prior Labs licence page to the "Tutorials, posts" table, with URL and access date 2026-10-22. That table is empty now.

**Presentation problems.**

- **Citations hidden in comments.** E1's visible text cites only Nadeau-Bengio. Kohavi, Bengio & Grandvalet, Hanley & McNeil, Friedman and Pedregosa appear only inside `<!-- -->`. E2 visibly cites only Figueroa; Efron, Dempster, Friedman, Pedregosa and Virtanen are in comments (checked by stripping comments). pandoc and the md2tex route both drop comments.
- **Page budget.** About 50 `used` entries do not fit the 0.4-page budget in `report_core.md`. Options:
  - use a numbered compact style;
  - cite Hastie 2009 for textbook methods (LDA, naive Bayes, logistic regression, PCA, kNN), plus sklearn for the implementations;
  - put the exploratory-run software and methods in one dense line.

  An exploratory run is still "used", so the foundation models and NumPyro cannot be dropped silently.

### 3. GenAI disclosure (`registry/ai_usage.md`)

Problems:

1. **False draft sentence** (`ai_usage.md:11`): "All experiments, results, interpretations and the final text were produced and checked by the team; every member has reviewed the code they rely on."
   - All 191 runs have author `TODO_ID_1` and were designed and run by AI agents (rows 2–3; journal 2026-10-22).
   - D007 says "AI 提出", D009 says "Decided by: AI", and all 14 ideas have proposer "AI (Claude Code)".
   - Submitting this sentence would be a false disclosure. **Severity: high.**
2. **Broken Markdown.** Line 11 (blockquote) is immediately followed by line 12 with no blank line, so rows 2–3 become a lazy continuation of the quote and are not rendered as a table. Move the draft paragraph below the table.
3. **"Member (operator)" and "Verified by" are TODO in every row.** §G requires that members can explain every detail; this column is where that is evidenced.
4. **Not logged** (ARCHIVE_RULES §11):
   - (a) the upload channel: Claude Code drove the Kaggle web page in the member's logged-in Chrome (session memory; no API token configured);
   - (b) S004: pre-registration B005, upload at 16:08 JST, E3 `--refresh-lb` (`reports/tables/e3_leaderboard.csv`, 16:08);
   - (c) registry authorship: decisions D005–D009 (including the rule interpretation D007 and the rule-4 position on foundation models, I014), beliefs B001–B005, ideas I010–I014, the journal entries;
   - (d) `scripts/archive_check.py`, the Conclusions backfilled for R001–R003, the snapshot zip;
   - (e) this verify round (`research/r900–r945`, `research/reports/verify_*.md`);
   - (f) the "Text for the report" passages in `research/reports/*.md`, if they are reused;
   - (g) the figure palette chosen by an AI design skill (E1–E4 figures);
   - (h) future uses: report text, polishing, video script and slides, any help with reflections.
5. **No statement of how outputs were verified.** And one tempting claim would be wrong: `tests/` has 107 test functions, but **none imports `ssc.transductive`**, so the final model (S003) is not unit-tested. Its evidence is:
   - bit-exact re-runs on a fresh Colab VM (R910/R911: S003, E3 and r201);
   - permutation nulls with positive controls (R900);
   - replica datasets with known truth (R103, R152, R153).
6. **Extensions are marked individually, but their AI notes are invisible.** "Drafted by Claude Code (agent ext12)" (E1:12, E2:12) and "Draft by ext34 (Claude Code)" (E3:9, E4:11) sit inside HTML comments.

Proposed rows (append after fixing the table; operator and verifier are for the team to fill, and only with what they actually did):

```
| 2026-10-22 | <member> | Claude Code (Claude Opus 5.5) driving the Kaggle web page in the member's logged-in browser | Uploaded S001-S004 on the member's account after the member authorised autonomous submissions (max 3/day, each pre-registered with expected_public and a belief); recorded public scores; refreshed the E3 leaderboard table | registry/submissions.csv, registry/beliefs.md B001-B005, reports/tables/e3_leaderboard.csv | <member> |
| 2026-10-22 | <member> | Claude Code | Wrote registry entries: decisions D005-D009 (incl. rule interpretation D007 and the foundation-model / rule-4 position), beliefs B001-B005, ideas I010-I014, journal entries | registry/decisions.md, beliefs.md, ideas.md, journal.md | team (D007/D009 confirmation pending) |
| 2026-10-22 | <member> | Claude Code | Archive tooling: scripts/archive_check.py; Conclusions of R001-R003; snapshot backup | scripts/archive_check.py, experiments/2026102205*/notes.md | <member> |
| 2026-10-22 | <member> | Claude Code (verify workflow, 6 lenses) | Independent re-checks of round-1 claims: reproduction, leakage nulls, generator, extensions, rule compliance, citations | research/r900-r945*.py, research/reports/verify_*.md | <member> |
| <date> | <owner of E#> | Claude Code | E# draft text, tables and figure styling (figure colours from an AI design-skill palette); the owner re-ran and rewrote it | reports/extensions/e#_*.md, reports/figures/e#_* | <owner> |
```

Proposed report paragraph (keep it true; delete any clause the team has not done):

> **Use of generative AI.** We used Claude Code (Anthropic; model Claude Opus 5.5), an agentic coding assistant with read access to our repository and the competition data, and with control of our Google Colab sessions. It set up the environment and the experiment-tracking package. It designed, coded and ran most experiments on Colab (runs R001–R321 and extensions E1–E4), including the generator identification, the transductive Gaussian mixture and the Bayesian and foundation-model comparisons. With our authorisation and after pre-registering each expected score, it uploaded our Kaggle submissions S001–S004 through the Kaggle web page on our account. It also drafted internal documentation, registry entries, the extension drafts and parts of this report. Team members set the scope, authorised every submission, made the decisions recorded in `registry/decisions.md`, and [re-ran and rewrote] the parts they own (see the contribution table). We checked AI output by exact re-runs on separate Colab machines (S003, E3 and r201 reproduced bit-for-bit, R910), by label-permutation tests with positive controls (R900), by replica datasets with known ground truth (R103, R152, R153), by unit tests of the shared validation and tracking code (`tests/`), and by a separate AI verification pass. That last pass does not replace our own review: each member has read, and can explain, the code and analyses in the parts they own.

Each owned extension should also carry one visible line under the author line, for example: "AI use: drafted with Claude Code; I re-ran the experiment and rewrote the text (see the core report's AI disclosure)."

### 4. Deliverable readiness (assignment §A–F)

**Table 4: what exists, what is missing, who must do it.**

| Deliverable | Exists | Missing / risk | Only the team can do |
|---|---|---|---|
| Core report ≤ 6 pp (`reports/report_core.md`) | Template with page budget; source passages in `research/reports/{gen,fm,mix,sup}.md` ("Text for the report") | **All 7 sections TODO (0 words).** Material is far beyond 6 pages and must be scoped to what members can explain | choose scope; write or approve text; names in the title block |
| 4 extension drafts ≤ 2 pp each | All 4 have the exact four headings and a table or figure | Measured 2.16–2.71 pp (verify_ext §1); cut about 400 words in E1/E2 and about 250 in E3/E4, and drop or merge one E4 table. Author "TODO" in all 4. E3 text still says "public LB score is pending" (`e3_leakage.md:66`), although S004 = 0.79136. E1/E2 citations only in comments. AI notes invisible | decide owners (2–3 members means at most 2–3 graded extensions; the user's "all four" plan puts the rest into the core report as support). Each owner re-runs, rewrites, names themself |
| Contribution table (1 p) | Template | All TODO. Author columns cannot serve as evidence: 191/191 runs and S001–S004 have `TODO_ID_1` | names, student IDs, Kaggle display names, an honest mapping (add an "AI assistance" column) |
| Video ≤ 5 min, all faces, ≤ 300 MB | `video/PLAN.md` | No script or slides; ffmpeg not installed (Clipchamp is the fallback); language TODO | record with every member on camera |
| Code reproducing the final submission | `src/ssc`, `scripts`, `research`, `requirements-lock.txt` | README step uses `configs\TODO_final.yaml` (`README.md:104`), but **no config route builds TransductiveGMM** (`src/ssc/models.py` has no builder). S003 reproduces only via `research/r003_transductive_vs_supervised.py`, which R910 shows is bit-exact on Colab. `logs/code_bundles/` is excluded from the zip. Lock file lacks the FM / NumPyro packages (versions are only in run configs) | git identity, first commit, private remote |
| Per-member reflection PDF | `reports/reflections/TEMPLATE.md` | Deadline and length TODO. **Q4 has no material:** every belief is AI-authored, and B001–B003 and B005 were confirmed. B004 (private ≥ 0.999 at 70%) has P = 0.14–0.51 under R922's priors (`reports/tables/r922_posterior_by_prior.md`), so it will probably be refuted, but it is not a member's own belief. Q2: all ideas are AI-proposed. Q3: all journal difficulties are AI's | write own beliefs before the next uploads; own journal lines, ideas and reflection |
| Canvas checkpoint (Kaggle display names, team fixed) | — | Date unknown. Team registration (deadline 10-16) is unconfirmed. The user joined the competition only on 10-22 | members only |
| Rules text, instructor answers | — | Not archived (§1) | members only |
| Zip | `scripts/package_deliverables.py` | `team.name` TODO blocks `--build`. The zip would include personal email addresses in `project.yaml:20-21` and `colab/README.md:35-36`. I could not run the dry run: the permission classifier blocked it | remove the addresses; run the dry run |

**Prioritised TODO.** All dates JST; the final deadline is 2026-11-29 23:59 HKT = 2026-11-30 00:59 JST. T = team only; A→T = AI can draft, the team checks and owns.

P0, this week:

1. **10-23 (T)** Canvas:
   - team registration status (deadline 10-16);
   - checkpoint date and the Kaggle display names it requires;
   - reflection deadline and length;
   - video language;
   - any GenAI-disclosure format.

   Update `docs/ASSIGNMENT_CHECKLIST.md` and fill `project.yaml: team`. If the checkpoint is earlier than 10-25, do it first.
2. **10-23 (T)** Teammates join the competition and accept the rules, and the team merges on Kaggle (check the merger deadline on the Overview timeline). No sharing of the repository, predictions or scores before the merge.
3. **10-24 (T)** Archive the Kaggle Rules tab and the Data-tab licence as `docs/competition_rules_2026-10-24.md`, plus the TabPFN LICENSE text (commit `49394b05`).
4. **10-24 (T)** Ask the instructor privately, in generic terms (no scores or methods):
   - (a) may unlabeled test features be used in unsupervised steps (scaling, clustering, mixture models)?
   - (b) are pretrained tabular foundation models allowed?

   Record the answers as D010.
5. **10-24 (T)** Set the git identity, make the first commit, push to a private remote (D004, §9). Confirm who owns pro2.
6. **10-25 (A→T)**
   - Fix `ai_usage.md`: table, false sentence, rows from §3.
   - Fix `references.md`: 39 entries and 13 status changes from §2. Re-run `research/r945_citation_inventory.py`.
   - Confirm D007/D009 and fill "Decided by" for D001–D003.
   - Add the decision "no seed or label recovery; replica rows never train a submitted model".
   - Correct `fm.md`'s "exclusively synthetic" and "ICML 2026" before any text is reused.
7. **10-23 (main session)** Replace "pending" in `e3_leakage.md:66` with S004 = 0.79136 (honest CV 0.793, leaky 0.867). Pre-register (belief + `expected_public`) and upload `20261022-071234_r201_qda_block` as the strictly inductive hedge. The Kaggle day of 10-22 UTC has 2 uploads left within the team rule, and it resets at 09:00 JST.

P1, by 11-01:

8. **from now (T)** Before each further upload, each member writes at least 2 own beliefs in `beliefs.md`, plus own journal difficulties and ideas. This is the material for reflection Q2–Q4.
9. **by 11-01 (T)** Decide extension owners (at most one per member). Each owner:
   - reads the code and re-runs the extension from a clean commit;
   - rewrites it in their own words, with visible citations and a visible AI line;
   - checks the 2-page fit by a trial export (`xelatex` is installed; pandoc is not).
10. **by 11-01 (T+A)** Write a decision freezing D002 with the E1 evidence, and fill `TODO(E1)` in `docs/VALIDATION_PROTOCOL.md`.

P2, by 11-15 (hard deadline 11-22):

11. **by 11-15 (T+A)** Final-selection rule decision:
    - If D010 allows test features (or no answer): select S003 plus the strictly inductive QDA submission.
    - If D010 forbids them: select two strictly inductive submissions, and never S002 or S003.
    - In every case, never an FM run.
    - Reconcile D008 ("CV, replica simulation and BIC") with VALIDATION_PROTOCOL §5.
    - Resolve the §5.1 clean-commit criterion: re-run the finalists on a commit, or amend §5.1.
12. **by 11-15 (A→T)** A reproduction path for the final submission: either a config builder for TransductiveGMM, or the `research/` script route documented in the README with the expected sha256 (`3aae140e5837753f` for S003). Test it on a clean checkout. Add the FM/NumPyro versions to a note, or state that they are not needed to reproduce the final.
13. **by 11-22 (T+A)** Core report §1–§3, the contribution table with an AI-assistance column, and the video script (about 600 words).
14. **ongoing (T)** "Explain-every-detail" sessions. Each member must be able to explain:
    - the official CV;
    - the D007 argument;
    - GMM/EM and QDA;
    - the generator identification;
    - LB noise (Hanley–McNeil);
    - their own extension.

    Anything nobody can explain leaves the report.

P3, 11-23 to 11-29:

15. **11-23 to 11-25 (T)** Apply the rule mechanically, select the 2 finals on Kaggle, run `record_lb.py --final yes`, write the decision entry.
16. **11-24 to 11-25 (T+A)** Report §4–§5, the AI disclosure, the references, page checks (`reports/README.md`).
17. **11-25 to 11-26 (T)** Record the video: all faces, ≤ 5:00, ≤ 300 MB (`ffprobe`, or the file properties if ffmpeg is not installed).
18. **11-27 (T)** Remove personal email addresses from the zipped files, run the `package_deliverables.py` dry run, draft the reflections.
19. **11-28 (T)** Final `reports/report.pdf`, a zip named after the team, the Canvas upload, and each member's reflection PDF.
20. **11-29** Buffer only. Canvas and Kaggle close at 23:59 HKT (00:59 JST on 11-30).

### 5. Submission advice (rules lens)

- **Next upload:** `20261022-071234_r201_qda_block` (CV 0.99990; block re-derived in each training fold; `uses_unlabeled_test_features: false`; reproduced bit-for-bit in R910 / `20261022-091224_r910_r201_qda_block`).
  - It is compliant under any reading of rule 4, and it must be on Kaggle before it can be selected as a final.
  - `20261022-082047_r210_qda_block_nll` (CV 0.99998) is the alternative. It is equally compliant. Its test_pred was reproduced bit-identically by an independent re-implementation (r932, `20261022-092058_r932_real_repro_agreement`) and its CV by R900 (`reports/tables/r900_perm_null_A_summary.csv`, real − archived = 0). Leave the model choice between the two to the candidate verifier.
- **Final two:**
  - If D010 allows test features (or there is no answer): S003 plus the strictly inductive QDA.
  - If D010 forbids test features: two strictly inductive submissions, and neither S002 nor S003.
  - Never r251 (TabPFN/TabICL).
- **Do not select** S002 or S004 as finals (dominated). S004 is pre-registered as "not a final candidate".

### 6. What I could not verify

- The exact Kaggle and course rule text, because neither is archived.
- The instructor's view on transductive use and on foundation models.
- The pretraining data of the two foundation-model checkpoints, and the exact TabPFN attribution wording.
- Colab's multi-account terms.
- Who owns pro2.
- Whether a Kaggle team or a Canvas team registration exists, and the Canvas dates.
- Whether B001–B003 were written before the uploads (there is no git history).
- The output of `scripts/package_deliverables.py`, because the dry run was blocked by the permission system.
- Real page counts in Word (verify_ext's XeLaTeX measurement is the best available).
