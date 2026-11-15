# verify：第 1 轮核查汇总（6 个核查方向合并）

- 编辑：AI 汇总代理（Claude Code，Claude Opus 5.5），2026-10-22 约 17:55–18:30 JST。
- 输入：6 份核查报告 `research/reports/verify_{leak,repro,gen,cand,ext,rules}.md` 及其结构化结果；第 1 轮探索结果 `round1_explore.json`（scratchpad）和 `research/reports/{gen,mix,sup,fm,ext12,ext34}.md`。
- 方法：只读。没有跑 Colab，没有建 Run，没有上传 Kaggle，除本文件外没有改任何文件。有争议的点我在档案里亲自抽查过（清单见英文 §1.2）。

---

## 中文摘要（给队友）

**一句话**：主力方案 S003 没有问题：没有标签泄漏，能逐位复现，离理论上限只差约 2.6e-7。被推翻或说过头的主要有三类：
1. 对标签噪声和 private 分数的估计过于乐观；
2. D009 的"不可区分"措辞没有信息量；
3. 四份扩展草稿的数字和篇幅有问题。

规则方面最大的风险没有变：**用测试集特征（D007）还没有被规则原文或讲师确认**。

**站得住的结论**
- **数据生成方式确定了，证据比第 1 轮更强**：`make_classification`，12 个信息列、8 个冗余列、130 个噪声列，再做逐列仿射变换。gen 核查者用独立代码、不看标签，重新得到同一组信息列（领先 13 nats，24/24 个已知真值复刻全对）；每类 2 簇和 hypercube=False 都被排除。
- **没有标签泄漏**：leak 核查者做了代码审查和两种置换检验（带阳性对照），S003、r155、r201、r210、r209、r255 和 E3 诚实流程全部通过。id、行序、重复行也都没有问题。
- **能复现**：S003、E3 最终模型、r201 在新的 Colab VM 上逐位复现；r209、r210 也被独立重写的代码逐位复现。S001–S004 的提交文件格式正确，每次上传前都做了预登记。
- **S003 离贝叶斯上限只差约 2.6e-7**：差距是真实的，不是模拟噪声，但实际上看不出来，约等于 9,400 万对样本里排反 24 对。8 位小数造成的大量并列也不损失 AUC，不用重交。
- **没有任何候选在 private 上能显著超过 S003**（222 个复刻）。
- **E3 的两条流程符合题目要求**：S004 的 public 是 0.79136，几乎等于诚实 CV（0.793），比泄漏 CV（0.867）低 0.075。

**被推翻或说过头的**
1. **标签噪声和 private 预期**：
   - "flip_y = 0 几乎不可能"说过头了，实际只是不太可能，约 5:1。
   - flip_y 的合理范围是 0–0.015，sklearn 默认值 0.01 完全可能。
   - 所以 S003 的 private 更可能在 **0.997 左右（90% 区间 0.992–0.9996）**，不是 0.998。B004（"private ≥ 0.999"，70%）实际只有约 15% 的概率。
   - mix 说"private 榜 SE 约 1e-4"也不对：有噪声时 sd 是 3–6e-4。
2. **D009 的"调好的监督 QDA 与 GMM 在 CV 上不可区分"**：字面上对，但 CV 在这个分数段根本没有分辨力。222 个复刻里检验 222 次都不显著，而监督 QDA 的真实 AUC 每次都更低（低 1.7–2.6e-4）。另外 r210 的 reg 0.001 是网格下边界，不是最优点。
3. **四份扩展草稿**：
   - 篇幅：每份排版后 2.2–2.7 页，都超过 2 页。
   - 数字：15 处和档案对不上。
   - E2：外推区间名义 90%，实际只覆盖 21%。
   - E1、E4：结论措辞比证据强。
   - E3：还写着"LB pending"，已过时。
4. **文书**：`ai_usage.md` 里有一句与事实不符的话（"所有实验都由团队完成并检查"）；引用缺 39 条；README 的复现路径走不通；191 个 run 没有一个来自 git commit；单元测试没有覆盖最终模型。

**这改变了什么**
1. **今天的提交**：E3 已在 16:08 JST 作为 S004 上传，所以今天（Kaggle 日到 10-23 09:00 JST）按队规还剩 2 次。建议都用掉，先预登记：
   - r201（严格监督 QDA），作为"规则保险"；
   - r209（自训练 QDA），作为期望值意义上最好的第二选择。
2. **最终两个选择（我的裁决）**：
   - 默认是 **S003 + r201**。r209 和 S003 一样用了测试集特征，对冲不了规则风险；它比 r201 只多约 1e-5 的期望 AUC，约为 private 波动的 1/40。
   - 只有讲师明确允许用测试集特征时，才换成 S003 + r209。
   - 如果讲师不允许，S002、S003 都不能选，两个选择都要从严格监督的模型里挑。
3. **报告措辞**：S003 的 private 写成约 0.997；D009 改成"CV 饱和，分辨不出"；"有标签的设计是在看过标签后定的"要写一句；监督模型的水平写成 0.9995–0.9999（扣掉 587 个配置的挑选效应）。
4. **Beliefs**：B004 不改（只追加原则），另登记一条新的。

**核查者之间的分歧（详见英文 §3）**：
- private 预期：gen 对，cand 用的先验在 0.01 截断了。
- 第二个最终选择：rules 对默认情形，cand 对"讲师允许"的情形。
- E3 的 ±0.014：leak 说"不是 SE"是对的，但它建议的 SE 0.003 不对。我按 Nadeau–Bengio 校正重算，SE 约 0.020。
- rules 说"r210 没有被独立重跑"：已过时，cand 的 r932 逐位复现过。
- 扩展页数：以 ext 的实测为准。

**最紧急**：
1. 主会话今天预登记并上传 r201 和 r209，同时把"最终选择规则"写进 decisions.md，必须在看到这两个分数之前写；
2. 队员存档比赛规则原文，并私下问讲师 D007 的问题；
3. 修 `ai_usage.md` 那句话；
4. 各扩展负责人按英文 §4 改数字、删字数。

---

## English

### 1. Sources and method

#### 1.1 Inputs

| Verifier | Report | New runs (all `kind=analysis`, Colab) | Lens |
|---|---|---|---|
| leak | `research/reports/verify_leak.md` | R900 Null A `20261022-091506`, Null B `20261022-092943`, R901 `20261022-094812` | label leakage, id/order/duplicates |
| repro | `research/reports/verify_repro.md` | R910 (`20261022-091211/091223/091224/091403`), R911 `20261022-091852`, R912 `20261022-092445`, R913 `20261022-093023` | bit-exact re-runs, submission integrity, environment |
| gen | `research/reports/verify_gen.md` | R920 `20261022-100551`, R921 `20261022-094951`, R922 `20261022-092146` | generator, Bayes regret, label noise, private distribution |
| cand | `research/reports/verify_cand.md` | R930–R936 (pooled `20261022-172146_r936_final_selection`, 222 + 136 replicas) | candidates' private distribution, D009, final selection |
| ext | `research/reports/verify_ext.md` | R940–R944 (`20261022-091211_r940`, `092201_r941`, `094142_r942`, `100104_r943`, `161150_r944`) | E1–E4 drafts |
| rules | `research/reports/verify_rules.md` (v2) | none (read-only; `research/r945_citation_inventory.py`) | rule compliance, citations, disclosure, deliverables |

All Colab sessions used by the verifiers (pro2/vleak, pro1/vrepro, pro1/vgen, pro2/vcand, pro2/vcand2, pro1/vext) were reported stopped.

#### 1.2 My own spot checks (local, read-only)

| Point | What I read | Finding | Bears on |
|---|---|---|---|
| E3 "0.073 ± 0.014" | `experiments/20261022-063908_e3_leakage/cells.csv` (k_feat 20, 5-fold) | gap_mean 0.0733, gap_std 0.0143 (SD over the 20 repeats), corrected_t 3.72, corrected_p 0.0003. So the Nadeau–Bengio SE = 0.0733 / 3.72 = **0.0197**; the naive SE 0.0143/√20 = 0.0032 measures only split noise *given these 250 rows*. Same pattern at every k_feat (e.g. k=30: SD 0.014, corrected SE 0.023). Table 1 of `reports/extensions/e3_leakage.md` labels the column "gap +/- SD" correctly; the prose at line 72 does not. | leak's correction (§3 C5) |
| Log-loss comparison | `summary.json` of r209/r210/r201/r255 | r209 `oof_logloss` 0.00709, r210 0.00869, r201 0.01464; r255 reports `cv_logloss` (fold mean) 0.00937, no OOF value | cand is right: "r255 lowest log-loss" compares different metrics |
| r300 block flag | `experiments/20261022-063713_r300_block_l2_c1/config.yaml:37`, `research/lib_ext12.py:36` | `uses_unlabeled_test_features: false` while the model uses the hard-coded label-free BLOCK (derived on train+test) | rules is right |
| Unit tests of the final model | `grep transductive tests/*.py` | no match | rules is right |
| E3 LB status | `reports/extensions/e3_leakage.md:66` | still "public LB score is pending"; `registry/submissions.csv` S004 = 0.79136 at 2026-10-22T07:08:41Z | ext and rules are right |
| r209 / r210 reproduction | `experiments/20261022-092058_r932_real_repro_agreement/notes.md` | r931 re-implementation reproduces S003, r209, r210, r201 test_pred bit-identically and r201/r210 reg choices in 100/100 folds | corrects rules' "r210 not independently re-run" |
| sup.md disclosures | `research/reports/sup.md:96, :156` | sup already flags r209 `*_r003` (reg from R201), r210's second-level metric choice, and reg 0.001 as "the grid bottom" | leak's and cand's corrections hit the headline sentences (sup.md:42, :52, :104), not the limitations section |
| "class_sep 0.6–0.9 fit equally well" | `research/reports/gen.md:21, :51, :62`, run notes | no such statement exists; gen.md says 0.71 (0.61–0.81) | gen's "refuted" verdict on that sub-claim targets a strawman; only the interval width changes |
| Candidate files | `test_pred.npy` of r209, r201, S003 | 19,750 finite values in [0, 1] each. Distinct values: r209 16,716 raw → 4,638 at `%.8f`; r201 18,332 → 7,724 (largest tie 5,157). Spearman with S003 on S003's 204 boundary rows: r209 0.698, r201 0.451 | cand/sup numbers reproduced |
| Public predictions | `reports/tables/r936_public_abc.md` (tn0, csv8, window ±1.5e-4) | r209 0.99913, predictive sd 0.00033, P(higher than S003) 0.28, P(equal) 0.04; r201 0.99884, sd 0.00066, P(higher) 0.20 | submission plan (§5) |
| Private posterior | `reports/tables/r922_posterior_by_prior.md` | flat prior: mean 0.99684, median 0.99733, 90% [0.99230, 0.99959], P(≥0.999) 0.163; gen's 4-point grid: 0.99777 | §3 C1 |
| Upload count today | `registry/submissions.csv` | Kaggle day 2026-10-22 00:00 UTC (= 09:00 JST) → S004 only, so 1 of the team's 3 slots is used | §5 |
| Hanley–McNeil SE, 395 public rows (n1 ≈ 198, n0 ≈ 197) | computed | AUC 0.99913 → 0.0015; 0.99884 → 0.0017; 0.82 → 0.021; 0.79 → 0.023 | §5 |

### 2. Claims table (for the course report)

Verdicts: confirmed / overstated / refuted / unverified. "Consequence" says what the report or the plan should do.

**A. Data and generator**

| # | Claim (round 1) | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| A1 | Data = `make_classification(n_informative=12, n_redundant=8, n_repeated=0)` + per-column affine map; no other split fits | confirmed, stronger than stated | R920: λ₁₃…λ₂₀ ≈ 2.5e-14 (rounding floor), the 8 dependency residual sds are 0.992–1.004 × the 6-decimal rounding prediction | State it as a deterministic finding, not a best fit |
| A2 | Informative and redundant sets were identified without labels | confirmed | R920: an independent Wishart profile likelihood picks the same 12 columns (margin 13.0 nats, 20/20 bootstraps, 24/24 known-truth replicas); `identify_structure(X)` takes no labels (verify_leak §1); col 61 is in the set although its |corr| rank is 139/150 (R901) | Caveat for the report: exactness alone does not identify the roles (all 125,970 12-subsets are full rank) |
| A3 | n_clusters_per_class = 1, hypercube = True | confirmed | R920: LR(K4 vs K2) 480 vs 1.2–1.8e5 under two clusters per class; hypercube favoured ≈ 360:1 | – |
| A4 | class_sep ≈ 0.71 (95% 0.61–0.81) | confirmed, interval conservative | R920 MLE 0.703, LR 95% [0.639, 0.760], replica-calibrated width ≈ ±0.075 | Quote 0.70 (≈ 0.63–0.78) |
| A5 | 130 noise columns carry no class signal; train is a random subset; no id/order/duplicate signal | confirmed | R920 (KS p 0.31/0.58); R901 (AUC(id, y) 0.552, p 0.16; 0 exact duplicates; adversarial AUC 0.499) | One sentence in the leakage audit |

**B. Bayes limit, label noise, private score**

| # | Claim | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| B1 | S003 is ~2–3e-7 below the Bayes AUC | confirmed (real, not MC noise) | R921, 1000 parametric replicas: expected-AUC regret 2.57e-7 ± 0.018e-7, positive in 100%; ≈ 24 of 9.4e7 private pairs | Quote "≈ 2.6e-7 on replicas fitted to our data". On generic make_classification draws the block misses an informative direction in 5.5% (regret 5.4e-4); the real block spans all 12 dims, so this does not apply (R921, also seen by repro R912) |
| B2 | Bayes private AUC 0.99995 clean, ≈ 1 − flip_y/2 with noise | confirmed | R921, R922 | – |
| B3 | 8-decimal CSV ties cost no AUC | confirmed | R912/R913: csv8 − raw = 0 clean, +5e-6 ± 1e-5 with noise; per-file noise ≤ 1.5–3e-4 | No resubmission. Writing ranks is optional |
| B4 | No train label contradicts a confident posterior | confirmed | R922: 0 rows with P(observed) < 0.2; row 9 (y = 0, π = 0.79) is a boundary row | Justifies cand's "train labels clean" replica design |
| B5 | Public 0.99923 is "nearly impossible" at flip_y = 0; P(flip_y > 0) = 0.92 | overstated | R922 combined LR(f = 0) = 0.20 (0.03–0.84 under ±20% posterior temperature) | "flip_y = 0 is disfavoured about 5:1, not excluded" |
| B6 | flip_y ≈ 0.0044; wrong-label rate 0.21% (0.05–0.4%) | overstated | R922: MLE 0.003, 1/8-support [0, 0.0147]; LR(0.01) = 0.37; flat prior → 0.31% (0.05–0.77%) | Report the likelihood interval, not a grid posterior |
| B7 | Expected private AUC of S003 ≈ 0.9977–0.998 | overstated | R922 flat prior: mean 0.9968, median 0.9973, 90% [0.9923, 0.9996]; 0.9977 comes only from the 4-point grid prior truncated at 0.01 | Report "≈ 0.997 (90% 0.992–0.9996); private ≈ 0.002 below public is expected" |
| B8 | B004: S003 private ≥ 0.999 and \|pub − priv\| ≤ 0.001 (70%) | refuted (miscalibrated) | R922: P = 0.14–0.17 under the reasonable priors (0.51 only log-uniform) | Leave B004 untouched (append-only); register a new belief (§5) |
| B9 | Private-LB SE ≈ 1e-4 (mix) | refuted | R922: sd 1.1e-5 clean; 2.4e-4 / 3.9e-4 / 5.5e-4 at flip 0.002 / 0.005 / 0.01 | "sd 3–6e-4 at plausible noise". mix's conclusion (boundary refinements undetectable) becomes stronger |

**C. Candidates and D009**

| # | Claim | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| C1 | r201/r210 nested CV is honest: inner CV inside the outer training fold, block re-derived in-fold, no test features | confirmed | leak: 100/100 `fold_fit_info` SHA1 keys match the official outer folds; R900 nulls at chance; cand r932 reproduction | Can be cited as strictly inductive |
| C2 | CV: r210 0.99998, r201 0.99990, r209 1.0, r255 1.0 | confirmed | archived = recomputed (R900 true-label rows; cand on 2 VMs; R910 for r201) | – |
| C3 | "Best supervised model CV 0.99998" | overstated (winner's curse) | 587 configs (588 with the GMM row); selection-aware r205 = 0.99955 ± 0.0006 | Quote "0.9995–0.9999 after model selection" |
| C4 | r210's reg 0.001 is the inner log-loss optimum | refuted (grid floor) | r930: an extended grid picks 1e-4 (53 folds) / 3e-4 (36) / 1e-3 (11); OOF log-loss optimum 3e-4 (0.0071); CV AUC unchanged | sup.md:156 already says "grid bottom"; fix sup.md:52. True gap to S003 shrinks from −1.68e-4 to −6.3e-5 on replicas, invisible to CV |
| C5 | D009: tuned supervised QDA is "indistinguishable" from the GMM in CV | literally true, uninformative | r936_cv_power: corrected t-test non-significant in 222/222 replicas while the true gap is negative in 100% (r210 −1.64e-4, r201 −2.45e-4); real CV: 0 folds better, 1–4 worse | Reword D009 and sup.md (§4) |
| C6 | r209 `sup_qda_tt_r003` / `selftrain_qda_r003` CV 1.0 are honest estimates | overstated | reg 0.003 = modal R201 choice across all outer folds (`r209_semisup_followup.py:46,49`; sup.md:96 discloses) | Never cite them as nested-honest |
| C7 | r255 BQDA has the lowest log-loss | overstated (metric mismatch) | fold-mean CV log-loss 0.0094 vs r209 OOF 0.0071 (summary.json) | "lowest within the fm track", or compare OOF with OOF |
| C8 | No candidate beats S003 significantly on the private LB | confirmed | r936, 222 replicas (tn0, csv8): GMM family −1e-6, r209 −6.9e-5, r255 −1.16e-4, r210 −1.68e-4, r201 −2.57e-4; flat across noise levels | S003 stays the primary final |
| C9 | A second final selection adds meaningful expected AUC | refuted | best pair gain S003 + r209 = +1.45e-5 ± 1.9e-6 (≈ 1/30 of the private sd) | Choose the second final by rule robustness, not by AUC (§6) |
| C10 | The real CV is typical of the parametric replicas | refuted | r936_cv_calib: replica CV worse than real for supervised candidates (r201: only 8% of replicas show ≤ 4 imperfect folds); ranking survives conditioning (13 replicas) | Present replica gaps as model-based |
| C11 | r102 is a free "lottery" hedge (3.9e-5) | refuted (self-corrected by cand) | 6.8e-6 on the CSV scores Kaggle sees | Do not upload GMM-family variants |
| C12 | Unlabeled data matter on boundary rows: GMM 0.825 vs BQDA 0.702 (fm, r254, D009 evidence 2) | confirmed in direction by r404 (updated 2026-10-22 after fix round 1 / research round 2; was: unverified) | On 222 replicas (`20261022-194238_r404_inductive_pooled`, `reports/tables/r404_boundary.csv`), Bayes-boundary rows (posterior 0.1–0.9): S003 0.804 (Bayes 0.806), r255 0.678, r209 0.718, r201 0.664; S003 better in 222/222 replicas against every procedure. On r254's own definition 0.814 vs 0.686 | Quote r404 (222 replicas), not r254's 0.825 / 0.702; D013 item 3 |
| C13 | Label budget: GMM 4 labels, self-training ~20, strict supervised ~80 (R206, D009 evidence 1) | confirmed by R441 (`20261022-182022_r441_label_budget_rerun`), with a qualifier *(updated 2026-10-22 after fix round 1 / research round 2; was: unverified)* | R441 re-ran R206 on 80 new folds (official repeats 4–19) with new label draws: same first budgets as R206 for all matched models, max \|z\| 1.98 over 37 cells (`reports/tables/r440_label_budget_first_hit.csv`, `r440_label_budget_compare.csv`); the nested S005 procedure needs 120, not 80 (80 holds only with reg fixed at 0.003; `research/reports/r2_audit.md` §2). ext had confirmed the related R304 (GMM AUC 1.0 with 2 labels in 50/50; QDA ≥ 0.99 from ~75) | Cite R206 + R441 as reproduced; write "strictly supervised about 80–120 labels" *(updated 2026-10-22 after fix round 1 / research round 2)* |

**D. Leakage**

| # | Claim | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| D1 | No fold-level label leakage in S003, r155, r201, r210, r209, r255, E3 honest | confirmed | code audit (`src/ssc/validation.py:137` is the only label entry); Null A: honest 0.477–0.496, E3-leaky control 0.697 (20/20 > 0.6); Null B: honest symmetric, disk-reader control 1.000 (20/20) | Add a "Leakage audit" paragraph with `reports/figures/r900_perm_null_{A,B}.png` |
| D2 | D007: unsupervised steps never see labels | confirmed for the code; the design saw labels | R001 plotted std vs \|corr\| with the target, R002 checked label purity before the block + K=2 design. Bounded: the block is reproduced from test-only features, BIC picks K=2 (R101), public is scored on unseen labels | One disclosure sentence in Problem formulation |
| D3 | E3 leaky = brief's procedure (1), honest = (2) | confirmed | `extensions/e3_leakage/run.py:131-142`; gap 0 at k_feat 150; null honest 0.49–0.51 | – |
| D4 | E3: leakage inflates CV by 0.073 ± 0.014 at k_feat 20 | mean confirmed; "±" is the SD over 20 split seeds | cells.csv gap_std 0.0143; corrected SE 0.020 (t 3.72, p 0.0003) | Write "0.073 (SD over split seeds 0.014; corrected SE 0.020, p = 0.0003)" |

**E. Reproducibility and submission integrity**

| # | Claim | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| E1 | S003, E3 final, r201 can be re-created from the archive | confirmed (bit-identical on a fresh VM) | R910/R911 sha16 71fb85e7…, eadb9ff1…, 0d7e9b64…; r932 also reproduces r209/r210 | Cite R910 in "how we verified AI output" |
| E2 | Thread count and seeds cause no nondeterminism (Colab) | confirmed | R910 thread_sensitivity.csv | – |
| E3 | Local and Colab results reproduce each other | overstated | laptop vs Colab: S003 test_pred differs by ≤ 4.4e-11 (different OpenBLAS kernel), 1 CSV value changes | Fix colab/README.md |
| E4 | S001–S004 files are valid and were pre-registered | confirmed | r911 B_submissions.csv; created_at < submitted_at for all four; B005 pending before the S004 upload | – |
| E5 | Runs record enough to identify code and environment | refuted | 47 `--no-sync` runs have no bundle/env_hash; 10 pre-D005 runs have no compute block; no BLAS/CPU info; configs lack estimator hyperparameters | Fix remote.py/tracking (§4); future candidate runs must not use `--no-sync` |
| E6 | S003 bundle b8534fcd / E3 bundle b413e6aa are the producing code | unverified (moot) | consistent but not provable; the bit-exact re-runs make it irrelevant | Note in D005 |
| E7 | README reproduces the final submission; final candidates come from a clean commit; the final model is unit-tested | refuted | `README.md:104` uses `configs\TODO_final.yaml`, no TransductiveGMM builder; 191/191 runs `no-commit`; no test imports `ssc.transductive` | Document the `research/r003_*.py` route (sha 3aae140e5837753f); first commit; amend VALIDATION_PROTOCOL §5.1 or re-run finalists |
| E8 | Archive clocks consistent (D006) | overstated | laptop ≈ 100 s ahead of the VMs (51 jobs) | Sync Windows time; note in D006 |

**F. Extensions**

| # | Claim | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| F1 | E1: only 5×20 resolves 0.01 AUC (MDD 0.0061 [0.0051, 0.0076]) | confirmed under its own rule (MDD = 2 sd = 50% power) | R940 reproduces sd 0.003056; at 80% power (2.8 sd) 5×20 gives 0.0086 [0.0071, 0.0107], inconclusive | Name the criterion and its power |
| F2 | E1: the chosen pair's true gap is 0.003 (and the study is about a 0.01 pair) | overstated | R940: in-sample infinite-repeat gap −0.0044 ± 0.0004; official seed −0.0101 is 1.84 sd away; data-sampling sd 0.0105 (R941) | Say the pair is ~0.004 apart; p_correct at 0.01 is an extrapolation |
| F3 | E1: Nadeau–Bengio never significant for the linear pair | refuted | 2% of executions for 3- and 5-fold | "in at most 1 of 50 executions" |
| F4 | E2: L1 AUC(500) = 0.906 [0.881, 0.916]; quote 0.88–0.92 | overstated | R942: the nominal 90% interval covers the replica truth in 21%; bias −0.031; replica truth 0.923 (R941) | "≈ 0.91–0.92, honest uncertainty ≈ ±0.04 (envelope 0.86–0.97)" |
| F5 | E2: gain positive in every bootstrap replicate | uninformative | forced by the fit bounds b ≥ 0 | Drop or explain |
| F6 | E2: QDA gains < 0.002; GMM flat at 1.0; 2 labels suffice for the GMM | confirmed | R942 QDA bias −0.0005; r304 50/50 | – |
| F7 | E3: gap-vs-k_feat explanation | confirmed on average | R944, 100 replicas: honest CV unbiased (≤ 0.004), gap = leak optimism; real k_feat 2 gap (0.056) is a 96th-percentile sample driven by honest-CV under-estimation; the exact zero at k_feat 7 is partly luck | Add those two sentences |
| F8 | E3: LB pending | refuted (stale) | S004 public 0.79136: leaky − public +0.075 (≈ 3.6 SE), honest − public +0.002, proxy − public +0.029 (1.4 SE) | Run `--refresh-lb` and rewrite the Leaderboard/Conclusion paragraphs |
| F9 | E4: B = 2000, sharp drop after the 7th–8th feature | confirmed at ≈ 5% | R943: each observed max step exceeded by 1 of 50 null curves (p ≈ 0.04) | Report the null comparison |
| F10 | E4: drop position does not move with k (k ≥ 8) | overstated | the pre-registered `perm` rule moves 7/8/9/17; only k ∈ {5, 10, 13, 20} were run | Restate under the primary rule |
| F11 | E4: stable ≠ useful; six block features never stable | first supported, second refuted | r320 paired p 0.018/0.0006/0.009; 67 stable at k = 13/20 (perm), 8 and 26 at k = 20 | Qualify "never" |
| F12 | Each draft ≤ 2 pages; numbers trace to the archive | refuted | typeset 2.16–2.71 pages; 15 numeric mismatches (verify_ext §3) | Cut 250–400 words each, E4 drops one table; fix the numbers |

**G. Rules and deliverables**

| # | Claim | Verdict | Evidence | Consequence |
|---|---|---|---|---|
| G1 | Upload limits respected; each upload pre-registered | confirmed | 3 uploads on 10-21 UTC, 1 on 10-22 UTC | – |
| G2 | Using unlabeled test features is allowed (D007) | unverified | rule text not archived; "Foundational Rules 4b" has no archived source; D007 says "团队确认 TODO" | Archive the rules; ask the instructor (D010); upload a strictly inductive hedge |
| G3 | 16 strictly supervised candidates use no test features | confirmed | config flags + feature provenance | – |
| G4 | r300_block_l2_c1/_c01 are strictly supervised | refuted (wrong flag) | config:37 says false, but they use the train+test BLOCK | Count them as group B in the report |
| G5 | TabPFN v2 / TabICL v2 were pretrained "exclusively on synthetic data"; TabICLv2 is "ICML 2026" | unverified / overstated | archived model cards state neither; the TabICL card cites an arXiv preprint | Reword fm.md (§4); analysis-only, never a submission input |
| G6 | FM predictions are not in any submission; no seed or hidden-label recovery attempted | confirmed | grep of submissions.csv and the code | Add a decision forbidding both |
| G7 | Colab use is not sharing outside the team | confirmed, conditional | 45/45 bundles contain no data or credentials; pro2 ownership not archived | Confirm pro2 belongs to a team member |
| G8 | references.md is complete; ai_usage.md is accurate | refuted | 39 entries missing, 13 status changes; `ai_usage.md:11` sentence is false, the table is broken | Fix before any submission of the report |
| G9 | Core report, contribution table, reflections, video are ready | refuted | all templates or TODO | Follow the dated TODO in verify_rules §4 |

### 3. Conflicts between verifiers and how I resolve them

| # | Conflict | Who is right | Why |
|---|---|---|---|
| C1 | **S003's expected private AUC**: cand says "≈ 0.998, confirmed" (0.99775, R156 weights); gen says "overstated, ≈ 0.997 [0.992, 0.9996]" | **gen** | Both use the same machinery; the difference is the prior on flip_y. cand's mixture uses R156's 4-point grid {0, 0.0025, 0.005, 0.01}, which assigns no mass above 0.01. gen's likelihood keeps 0.01–0.015 well supported (LR(0.01) = 0.37, LR(0.015) = 0.12). Under gen's grid prior R922 reproduces 0.99777 exactly, so the computations agree and only the truncation differs. cand's *paired differences* are flat in flip_y, so its ranking and hedge results do not depend on this |
| C2 | **Second final selection**: cand recommends S003 + r209 (highest hedge, +1.45e-5); rules recommends S003 + a strictly inductive QDA (r201) as a rule hedge | **rules for the default; cand for the case where test-feature use is explicitly confirmed** | The two answer different questions. cand's world assumes D007 holds; the gain difference between r209 and r201 as second choice is 1.45e-5 − 0.41e-5 = **1.0e-5**, about 1/40 of the per-level private sd. r209 is itself transductive (it self-trains on the 19,750 test rows), so S003 + r209 leaves both finals exposed if the instructor rejects D007, while S003 + r201 keeps one final compliant under any reading of rule 4. A first-order rule risk dominates a 1e-5 AUC difference. r201 is also more complementary to S003 than r209 under D008 (strictly supervised vs. transductive; boundary-row Spearman 0.45 vs 0.70) |
| C3 | **r201 vs r210 as the inductive candidate**: rules left it open (and said r210 was "not independently re-run"); cand said "use r201" although its own tn0 numbers favour r210 by ≈ 9e-5 | **r201, by a narrow margin** | r210 is better in the tn0 replicas (−1.68e-4 vs −2.57e-4) and under CV conditioning (−1.09e-4 vs −2.73e-4). But (a) r201's design was fixed before these analyses (journal Next (3), sup.md:58), while r210's inner metric is a second-level choice made after reading r201/R209; (b) r210's reg is a grid-floor artefact and it loses 6.3e-4 when training labels are flipped (tn1), whereas r201 is stable; (c) r201 has a fresh-VM bit-exact re-run (R910). The 9e-5 difference is ≈ 1/5 of the per-level private sd. rules' "r210 not independently re-run" is outdated: r932 reproduced r210's test_pred bit-identically, and R900 reproduced its CV |
| C4 | **D009 wording**: leak says D009's "indistinguishable" "stands"; cand says "literally true but uninformative" | **cand** | leak's lens was leakage, and it only confirms that the CV numbers are honest. cand's power analysis shows the CV test cannot detect gaps ≤ 3e-4 (222/222 non-significant while 100% of true gaps are negative). Both agree the statement is literally true, so only the interpretation changes |
| C5 | **E3 "± 0.014"**: leak says it is an SD, report "SE 0.003" instead | **leak on the label, not on the replacement** | ±0.014 is the SD of the per-repeat gap across 20 split seeds (cells.csv), and Table 1 already says "SD". But 0.003 = SD/√20 is a split-noise SE conditional on the 250 rows; the inferential SE that matches the reported p = 0.0003 is the Nadeau–Bengio corrected **0.020** (t 3.72). Use SD 0.014 for spread and corrected SE 0.020 for inference. R944 gives the across-sample view (replica mean gap 0.064 at k_feat 20; real 0.073 is the 70th percentile) |
| C6 | **Extension page counts / cuts**: rules quotes "about 2.25–2.5 pages, cut 150–250 words"; ext measured 2.16–2.71 pages, cut 250–400 | **ext** | ext typeset the drafts (XeLaTeX, Calibri 11 pt, A4, 2.54 cm): E1 2.55–2.66, E2 2.57–2.71, E3 2.16–2.22, E4 2.19–2.23. rules cited ext from memory |
| C7 | **"S003 within 3e-7 of Bayes"**: repro saw 3/30 generic make_classification replicas where S003 misses by 6e-4–7e-3 | **both, conditionally** | gen's R921 quantifies it: 5.5% of generic draws have a block that misses an informative direction. The real block spans all 12 dims (R150, R920), so the real data are in the safe regime. The claim must say "on replicas fitted to our data" |
| C8 | **class_sep**: gen marks "alternatives 0.6–0.9 fit equally well" as refuted | **gen's numbers stand; the refuted sentence is not in the archive** | gen.md never says that; it says 0.71 (0.61–0.81). Only the interval narrows (≈ 0.63–0.78) |
| C9 | **B004 probability**: rules quotes 0.14–0.51; gen 0.14–0.17 | **gen's framing** | Same table (`r922_posterior_by_prior.md`); 0.51 occurs only under a log-uniform prior that favours tiny noise. Use ≈ 0.15 |
| C10 | **flip_y conditioning in cand's replicas** (train labels clean, test labels noisy) vs gen's MLE 0.003 (≈ 0.4 expected flipped train labels) | **no conflict** | A flipped training label at a confident row would be visible (GMM CV < 1, P(observed) < 0.2). None is, so the data condition on clean training labels, which is what tn0 simulates |

### 4. Corrections list (file → exact correction)

These are for the owners; verifiers and I edited none of these files. Main-session files are marked (main).

**Registry and decisions (main)**
1. `registry/beliefs.md`: do not edit B004. Append a new belief: "S003 private AUC median ≈ 0.997, 90% [0.992, 0.9996], P(private ≥ 0.999) ≈ 0.15, private ≈ 0.002 below public" (source `reports/tables/r922_posterior_by_prior.csv`). Also append B006/B007 before the uploads (§5).
2. `registry/decisions.md` D009: replace "不可区分" with: "250 个标签下，调好正则的严格监督 QDA 的 CV 也饱和（0.9999–1.0000），所以 CV 分不出它和 GMM（配对差 95% 区间约 ±2–5e-4；100 折中 0 胜 1–4 负）。这是 CV 分辨率不够，不是两者一样好：在 222 个复刻上同样的检验 222/222 次不显著，而监督 QDA 的真实 AUC 每次都更低（log-loss 调参低 1.7e-4，AUC 调参低 2.6e-4）。" Replace evidence (2) "0.825 vs 0.702 (r254)" by r936 (`reports/tables/r936_private.csv`), or mark it unverified.
3. `registry/decisions.md`: add a decision fixing the final-selection rule **before** the r201/r209 public scores are seen (§6), and a decision "never attempt to recover the generator seed or hidden labels; replica rows never train a submitted model".
4. `registry/decisions.md` D005: "the S003 bundle b8534fcd0d68fc89 is a later-saved copy consistent with the run; E3's bundle is unrecorded (--no-sync); both are covered by the bit-exact re-runs R910". D006: "laptop clock ≈ 100 s ahead of the Colab VMs; sync Windows time".
5. `registry/journal.md` (10-22 round-1 entry): "public 0.99923 说明存在少量标签噪声（约 0.2%），所以 S003 的 private 期望值约 0.998" → "flip_y 的似然支持 0–0.015（MLE 0.003），S003 的 private 约 0.997（90% 0.992–0.9996）（R922）". Append a new entry rather than rewriting if append-only applies.
6. `registry/ai_usage.md:11`: replace the false sentence with the paragraph in `research/reports/verify_rules.md` §3; move the blockquote below the table; append the 5 proposed rows; fill "Member" and "Verified by" honestly.
7. `registry/references.md`: add the 39 BibTeX entries and the 13 status changes from verify_rules §2; set kohavi1995/bengio2004/ambroise2002/meinshausen2010 to used when the extension is submitted.

**Round-1 research reports**
8. `research/reports/gen.md:22` (Table G1, flip_y row): "约 0.0025–0.005（后验均值 0.0044，P(>0) = 0.92）… flip_y = 0 时几乎不可能" → "MLE 0.003，似然支持 0–0.015（sklearn 默认 0.01 的似然比 0.37）；flip_y = 0 被压低约 5:1，没有被排除（LR 0.20，0.03–0.84）（R922）".
9. `research/reports/gen.md:36` and `:53` (Text for the report): "flip_y ≈ 0.004 (P(flip_y > 0) = 0.92) and an expected private AUC of about 0.9977" → "flip_y MLE 0.003 (likelihood support 0–0.015) and an expected private AUC of about 0.997 (90% 0.992–0.9996; R922)"; "within 3·10⁻⁷ of the Bayes AUC" → "about 2.6·10⁻⁷ below the Bayes AUC on replicas fitted to our data (R921)". `:21, :51, :62`: class_sep "0.71 (95% 0.61–0.81)" → "0.70 (≈ 0.63–0.78; R920)".
10. `research/reports/mix.md:8, :27`: "private LB 的 SE 约 1e-4" / "private-LB standard error (~10⁻⁴)" → "private-LB sd ≈ 1e-5 with clean labels and 3–6e-4 at plausible noise rates (R922)"; "three orders of magnitude" → "three to four orders". `:16, :27`: "0.21%（0.05–0.4%）… ≈ 0.998" → "≈ 0.31% (0.05–0.77%) under a flat prior … ≈ 0.997 (90% 0.992–0.9996)".
11. `research/reports/sup.md:42, :104`: "statistically indistinguishable" → the D009 wording above; ":104 0.9999 ± 0.0002, or 1.0000 ± 0.0001" add "(0.9995 after model selection, r205)". `:52`: "把 inner 指标换成 log-loss 就解决了（R210：reg 每折都选 0.001）" → "inner log-loss 选到了网格下边界 0.001；扩展网格选 1e-4/3e-4（r930），CV 不变，复刻上的真实差距缩小约 1e-4；log-loss 调参对错标训练行更脆弱（tn1 −6.3e-4）".
12. `research/reports/fm.md:62, :87`: "只在合成先验数据上预训练" / "pretrained exclusively on synthetic prior data" → "pretrained by their authors on prior-generated data (Hollmann et al. 2025; Qu et al. 2026); never trained on this competition's data", unless an archived source for the exact checkpoints is added. `:62, :74`: "ICML 2026" → "arXiv preprint arXiv:2602.11139". The "lowest log-loss" sentence → "lowest CV (fold-mean) log-loss in the fm track (0.0094); self-training QDA-LW has a lower OOF log-loss (0.0071)". "Where unlabeled data still matter" (0.825 vs 0.702): mark "6 replicas, not re-verified" or cite r936.
13. `research/reports/ext12.md:49`: "所以报告里给 0.88–0.92 … 推翻了我事先的假设（以为 power law 区间会覆盖不足）" → "R942 shows the power-law 90% interval covers the replica truth in only 21% (bias −0.031); the original hypothesis of under-coverage was right on average".
14. `research/reports/ext34.md:12, :51`: "gap 0.073 ± 0.014" → "gap 0.073 (SD over 20 split seeds 0.014; corrected SE 0.020)"; "LB comparison pending" → S004 = 0.79136.

**Extension drafts (owners)**
15. `reports/extensions/e3_leakage.md:66`: "public LB score is pending" → S004 = 0.79136 (run `.venv\Scripts\python.exe extensions\e3_leakage\run.py --refresh-lb 20261022-063908_e3_leakage` first): leaky − public +0.075 (≈ 3.6 public SE), honest − public +0.002, proxy 0.820 − public +0.029 (1.4 SE), inside the pre-registered 90% range 0.785–0.854. `:72`: "0.073 +/- 0.014" → "0.073 (SD over split seeds 0.014; Nadeau–Bengio corrected SE 0.020, p = 0.0003)". Numbers: k_feat 5 leaky 0.817 → 0.816; k_feat 10 gap 0.026 → 0.025; k_feat 13 gap 0.045 → 0.044; caption "leaky 0.62–0.71" → "0.61–0.71"; "~48 features" → "61 features on the real labels (48 is the null expectation)"; "peaks at k_feat = 30 under the null" → "plateau at 20–30". Add: the real k_feat 2 gap is a 96th-percentile sample driven by honest-CV under-estimation; the exact zero at k_feat 7 is partly luck (R944); honest CV picks k_feat 7 while the proxy optimum is 10.
16. `reports/extensions/e1_validation_protocol.md`: hold-out 30% MDD CI "[0.036, 0.053]" → "[0.035, 0.053]"; "4.5–6 times" → "4.5–6.1 times"; "about 2 sd away" → "1.84 sd away"; "never finds … significant" → "in at most 1 of 50 executions"; reference "SE 0.0014" → "≈ 0.0019"; name the criterion (MDD = 2 sd ≈ 50% power; at 80% power 5×20 gives 0.0086 [0.0071, 0.0107]); "two similar models about 0.01 apart" → "the studied pair is ≈ 0.004 apart (R940); p_correct at 0.01 is a normal-shift extrapolation"; add the data-sampling sd 0.0105 to the lower-bound caution; move the citations out of `<!-- -->`.
17. `reports/extensions/e2_learning_curve.md`: Table 1 n = 100 L1 min 0.696 → 0.695; n = 200 0.786 → 0.785; QDA envelope "[0.995, 1]" → "[0.995, 1 (log-linear exceeds 1: 1.008)]"; Figure 1 caption: say the plotted value is the bootstrap median 0.904 (text uses the point fit 0.906); "~1% of test rows near the boundary" → cite r321 (203 rows, 1.03%). Line 44: delete "positive in every bootstrap replicate" or add "(forced by the bounds b ≥ 0)". Line 51: "The back-test favours the power law …, so we would quote 0.88–0.92" → "A coverage simulation on replicas shows the bootstrap interval under-covers (21% for a nominal 90%, bias −0.031; R942), so we quote about 0.91–0.92 with an honest uncertainty of about ±0.04 (envelope 0.86–0.97)". Move citations out of comments.
18. `reports/extensions/e4_feature_stability.md`: "0.769 +/- 0.019" → "±0.018"; "0.832 +/- 0.006" → "±0.005"; "−0.007" → "−0.006"; "≥ 98.6%" / "≥ 0.986" → "≥ 98.5%" / "≥ 0.985"; "once k ≥ 8" → "for k = 10, 13 and 20"; "its position does not move with k" → "the strong group of 7 (+49) is above the drop for k = 10/13/20; under the pre-registered perm rule m* = 7/8/9/17, so whether weak features clear the calibrated envelope depends on k"; "six block features are never stable" → "never in the top-10/freq ≥ 0.5 set (67 is perm-stable at k = 13/20, 8 and 26 at k = 20)"; "far above any linear model" → "+0.019 above the best linear model (0.898)"; add the primary-rule usefulness test (+0.028, p 0.30) and the sharp-drop null (p ≈ 0.04); drop or merge one table.
19. All four drafts: fill the author line; add one visible AI-use line; cut 250–400 words (E1/E2 ≈ 400, E3/E4 ≈ 250) and re-check by a trial export.

**Code and docs (maintainers; not research files)**
20. `colab/README.md`: "local and Colab results reproduce each other" → "bit-identical Colab-to-Colab; the laptop agrees to about 1e-11 (GMM)".
21. `colab/remote.py`: under `--no-sync`, read the VM `.sync_code/.sync_env` markers into `compute.code_bundle/env_hash`. `src/ssc/tracking.py`: record `threadpoolctl.threadpool_info()`, CPU model, joblib/threadpoolctl versions; store `estimator.get_params(deep=True)` in config.
22. `extensions/e3_leakage/run.py`: write `official_cv: true` into summary.json (make_submission otherwise prints a spurious banner).
23. `scripts/make_submission.py`: add `--dry-run`; in `ssc.submission.create`, assert test ids equal the sample ids; enforce the team's 3/day limit (only Kaggle's 5 is enforced).
24. `README.md:104`: replace `configs\TODO_final.yaml` by a working route (config builder for TransductiveGMM, or `research/r003_transductive_vs_supervised.py` with the expected sha256 prefix 3aae140e5837753f).
25. `tests/`: add a unit test for `ssc.transductive.TransductiveGMM` (none exists).
26. Archive metadata: `experiments/20261022-063712_r300_block_l2_c01` and `063713_r300_block_l2_c1` carry `uses_unlabeled_test_features: false` but use the train+test BLOCK; run folders are append-only, so record the correction in the journal and treat them as group B.
27. `research/r935_r933_pooled.npz` (3.5 MB): delete once r936 is frozen (it travels in every code bundle; a copy is in `logs/code_bundles`).

**Verifier reports (for the record)**
28. `research/reports/verify_leak.md` §4.4: "SE of the mean ≈ 0.0032" → add "this is split noise conditional on the 250 rows; the inferential (Nadeau–Bengio corrected) SE is 0.020".
29. `research/reports/verify_rules.md` §1/§5: "r210 … has not been independently re-run" → "r210's test_pred was reproduced bit-identically by an independent re-implementation (r932) and its CV by R900"; §1/§4 page counts "about 2.25–2.5" → "2.16–2.71 (verify_ext §1)"; cuts "150–250 words" → "250–400 (E1/E2), ≈ 250 (E3/E4)".
30. `research/reports/verify_cand.md` §2 "S003 expected private ~0.998, confirmed" → "0.9977 under the R156 4-point prior truncated at 0.01; 0.9968 [0.9923, 0.9996] under a flat prior (verify_gen)".

### 5. Submission plan for today

Kaggle day 2026-10-22 00:00 UTC – 2026-10-23 00:00 UTC (= 09:00 JST 10-22 to 09:00 JST 10-23). Already used: S004 at 16:08:41 JST. Team limit 3, so **2 uploads remain**. The main session uploads; each needs the belief line appended to `registry/beliefs.md` and `--expected-public` set before the upload. SE below = Hanley–McNeil on 395 rows (n1 ≈ 198, n0 ≈ 197); "pred. sd" = cand's ABC predictive sd conditional on S003's public 0.99923 (`reports/tables/r936_public_abc.md`, csv8 scores, ±1.5e-4 window), which is much tighter because the public rows and their label noise are shared with S003.

| Priority | Run | Purpose | Expected public | HM SE | pred. sd | Status |
|---|---|---|---|---|---|---|
| 1 | `20261022-063908_e3_leakage` (E3 final model) | E3 leaderboard comparison | 0.82 (registered) | 0.021 | – | **Done: S004, public 0.79136, B005 resolved (confirmed). Do not re-upload.** No verifier found it invalid (leak: honest pipeline clean; repro: bit-exact) |
| 2 | `20261022-071234_r201_qda_block` | Strictly inductive QDA; makes a rule-robust final selectable (compliant under any reading of rule 4) | 0.9988 | 0.0017 | 0.00066 | to upload |
| 3 | `20261022-080632_r209_selftrain_qda_lw` | Best expected-AUC second final if D007 is confirmed (+1.45e-5 hedge) | 0.9991 | 0.0015 | 0.00033 | to upload |

*(updated 2026-10-22 after fix round 1 / research round 2)* S004 was uploaded at 16:08:41 JST (2026-10-22T07:08:41Z), public 0.79136 (`registry/submissions.csv`). S005 (r201) and S006 (r209) have been created (`registry/submissions.csv`, 18:10 JST) and are not uploaded yet. D010 switch 3 (instructor forbids test features) is updated by research round 2: ext-grid QDA r930 (`20261022-095943_r930_qda_block_nll_ext`) + r201, instead of "r201 plus r210 or r255_bqda_eb_trainonly" (`research/reports/r2_inductive.md`). This is pending decision D013 by the main session (written in `registry/decisions.md`, status proposed, team confirmation TODO).

Do **not** upload today: r105/r155/r102 (public equals S003's in 87–95% of replica draws), r210 (inductive alternative; second-level selection), r255 (equivalent to r209), any r251 FM run (never).

Ready-to-paste lines for `registry/beliefs.md` (next free ids B006, B007; adjust if others were added first):

```
| B006 | 2026-10-22 | TODO (AI-assisted) | 下一次提交 r201_qda_block（`20261022-071234_r201_qda_block`，严格归纳 QDA：块在每个训练折内重新推导，不用任何测试集特征，CV 0.99990）的 public AUC ≈ 0.9988，80% 落在 0.9980–0.9997（以 S003 的 public 0.99923 为条件，预测 sd 0.00066；不加条件的 Hanley–McNeil SE 0.0017）；高于 S003 的概率约 20%。这个分数不影响最终选择（public 与 S003 之差的 sd 6.5e-4 大于真实差距 2.6e-4）；低于 0.9960 说明文件有问题 | 80% 落在区间内 | 上传后 record_lb；对比 reports/tables/r936_public_abc.md（tn0，csv8，±1.5e-4） | pending | |
| B007 | 2026-10-22 | TODO (AI-assisted) | 下一次提交 r209_selftrain_qda_lw（`20261022-080632_r209_selftrain_qda_lw`，自训练 QDA-LW，用测试集特征，CV 1.0）的 public AUC ≈ 0.9991，80% 落在 0.9987–0.9996（以 S003 的 public 为条件，预测 sd 0.00033；Hanley–McNeil SE 0.0015）；高于 S003 的 0.99923 的概率约 25–28%，与它相同的概率约 4%。最终选择规则在看到这个分数之前已经写进 decisions.md，不因分数改变；低于 0.9975 说明文件有问题 | 80% 落在区间内 | 上传后 record_lb；对比 r936_public_abc.md | pending | |
```

Commands (main session, after appending the lines):
- `scripts/make_submission.py --run 20261022-071234_r201_qda_block --expected-public 0.9988 …`
- `scripts/make_submission.py --run 20261022-080632_r209_selftrain_qda_lw --expected-public 0.9991 …`
- Then `record_lb.py` for each.

Also append the new S003 private belief from §4 item 1.

### 6. Final-two recommendation

**Default (now, and if the instructor does not answer): S003 (`20261022-050708_r003_gmm2_hv`) + r201 (`20261022-071234_r201_qda_block`).**

1. **S003 stays.** It is leakage-free (R900/R901), bit-exact (R910), about 2.6e-7 below Bayes, and no candidate beats it on replicas (r936). Its private score is set by label noise: median ≈ 0.997, 90% [0.992, 0.9996].
2. **The second final hardly matters in expected AUC.** Pair gains over S003 alone:

   | Second final | Gain |
   |---|---|
   | r209 | +1.45e-5 |
   | r255 | +1.37e-5 |
   | r210 | +5.0e-6 |
   | r201 | +4.1e-6 |

   The largest difference, 1.0e-5, is about 1/40 of the per-level private sd (≈ 4e-4). So the second slot should insure against the risk that is not in the replica world: the rule interpretation.
3. **That risk is first-order and unresolved.** D007 rests on paraphrased rule text (G2), and neither the team nor the instructor has confirmed it. r209 self-trains on the test rows, so it shares S003's exposure. r201 is compliant under any reading of rule 4, was pre-designated before the replica analyses, has a fresh-VM bit-exact re-run, and is the most methodologically different from S003 (D008).
4. **Conditional switches:**
   - If D010 (instructor) explicitly allows unlabeled test features: S003 + r209 (r255 equivalent). This gains about 1e-5.
   - If D010 forbids them: S002 and S003 are out. Choose two strictly inductive submissions, for example r201 plus r210 or the fully inductive Bayesian QDA `20261022-070202_r255_bqda_eb_trainonly` (CV 0.9990). First check the latter on replicas (§7); either one needs an upload. *(updated 2026-10-22 after fix round 1 / research round 2: the replica comparison is done; research round 2 recommends ext-grid QDA r930 + r201 for this case, pending D013; see the note in §5.)*
   - In every case: never an FM run, never S004.
5. **Write the rule into `registry/decisions.md` before the r201/r209 public scores are seen.** Neither public score can move the choice: the sd of the public difference to S003 is 3.2–6.5e-4, larger than any true gap.

### 7. Completeness critic

#### 7.1 Round-1 claims that no verifier checked

| Claim (source) | Why it matters | Suggested check |
|---|---|---|
| R206 label-budget curve: GMM 4 labels, self-training ~20, supervised ~40, strictly supervised ~80 for AUC ≥ 0.99 (sup) | D009 evidence (1) and a headline "value of unlabeled data" result. Only the related R304 was confirmed (ext) | done (R441, R440): R206 reproduced on 80 new folds; strictly supervised needs 80–120 labels (see C13) *(updated 2026-10-22 after fix round 1 / research round 2; was: independent re-run on Colab with a different seed set)* |
| Boundary-row AUC on replicas: GMM 0.825 vs BQDA 0.702, average 0.795 (fm, r254; 6 replicas) | D009 evidence (2). r936 supports the direction overall, but the boundary-row numbers rest on 6 replicas | done (r404, 222 replicas: S003 0.804 vs r255 0.678; updated 2026-10-22 after research round 2; was: compute boundary-row AUCs from the 222 r933 replicas) |
| Fully inductive Bayesian QDA `r255_bqda_eb_trainonly` CV 0.9990; EB picked kappa0 = 1 (grid edge) in 100/100 folds (fm) | the strongest strictly inductive alternative if D010 forbids test features; a second grid-edge artefact like r210's | add both to the r933 replica design |
| R256 controls (Student-t vs plug-in), DP-GMM always 2 components, GPC-ARD 0.978, horseshoe logistic 0.926, R257 identifiability (fm) | negative/supporting results that will appear in Analysis | archive re-read only; low priority |
| TabPFN v2 0.985/0.989, TabICL 0.993/0.999 scores (fm) | quoted in the report as comparisons | not re-run by anyone; rules checked compliance only |
| R154: best linear projection capped at population AUC 0.947, Bayes error 0.33%, ~186 boundary rows (gen) | explains why linear models stall; quoted in gen's report text | R921's parametric generator can compute it directly |
| R151 replica validation (R003 ranking reproduced in 97%) (gen) | underpins every replica-based argument. cand's r936_cv_calib shows the replica is harder for supervised QDA than the real data, a partial qualification | compare replica vs real CV for all 8 candidates (partly done in r936) |
| R104 reg_covar analysis, R103 refinement differences (~1e-9), R101 "30 restarts, one optimum" (mix) | supporting, tiny effects | none needed |
| "CV log-loss does not measure ranking" (mix) | affects every log-loss comparison (fm, sup, cand) | none needed; cite it when log-loss appears |
| Supervised ablations: LDA 0.86–0.88, GNB 0.87, linear ceiling 0.938, poly 0.930, SVM 0.951, kNN 0.902, trees ≤ 0.914; "label-free block beats in-fold \|corr\| filter" (sup) | quoted in the core report | done (R441, R440): R440 recomputed all 64 rows of `r249_comparison.csv` from `fold_scores.csv` and `summary.json`, all match (`reports/tables/r440_r249_recheck.csv`) *(updated 2026-10-22 after fix round 1 / research round 2; was: spot-check 3–4 rows against `fold_scores.csv`)* |
| R300 side result: L1 C = 0.2 0.8729 > C = 0.1 0.8469 (ext12) | used in E2's "part of the gain is regularisation" | archive read |
| "GridSearchCV tie-breaking over-regularises once AUC saturates" (sup) | a methodological claim for the report | cand's r930 is consistent; a direct test is easy |
| "compare_runs.py se_split is trustworthy" (within-run SE = seed-to-seed sd) (ext12) | the official CV's SE is used everywhere | ext reproduced sd 0.003056, but not this equality explicitly |

#### 7.2 Checks planned but not run

| Check | Verifier | Reason |
|---|---|---|
| Permutation nulls for r102 and r205 | leak | not final candidates; audited by reading code only |
| Hidden-test-side leakage (public/private construction) | leak | not observable from our side |
| Re-runs of the other 46 `--no-sync` runs | repro | out of scope; their code provenance remains unverified |
| A real `make_submission` dry run | repro | the script has no `--dry-run`; steps were simulated in a temp folder |
| Which clock is wrong (laptop vs VM) | repro | no external time source consulted |
| Stratified or differently sized public split; generators outside make_classification | gen | not modelled (immaterial for ±5 rows) |
| Candidates other than S003 in the noise/private model | gen | delegated to cand |
| `r934_replica_summary.py` | cand | superseded by r936, never run |
| make_classification ("mc") replica variant for candidates; flip_y > 0.01; tn1 on CSV scores | cand | time; gaps are flat in flip_y, so the ranking should hold |
| Real-population E2 value at 500 rows | ext | replica too steep at t = 80; the under-coverage finding does not depend on it |
| `scripts/package_deliverables.py` dry run | rules | blocked by the permission classifier |
| Word page counts | rules/ext | XeLaTeX approximation used |
| Rule text and instructor view | rules | not archived; team action |

### 8. Open questions

1. What do the Kaggle Rules tab and the course brief say about unlabeled test features and pretrained models? What will the instructor say (D010)? This decides the final two (§6).
2. Who owns the pro2 Colab account? Do Colab's terms allow two paid accounts in parallel?
3. Is there a Kaggle team merge and a Canvas team registration? What are the checkpoint date, the reflection deadline, and the number of extensions that can be owned (team size vs "all four")?
4. Did the competition author keep sklearn's default flip_y = 0.01? The data cannot separate 0.001 from 0.012; only the private LB will tell.
5. Is the real training sample "lucky" for supervised QDA, or is the parametric replica slightly too hard (r936_cv_calib)? The ranking survives conditioning, but on only 13 replicas.
6. Should the second hedge be the most different candidate (r255, boundary Spearman 0.55 with S003) rather than r209 (0.70) against failure modes outside the replica world? In the replica world they are equivalent.
7. If test features are forbidden, which two inductive submissions? r201, r210, r255_bqda_eb_trainonly and the extended-grid QDA have not been compared on one replica design.
8. Disclose the label-informed design of the block/K = 2 pipeline (R001/R002 looked at labels)? I recommend one sentence; the team decides.
9. What data were the exact TabPFN (v2_default alias) and TabICL v2 checkpoints pretrained on? What is the exact TabPFN attribution wording?
10. Page format of the course (font, size, margins) for the 2-page extension limit.
