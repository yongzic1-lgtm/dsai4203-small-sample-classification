# Validation Protocol (draft for the report)

Status: **frozen without change by D015 (2026-11-01, proposed; team confirmation TODO)**: the E1 results support Sections 2-4 as written (5 × 20 is the only tested scheme whose MDD CI lies wholly below 0.01). Section 5 was **not applied** and is replaced by D010 (+D013); see the note there. Original status: draft, adopted as decision D002 on 2026-10-22, to be frozen after the E1 results, target 2026-11-01. `TODO(E1)` marks the places where E1 evidence must justify a choice. This text is written to be cut down into Section 2 of `reports/report_core.md`.

---

## 1. Setting

- 250 labelled rows (113 positive, 45.2 %), 150 numeric features, most of them presumably noise; 19,750 unlabelled test rows. Metric: ROC AUC.
- With so few rows, every score we compute is noisy. The protocol has to (a) estimate AUC with known precision, (b) compare two models fairly, and (c) prevent information from the validation rows leaking into model fitting.

## 2. Official protocol

| Item | Value | Where it is defined |
|---|---|---|
| Scheme | repeated **stratified 5-fold** CV, **20 repeats** (100 fits per model) | `project.yaml: official_cv` |
| Split seeds | repeat *r* uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026 + r)`, r = 0..19 | `ssc.validation.make_splits` |
| Fold sizes | 200 training rows, 50 validation rows (about 22-23 positives) | |
| Leakage control | all preprocessing (scaling) and feature selection are steps of one sklearn `Pipeline`, re-fitted on the training part of every fold | `ssc.models.build_pipeline`, `ssc.features` |
| Model seed | `pipeline.random_state` (default 0) held fixed for every fold and repeat, so the spread across repeats is caused by the split alone | config `pipeline.random_state` |
| Split identity | SHA1 of all validation index sets ("split signature", `split_sig`); two runs are comparable only if their signatures match | `ssc.validation.split_signature` |
| Reported per model | mean AUC over the 100 folds **± std across the 20 repeat means** (`auc_std_repeat`); also std across folds and pooled out-of-fold AUC | `summary.json`, `registry/experiments.csv` |

Notes:
- sklearn's `(Stratified)KFold` does not shuffle unless `shuffle=True`; without shuffling every "repeat" is the same split and the repeats carry no information. Our scheme always shuffles.
- Stratification keeps the 45 % positive rate in every 50-row fold, which removes one source of fold-to-fold variance.
- `auc_std_repeat` measures how much a whole 5-fold estimate would move under a different random split of the same 250 rows. `auc_std_fold` is much larger (one 50-row fold has a Hanley-McNeil SE of about 0.065 at AUC 0.80) and is not the precision of the mean.
- E1 result (real data; D015): on the linear neighbouring pair the minimum detectable difference MDD = 2 sd(A − B) over 50 split seeds is 0.057 [0.048, 0.072] for a 20 % hold-out, 0.026-0.030 for a single 3-, 5- or 10-fold, 0.0083 [0.0070, 0.0104] for 5-fold × 10 and **0.0061 [0.0051, 0.0076] for 5-fold × 20**, the only tested scheme whose whole CI lies below 0.01 (`reports/extensions/e1_validation_protocol.md` Table 1; runs `20261022-064054_e1_validation_protocol`, `20261022-064148_e1_validation_protocol_qda_pair`; R306, R940). At 80 % power even 5 × 20 is not enough (0.0086 [0.0071, 0.0107]), and near the CV ceiling no scheme separates the candidates (D011). Original to-do (done): justify 5 folds and 20 repeats with the E1 spread table (`reports/tables/e1_scheme_spread.md`) and the paired-difference table (`reports/tables/e1_paired_difference.md`): report the std of a single hold-out, 3/5/10-fold and repeated 5-fold estimate, and show that the official scheme resolves the differences we care about (MDD with its 95 % CI below 0.01 for a pair of neighbouring candidates; the MDD depends on how similar the two models are).

## 3. Comparing two models

1. Both models are evaluated on the **identical** 100 splits (same split signature). Compute the per-fold difference d<sub>j</sub> = AUC<sub>A,j</sub> - AUC<sub>B,j</sub>, j = 1..J, J = 100.
2. Report the mean difference, its **split standard error** se<sub>split</sub> = std of the 20 per-repeat mean differences / √20 (`se_split`), and the fraction of folds where A wins (ties count ½). Note that `std_diff_repeat` itself is the spread of a *single* repeat's difference; the reported mean over 20 repeats moves about √20 ≈ 4.5 times less under other split seeds (E1 confirms this on synthetic data: within-run se<sub>split</sub> 0.0027 vs seed-to-seed std 0.0026 for 5-fold × 20).
3. Significance: **Nadeau-Bengio corrected resampled t-test** (Nadeau & Bengio, 2003). The overlap between training sets makes fold differences positively correlated, so the naive variance σ²/J is replaced by (1/J + n<sub>val</sub>/n<sub>train</sub>) σ². With J = 100 and n<sub>val</sub>/n<sub>train</sub> = 50/200 = 0.25 the corrected standard error is about 5.1 times the naive one. Degrees of freedom J - 1. This is the data-sampling view (would the difference hold on new rows?), far more conservative than se<sub>split</sub>, which only asks whether other splits of the same 250 rows could flip the sign.
4. Decision language: |mean difference| < 2 se<sub>split</sub> is "within split noise" (the rule `compare_runs.py` prints); we do not claim an improvement unless |mean difference| > 2 se<sub>split</sub> **and** the corrected p-value is below 0.05 **and** A wins in most folds (`win_rate_fold` > 0.5).
5. Implementation: `scripts\compare_runs.py RUN_A RUN_B` or `--ref RUN_REF RUN_1 ...`. It refuses runs with different split signatures or data, then calls `ssc.validation.paired_compare`, which also checks the per-fold validation-set hashes (`val_id` in `fold_scores.csv`) when both runs have them.

## 4. How much to trust the public leaderboard

The public LB scores about 2 % of the test set, ≈ 395 rows. Assuming the test positive rate equals the training rate (45.2 %), that is ≈ 179 positives and 216 negatives. The Hanley & McNeil (1982) standard error of an AUC on these rows:

| true AUC | SE public (395 rows) | SE private (≈ 19,355 rows) |
|---|---|---|
| 0.75 | 0.025 | 0.004 |
| 0.80 | 0.023 | 0.003 |
| 0.85 | **0.020** | 0.003 |
| 0.90 | 0.017 | 0.002 |

Reproduce with the formula in `ssc.validation.auc_standard_error`:
```powershell
.venv\Scripts\python.exe -c "from ssc.data import load; from ssc.validation import auc_standard_error as se; y=load('raw').y; p=y.mean(); n=395; k=round(n*p); print(round(p,3), k, n-k, [round(se(a, k, n-k),4) for a in (0.75,0.80,0.85,0.90)])"
```
`scripts\lb_noise.py` gives the same table (also for one 50-row CV fold and the private LB) plus a simulation of how often the public LB swaps the order of two submissions whose true AUCs differ by 0.005-0.03 (`reports/tables/lb_noise_*.csv`).

Consequences:
- A single public score is only known to about ±0.04 (95 %). Two submissions whose public scores differ by 0.01-0.02 are not distinguishable; their CV comparison is far more informative than their LB comparison.
- Repeatedly choosing whichever submission scores higher on the same 395 rows overfits those rows (adaptive overfitting); the private LB (≈ 98 % of the test set, SE ≈ 0.003) is what counts.
- The public LB is used for: detecting pipeline bugs (a public score more than ~3 SE ≈ 0.06 below the CV mean triggers a check of id order, sign, and run choice), and recording the CV-to-LB relation for the report. It is **not** used to rank candidate models.
- The private LB is published only after the deadline (2026-11-29 23:59 HKT), when the report and zip are already submitted. The core report's analysis therefore uses CV vs expected vs public; private scores are recorded for the archive (`record_lb.py --private`) and used in the individual reflection only if its Canvas deadline is later. `TODO`: if so, compare the public-vs-private gaps of all our submissions with this SE there.

## 5. Final-model selection rule (decided in advance)

> **Superseded (D015, 2026-11-01).** This rule was not applied. No run comes from a clean git commit (all 214 are `no-commit`, D004), and 30 of the 139 official runs score AUC 1.0 in all 100 folds, so the tolerance band cannot separate the candidates (D011). The final selections follow D010 (+D013), written before the public scores of S005 and S006 were seen; code provenance rests on the code bundles and bit-exact re-runs instead of a commit (D012, D013, R910). The text below is kept as the original plan.

To be confirmed as a decision in `registry/decisions.md` **before 2026-11-23** (the last week) and then applied mechanically.

1. **Candidates**: runs on the real data under the official protocol (same split signature), status `done`, made from a clean git commit, with test predictions.
2. **Best**: B = the candidate with the highest mean CV AUC.
3. **Tolerance band**: all candidates M whose paired difference to B (`compare_runs.py --ref B M ...`) is not a clear loss: mean diff(M − B) > −2 se<sub>split</sub>(M − B), or corrected p ≥ 0.05. The yardstick is the paired difference, not `auc_std_repeat(B)`: the single-repeat std of one model mixes split noise that pairing cancels, and E1 (synthetic, 5-fold × 20, two neighbouring candidates) shows se<sub>split</sub> of the paired difference of about 0.003 against an `auc_std_repeat` of about 0.015 for a single model. A candidate is thus excluded only when it is clearly worse in both views, mirroring the rule for claiming an improvement (Section 3.4). `TODO(E1)`: confirm the numbers with the real-data E1 run and the team's own neighbouring candidates as A/B.
4. **Choose the simplest model in the band**, by this order: fewer input features actually used; linear before kernel or tree ensembles; fewer tuned hyperparameters. Ties are broken by the higher mean CV AUC. This is the "one-standard-error rule" (Hastie, Tibshirani & Friedman, 2009, Sec. 7.10): with 250 rows, a simpler model inside the noise band is the safer bet on 19,355 private rows.
5. **Never select by the public LB alone.** The public score only vetoes a candidate if it indicates a bug (Section 4).
6. **Kaggle final selections**: the first selected submission is the rule's choice. If Kaggle allows a second one, it is B (the best-CV model) when B differs from the rule's choice; otherwise the next model in the band by the same order. Both are written down in the decision before the last week.
7. **Optimism of the winner**: the mean CV AUC of the selected model is biased upwards because it is the maximum over many evaluated configurations (Varma & Simon, 2006; Cawley & Talbot, 2010). The report states how many configurations were evaluated (`registry/experiments.csv`). `TODO`: optionally a nested-CV estimate for the final pipeline.

## 6. What this protocol does not cover

- Test predictions come from the run's `test_pred` mode: `fold_mean` (default: average of the 100 fold models' scores) or `refit` (one extra fit on all 250 rows). The final candidates use `fold_mean` (S003: `configs/final_s003_gmm2_hv.yaml`; S005: run `20261022-071234_r201_qda_block`): the average of 100 models fitted on 200 rows each, so no model is fitted outside the validated procedure and the score averages out the fold-to-fold variation.
- Distribution shift between train and test is not modelled; train and test feature spreads are compared in the problem formulation (`scripts\verify_data.py` summary).

## References used in this section

Nadeau & Bengio (2003); Hanley & McNeil (1982); Hastie, Tibshirani & Friedman (2009); Varma & Simon (2006); Cawley & Talbot (2010); Pedregosa et al. (2011). Full entries: `registry/references.md`.
