"""R255 - does a Bayesian treatment of QDA close the QDA (0.996) -> GMM (1.000) gap WITHOUT unlabeled data?

Supervised-only candidates on the high-variance block, official CV (project.yaml official_cv,
identical splits), each archived as an experiment run with test predictions:

  bqda_eb_hv        Bayesian QDA (Normal-Inverse-Wishart per class, Student-t posterior predictive),
                    prior hyperparameters (kappa0, nu0, prior covariance shape) by empirical Bayes
                    inside each fold; block + standardisation as in R003 (label-free train+test stats)
  bqda_eb_trainonly same model, but the block rule and the standardisation are computed on the
                    fold's 200 training rows only: no unlabeled row is used anywhere (fully inductive)
  bqda_weak_hv      Bayesian QDA with a fixed weak prior (kappa0 0.01, nu0 = d+3, isotropic): the
                    effect of averaging over Sigma alone, without shrinkage
  qda_trainonly     classical QDA(reg 0.1) with the train-only block (paired reference for the above)
  gpc_ard_hv        Gaussian-process classifier (Laplace), ARD RBF kernel, on the block (R003 basis)

References recomputed with identical code/splits (not archived again): R003 sup_qda_hv,
sup_logreg_hv and gmm2_hv (= S003). Tables: reports/tables/r255_*.csv.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm -- research/r255_bayes_qda.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.pipeline import Pipeline

from ssc import paths
from ssc.tracking import Run
from ssc.transductive import SubspaceModel, resolve_columns, unlabeled_features
from ssc.validation import cross_validate

import lib_fm
from lib_fm import BayesQDA, TrainHighVar

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--max-folds", type=int, default=None, help="smoke tests on synthetic data only")
a = p.parse_args()
src = a.data
if a.max_folds and src == "raw" and (lib_fm.ON_VM or not os.environ.get("SSC_ARCHIVE_ROOT")):
    raise SystemExit("--max-folds is for local smoke tests into a scratch archive only (official CV uses all 100 folds)")

ds, scheme, splits = lib_fm.official_setup(src)
if a.max_folds:
    splits = splits[: a.max_folds]
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)
hv = resolve_columns("highvar", unlabeled_features(src))
d_hv = len(hv)
print(f"cpus={os.cpu_count()} n_jobs={n_jobs} block={[int(c) for c in hv]} (d={d_hv})")

CANDIDATES = {
    "bqda_eb_hv": (lambda: SubspaceModel(BayesQDA(), "highvar", source=src),
                   "Bayesian QDA, NIW prior by empirical Bayes per fold, label-free block (R003 basis)", True),
    "bqda_eb_trainonly": (lambda: Pipeline([("block", TrainHighVar(1.5)), ("model", BayesQDA())]),
                          "Bayesian QDA, empirical Bayes; block + scaling from the fold's training rows only", False),
    "bqda_weak_hv": (lambda: SubspaceModel(BayesQDA(mode="fixed", kappa0=0.01, nu0_extra=2.0, prior="iso"),
                                           "highvar", source=src),
                     "Bayesian QDA, fixed weak NIW prior (kappa0=0.01, nu0=d+3, isotropic)", True),
    "qda_trainonly": (lambda: Pipeline([("block", TrainHighVar(1.5)),
                                        ("model", QuadraticDiscriminantAnalysis(reg_param=0.1))]),
                      "classical QDA(reg 0.1); block + scaling from the fold's training rows only", False),
    "gpc_ard_hv": (lambda: SubspaceModel(GaussianProcessClassifier(
        ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(d_hv), (1e-2, 1e3)), n_restarts_optimizer=1,
        random_state=0), "highvar", source=src),
                   "Gaussian-process classifier (Laplace), ARD RBF kernel, label-free block", True),
}
names = a.only or list(CANDIDATES)

# ---- references (recomputed R003 configurations, identical splits)
refs = {}
for rn in ("r003_sup_qda_hv", "r003_sup_logreg_hv", "r003_gmm2_hv"):
    refs[rn] = lib_fm.r003_reference(rn, ds, splits, scheme, src, n_jobs=n_jobs)
    print(f"reference {rn}: auc={refs[rn].fold_scores['auc'].mean():.5f}")
s003 = refs["r003_gmm2_hv"].test_pred

rows, cvs = [], {}
for name in names:
    make, desc, uses_unl = CANDIDATES[name]
    cfg = {"name": f"r255_{name}", "description": desc, "model_code": "research/r255_bayes_qda.py + research/lib_fm.py",
           "cv": scheme.to_dict(), "test_pred": "fold_mean", "uses_unlabeled_test_features": uses_unl,
           "block": [int(c) for c in hv], "code_sha": lib_fm.code_hash(__file__)}
    hyp = ("A Bayesian (NIW, Student-t predictive) QDA averages over the class covariances instead of plugging "
           "in noisy estimates from ~90 rows per class; predicted: CV AUC above QDA(reg 0.1) 0.9958 by more "
           "than 2 split-SE, but still below the transductive GMM (1.0000); refuted if not above QDA.")
    if name.startswith("gpc"):
        hyp = ("A GP classifier with an ARD RBF kernel can represent the quadratic (covariance) boundary; "
               "predicted CV AUC between L2 logistic (0.898) and QDA (0.996).")
    with Run(f"r255_{name}", cfg, kind="experiment", source=src, author=a.author, hypothesis=hyp) as run:
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
        s = run.save_cv(cv)
        cvs[name] = cv
        cal = lib_fm.oof_calibration(cv.oof, ds.y)
        agr = lib_fm.reference_agreement(cv.test_pred, s003)
        pc = {rn: lib_fm.paired_table({name: cv}, {rn: rcv})[0] for rn, rcv in refs.items()}
        run.save_table("paired_vs_references", pd.DataFrame(list(pc.values())))
        row = {"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
               "oof_auc_mean": s.get("oof_auc_mean"), **cal,
               "diff_vs_qda": pc["r003_sup_qda_hv"]["mean_diff"], "se_split_vs_qda": pc["r003_sup_qda_hv"]["se_split"],
               "p_vs_qda": pc["r003_sup_qda_hv"]["corrected_p"], "win_vs_qda": pc["r003_sup_qda_hv"]["win_rate_fold"],
               **{f"s003_{k}": v for k, v in agr.items()}, "uses_unlabeled_test_features": uses_unl, "description": desc}
        rows.append(row)
        run.finish(extra_summary={"official_cv": True, **cal, "agreement_with_s003": agr,
                                  "paired_vs_r003_sup_qda_hv": pc["r003_sup_qda_hv"],
                                  "uses_unlabeled_test_features": uses_unl})
        print(f"{name:18s} auc={s['auc_mean']:.5f} +/- {s['auc_std_repeat']:.5f}  logloss={cal['cv_logloss']:.4f}  "
              f"diff_vs_qda={row['diff_vs_qda']:+.5f} (se {row['se_split_vs_qda']:.5f}, p {row['p_vs_qda']:.3g})  "
              f"pseudoAUC={agr['pseudo_auc_confident']:.5f}  run={run.run_id}")

# ---- reference rows for the same table
for rn, rcv in refs.items():
    cal = lib_fm.oof_calibration(rcv.oof, ds.y)
    agr = lib_fm.reference_agreement(rcv.test_pred, s003)
    rows.append({"candidate": f"[ref] {rn}", "run_id": "(recomputed R003 config)", "auc_mean": rcv.fold_scores["auc"].mean(),
                 "auc_std_repeat": rcv.summary()["auc_std_repeat"], "oof_auc_mean": rcv.summary().get("oof_auc_mean"),
                 **cal, **{f"s003_{k}": v for k, v in agr.items()}, "description": lib_fm.R003_REFERENCES[rn]})

res = pd.DataFrame(rows)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
res.to_csv(tables / "r255_bayes_qda_candidates.csv", index=False)
pd.DataFrame(lib_fm.paired_table(cvs, refs)).to_csv(tables / "r255_paired_vs_references.csv", index=False)
print("\n" + res[["candidate", "auc_mean", "auc_std_repeat", "cv_logloss", "cv_brier", "s003_pseudo_auc_confident",
                  "s003_side_agree_boundary"]].to_string(index=False))

# ---- analysis: what does empirical Bayes choose, per fold and on all 250 rows?
cfg = {"name": "r255_bqda_eb_analysis", "kind": "analysis", "inputs": "train rows (labels), label-free block"}
with Run("r255_bqda_eb_analysis", cfg, kind="analysis", source=src, author=a.author,
         hypothesis="EB picks a weak-to-moderate prior (nu0 - d - 1 small relative to ~90 rows per class) "
                    "because the class covariances differ strongly (QDA >> LDA).") as run:
    choices = []
    for sp in splits:
        m = SubspaceModel(BayesQDA(), "highvar", source=src).fit(ds.X.to_numpy()[sp.train_idx], ds.y[sp.train_idx])
        e = m.estimator_
        choices.append({"repeat": sp.repeat, "fold": sp.fold, "kappa0": e.kappa0_, "nu0_extra": e.nu0_extra_,
                        "prior": e.prior_, "log_marginal": e.log_marginal_, "rank": e.proj_.shape[1]})
    ch = pd.DataFrame(choices)
    run.save_table("eb_choice_per_fold", ch)
    blocks = Counter(tuple(int(c) for c in TrainHighVar(1.5).fit(ds.X.to_numpy()[sp.train_idx]).columns_)
                     for sp in splits)
    block_sets = [{"columns": list(k), "n_folds": v, "same_as_label_free_block": list(k) == [int(c) for c in hv]}
                  for k, v in blocks.most_common()]
    print("train-only block sets:", block_sets)
    full = SubspaceModel(BayesQDA(), "highvar", source=src).fit(ds.X.to_numpy(), ds.y).estimator_
    eb = pd.DataFrame(full.eb_table_, columns=["kappa0", "nu0_extra", "prior", "log_marginal"]).sort_values(
        "log_marginal", ascending=False)
    run.save_table("eb_grid_all250", eb)
    eb.head(30).to_csv(tables / "r255_eb_grid_all250_top30.csv", index=False)
    freq = Counter(zip(ch["kappa0"], ch["nu0_extra"], ch["prior"]))
    top = [{"kappa0": k[0], "nu0_extra": k[1], "prior": k[2], "n_folds": v} for k, v in freq.most_common()]
    pd.DataFrame(top).to_csv(tables / "r255_eb_choice_frequency.csv", index=False)
    print("EB choice frequency over folds:\n" + pd.DataFrame(top).to_string(index=False))
    print("EB on all 250 rows: top grid points\n" + eb.head(8).to_string(index=False))
    run.finish(extra_summary={"official_cv": False, "eb_all250": {"kappa0": full.kappa0_, "nu0_extra": full.nu0_extra_,
                                                                  "prior": full.prior_, "rank": int(full.proj_.shape[1])},
                              "eb_fold_choices": top, "train_only_block_sets": block_sets})
