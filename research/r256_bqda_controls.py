"""R256 - controls for R255: is the Bayesian-QDA gain due to integrating over Sigma, or just to the
amount / target of covariance shrinkage?

Official CV (identical splits), experiment runs with test predictions, all on the label-free block
(R003 basis) unless noted:

  bqda_eb_plugin_hv   same empirical-Bayes NIW prior as r255_bqda_eb_hv, but scored with a Gaussian at the
                      posterior means (no Student-t predictive): isolates the posterior-predictive effect
  bqda_eb_widekappa_hv  EB with kappa0 grid up to 300 (R255's grid ended at kappa0 = 1, which EB chose in
                      every fold: a follow-up written after seeing R255, reported as such)
  bqda_weak_plugin_hv weak fixed prior of r255_bqda_weak_hv with the Gaussian plug-in (nearly ML-QDA)
  qda_reg{r}_hv       classical QDA, sklearn reg_param r in {0.01, 0.03, 0.3} (R003 used 0.1): sensitivity
                      of the plug-in model to its shrinkage strength. NOT tuned: the grid is reported in
                      full, and choosing its best value on these folds would be optimistic.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm --no-sync -- research/r256_bqda_controls.py --n-jobs 12
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ssc import paths
from ssc.tracking import Run
from ssc.transductive import SubspaceModel
from ssc.validation import cross_validate

import lib_fm
from lib_fm import BayesQDA

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
a = p.parse_args()
src = a.data
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)
ds, scheme, splits = lib_fm.official_setup(src)

CANDS = {"bqda_eb_plugin_hv": (lambda: SubspaceModel(BayesQDA(predictive="plugin"), "highvar", source=src),
                               "EB NIW prior, Gaussian plug-in at posterior means (no Student-t)"),
         "bqda_eb_widekappa_hv": (lambda: SubspaceModel(BayesQDA(kappa_grid=(0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)),
                                                        "highvar", source=src),
                                  "EB NIW prior with a wider kappa0 grid (R255 chose kappa0 = 1 = grid edge in all folds)"),
         "bqda_weak_plugin_hv": (lambda: SubspaceModel(BayesQDA(mode="fixed", kappa0=0.01, nu0_extra=2.0, prior="iso",
                                                                predictive="plugin"), "highvar", source=src),
                                 "weak fixed NIW prior (as r255_bqda_weak_hv), Gaussian plug-in: almost ML-QDA")}
for r in (0.01, 0.03, 0.3):
    CANDS[f"qda_reg{r}_hv"] = ((lambda r=r: SubspaceModel(QuadraticDiscriminantAnalysis(reg_param=r), "highvar", source=src)),
                               f"classical QDA reg_param={r} on the block (sensitivity, not tuned)")

refs = {rn: lib_fm.r003_reference(rn, ds, splits, scheme, src, n_jobs=n_jobs) for rn in ("r003_sup_qda_hv", "r003_gmm2_hv")}
s003 = refs["r003_gmm2_hv"].test_pred
rows = []
for name, (make, desc) in CANDS.items():
    cfg = {"name": f"r256_{name}", "description": desc, "model_code": "research/r256_bqda_controls.py + research/lib_fm.py",
           "cv": scheme.to_dict(), "test_pred": "fold_mean", "uses_unlabeled_test_features": True,
           "code_sha": lib_fm.code_hash(__file__)}
    hyp = {"bqda_eb_plugin_hv": "If the R255 gain comes from integrating over Sigma, the plug-in version of the same EB "
                                "prior loses most of it (AUC closer to QDA 0.996).",
           "bqda_eb_widekappa_hv": "A wider kappa0 grid moves the EB optimum beyond 1 but changes CV AUC / log-loss by less "
                                   "than 2 split-SE (the result does not hinge on the grid edge).",
           "bqda_weak_plugin_hv": "Near-ML QDA (weak prior, plug-in) is clearly worse than the same prior with the Student-t "
                                  "predictive (r255_bqda_weak_hv 0.9994): the predictive integration matters."}.get(
        name, "Classical QDA's AUC depends on reg_param; if a smaller reg already reaches the Bayesian QDA AUC, the gain "
              "is mostly shrinkage strength/target, not Bayesian averaging.")
    with Run(f"r256_{name}", cfg, kind="experiment", source=src, author=a.author, hypothesis=hyp) as run:
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
        s = run.save_cv(cv)
        cal = lib_fm.oof_calibration(cv.oof, ds.y)
        agr = lib_fm.reference_agreement(cv.test_pred, s003)
        pcs = lib_fm.paired_table({name: cv}, refs)
        run.save_table("paired_vs_references", pd.DataFrame(pcs))
        pc = pcs[0]
        rows.append({"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
                     **cal, "diff_vs_qda0.1": pc["mean_diff"], "se_vs_qda0.1": pc["se_split"], "p_vs_qda0.1": pc["corrected_p"],
                     **{f"s003_{k}": v for k, v in agr.items()}})
        run.finish(extra_summary={"official_cv": True, **cal, "agreement_with_s003": agr, "paired_vs_references": pcs,
                                  "uses_unlabeled_test_features": True})
        print(f"{name:20s} auc={s['auc_mean']:.5f} +/- {s['auc_std_repeat']:.5f} logloss={cal['cv_logloss']:.4f} "
              f"diff_vs_qda0.1={pc['mean_diff']:+.5f} (se {pc['se_split']:.5f}) run={run.run_id}", flush=True)
t = paths.tables_dir(src)
t.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(t / "r256_bqda_controls.csv", index=False)
print(pd.DataFrame(rows)[["candidate", "auc_mean", "auc_std_repeat", "cv_logloss", "diff_vs_qda0.1", "se_vs_qda0.1"]].to_string(index=False))

# EB choices of the wide-kappa variant, per fold and on all 250 rows (cheap refits, analysis output only)
from collections import Counter  # noqa: E402

WIDE = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
ch = Counter()
for sp in splits:
    e = SubspaceModel(BayesQDA(kappa_grid=WIDE), "highvar", source=src).fit(ds.X.to_numpy()[sp.train_idx],
                                                                           ds.y[sp.train_idx]).estimator_
    ch[(e.kappa0_, e.nu0_extra_, e.prior_)] += 1
full = SubspaceModel(BayesQDA(kappa_grid=WIDE), "highvar", source=src).fit(ds.X.to_numpy(), ds.y).estimator_
choice = pd.DataFrame([{"kappa0": k[0], "nu0_extra": k[1], "prior": k[2], "n_folds": v} for k, v in ch.most_common()])
choice.to_csv(t / "r256_eb_widekappa_choice.csv", index=False)
print("wide-kappa EB choices over folds:\n" + choice.to_string(index=False))
print(f"wide-kappa EB on all 250 rows: kappa0={full.kappa0_} nu0_extra={full.nu0_extra_} prior={full.prior_}")
