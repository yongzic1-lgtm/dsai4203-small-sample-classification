"""R003 - does modelling the unlabeled feature structure beat supervised learning on 250 rows?

Each candidate is evaluated with the OFFICIAL protocol (project.yaml official_cv, identical
splits for all candidates -> paired comparison) and archived as its own experiment run with
test predictions, so any of them can become a submission.

Candidates
  sup_l1_all        L1 logistic regression on all 150 features (supervised baseline)
  sup_lda_hv        shrinkage LDA on the label-free high-variance block
  sup_qda_hv        regularised QDA on the high-variance block
  sup_logreg_hv     L2 logistic regression on the high-variance block
  gmm2_hv           TransductiveGMM, 2 full-covariance components on the high-variance block
  gmm2_hv_pca       same in the PCA basis of the block (d = rank of the block)
  gmm4_hv           4 components (2 per class if the generator uses 2 clusters per class)
  gmm2_all          2 components on all 150 features (does noise hurt the mixture?)

    .venv\\Scripts\\python.exe colab\\remote.py run -- research/r003_transductive_vs_supervised.py --n-jobs 2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from ssc import data, paths
from ssc.models import build_pipeline
from ssc.tracking import Run
from ssc.transductive import SubspaceModel, TransductiveGMM
from ssc.validation import Scheme, cross_validate, make_splits, paired_compare

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--only", nargs="*", default=None, help="subset of candidate names")
a = p.parse_args()

official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
scheme = Scheme.from_dict(official)
ds = data.load(a.data)
splits = make_splits(ds.y, scheme)
src = a.data

CANDIDATES = {
    "sup_l1_all": (lambda: build_pipeline({"model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 1.0}}}),
                   "supervised L1 logistic, all 150 features, C=0.1"),
    "sup_lda_hv": (lambda: SubspaceModel(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "highvar", source=src),
                   "supervised shrinkage LDA on high-variance block"),
    "sup_qda_hv": (lambda: SubspaceModel(QuadraticDiscriminantAnalysis(reg_param=0.1), "highvar", source=src),
                   "supervised QDA (reg 0.1) on high-variance block"),
    "sup_logreg_hv": (lambda: SubspaceModel(LogisticRegression(C=1.0, max_iter=5000), "highvar", source=src),
                      "supervised L2 logistic on high-variance block"),
    "gmm2_hv": (lambda: TransductiveGMM("highvar", 2, source=src), "transductive GMM K=2 full, high-variance block"),
    "gmm2_hv_pca": (lambda: TransductiveGMM("highvar", 2, pca_dims=12, source=src), "transductive GMM K=2 full, PCA(12) of block"),
    "gmm4_hv": (lambda: TransductiveGMM("highvar", 4, source=src), "transductive GMM K=4 full, high-variance block"),
    "gmm2_all": (lambda: TransductiveGMM("all", 2, n_init=2, source=src), "transductive GMM K=2 full, all 150 features"),
}
names = a.only or list(CANDIDATES)

rows, folds = [], {}
for name in names:
    make, desc = CANDIDATES[name]
    cfg = {"name": f"r003_{name}", "description": desc, "model_code": "research/r003_transductive_vs_supervised.py",
           "cv": scheme.to_dict(), "test_pred": "fold_mean",
           "uses_unlabeled_test_features": name.startswith("gmm") or "_hv" in name}
    with Run(f"r003_{name}", cfg, kind="experiment", source=src, author=a.author,
             hypothesis="unlabeled-structure models beat 250-row supervised models by a wide margin") as run:
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=a.n_jobs)
        s = run.save_cv(cv)
        folds[name] = cv.fold_scores
        rows.append({"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"],
                     "auc_std_repeat": s["auc_std_repeat"], "auc_std_fold": s["auc_std_fold"],
                     "oof_auc_mean": s.get("oof_auc_mean"), "description": desc})
        print(f"{name:14s} auc={s['auc_mean']:.4f} +/- {s['auc_std_repeat']:.4f} (repeat sd)  oof={s.get('oof_auc_mean', float('nan')):.4f}  run={run.run_id}")

res = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
best = res.iloc[0]["candidate"]
pc = []
for name in res["candidate"]:
    if name != best:
        c = paired_compare(folds[best], folds[name])
        pc.append({"best": best, "vs": name, **c})
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
res.to_csv(tables / "r003_candidates.csv", index=False)
pd.DataFrame(pc).to_csv(tables / "r003_paired_vs_best.csv", index=False)
print("\n" + res[["candidate", "auc_mean", "auc_std_repeat", "oof_auc_mean", "run_id"]].to_string(index=False))
print("\npaired vs best:\n" + pd.DataFrame(pc)[["vs", "mean_diff", "std_diff_repeat", "win_rate_fold", "corrected_p"]].to_string(index=False))
