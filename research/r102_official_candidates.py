"""R102 - official-CV candidates for ranking the ~1 % boundary rows (ideas 1-5 of the mix line).

Every candidate = one experiment run under the OFFICIAL protocol (project.yaml official_cv,
identical splits), with test predictions (fold_mean), so any of them can become a submission.
CV AUC is saturated (1.0), so each run also stores evidence that does not rely on it:
  * CV log-loss / Brier of the out-of-fold scores, training rows on the wrong side of 0.5
  * label-free BIC of the underlying mixture (where there is one)
  * how the test ranking re-orders the S003 boundary rows (S003 prediction in (0.1, 0.9)):
    Spearman / Kendall on them, side flips
  * expected test AUC if test labels were Bernoulli(reference posterior), reference = polished
    K=2 full block GMM (a label-free private-LB proxy; circular for models close to the reference)

Candidates (labels enter only through the fold's training rows):
  ref_s003            reproduction of R003 gmm2_hv (TransductiveGMM, = S003) for the extra metrics
  gmm2_block_polish   K=2 full GMM, block in its non-degenerate PCA basis, EM polished to convergence
  gmm2_conn20_polish  same on the 20 label-free 'connected' features of R100 (same 12-dim span)
  gmm2_block_seed10   ensemble of 10 restarts (n_init=1 each, sklearn tol)
  gmm2_block_bag20    bootstrap-aggregated mixture posterior (20 resamples of the 20,000 rows)
  semisup_eps01       semi-supervised EM, labels as soft constraints (label-noise prob 0.01)
  semisup_hard        semi-supervised EM, hard label constraints
  semisup_2pc         semi-supervised EM, 2 components per class
  selftrain_qda99     self-training QDA on confident pseudo-labels (|p - 0.5| > 0.49), 5 rounds
  quadlogit_pseudo    quadratic logistic regression on soft pseudo-labels of all 20,000 rows
  gmm2_block_tied     tied covariance (BIC-rejected; negative control)
  gmm2_block_diag     diagonal covariance (BIC-rejected; negative control)
  supsel13_gmm2       supervised in-fold selection (top-13 marginal LRT on fold labels) + K=2 GMM

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rmix --high-mem --no-sync -- research/r102_official_candidates.py --n-jobs 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

import lib_mix as L
from ssc import data, paths
from ssc.tracking import Run
from ssc.transductive import TransductiveGMM
from ssc.validation import Scheme, cross_validate, make_splits

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--only", nargs="*", default=None)
a = p.parse_args()
src = a.data

official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
scheme = Scheme.from_dict(official)
ds = data.load(src)
splits = make_splits(ds.y, scheme)
B = L.BLOCK

CANDIDATES = {
    "ref_s003": (lambda: TransductiveGMM("highvar", 2, source=src), "reproduction of R003 gmm2_hv = S003",
                 "Reproduces the S003 test predictions exactly (max |diff| < 1e-9); gives its CV log-loss/Brier."),
    "gmm2_block_pca_polish": (lambda: L.MixGMM(B, 2, polish=True, source=src),
                              "K=2 full GMM on the block's 12-dim PCA basis, EM polished to tol 1e-10",
                              "Same optimum as S003 (R101 restarts): boundary ranking Kendall > 0.99 vs S003, CV log-loss within 0.002."),
    "gmm2_conn20_polish": (lambda: L.MixGMM(L.CONN20, 2, polish=True, source=src),
                           "K=2 full GMM on the 20 label-free connected features (PCA, 12 dims)",
                           "Same 12-dim span as the block (R100), so identical ranking up to rounding noise."),
    "gmm2_block_seed10": (lambda: L.MixGMM(B, 2, n_init=1, seeds=tuple(range(10)), polish=False, source=src),
                          "ensemble of 10 GMM restarts (n_init=1, sklearn default tol)",
                          "All restarts reach the same optimum: ensemble ranking = S003 ranking (Kendall > 0.99)."),
    "gmm2_block_bag20": (lambda: L.PostTableModel(B, table="bagged", B=20, source=src),
                         "bagged mixture posterior (20 bootstrap resamples of the 20,000 rows)",
                         "Bagging reflects parameter uncertainty; changes boundary posteriors by < 0.02 and the ranking slightly."),
    "semisup_eps01": (lambda: L.SemiSupEM(B, 1, eps=0.01, source=src),
                      "semi-supervised EM, fold labels as soft constraints (eps 0.01), K=2",
                      "250 labels vs 20,000 rows: parameters move negligibly; boundary Kendall vs S003 > 0.98."),
    "semisup_hard": (lambda: L.SemiSupEM(B, 1, eps=0.0, source=src),
                     "semi-supervised EM, hard label constraints, K=2",
                     "As semisup_eps01; hard constraints force boundary training rows, still negligible."),
    "semisup_2pc": (lambda: L.SemiSupEM(B, 2, eps=0.01, source=src),
                    "semi-supervised EM, 2 components per class (K=4)",
                    "BIC prefers K=2; extra components overfit and lower the expected AUC slightly."),
    "selftrain_qda99": (lambda: L.PostTableModel(B, mode="selftrain", thr=0.99, n_iter=5, source=src),
                        "self-training QDA on confident pseudo-labels (|p-0.5|>0.49), 5 rounds",
                        "Fitting on the confident core shrinks the class covariances: a small but measurable loss in expected AUC."),
    "quadlogit_pseudo": (lambda: L.PostTableModel(B, mode="quadlogit", C=1.0, source=src),
                         "quadratic logistic regression on soft pseudo-labels of all 20,000 rows",
                         "Distils the GMM log-odds; ranking close to S003 (Kendall > 0.95), no gain."),
    "gmm2_block_tied": (lambda: L.MixGMM(B, 2, cov_type="tied", polish=False, source=src),
                        "K=2 tied-covariance GMM (negative control)",
                        "BIC-rejected; the class difference is in the covariance, so CV AUC < 1 and expected AUC far lower."),
    "gmm2_block_diag": (lambda: L.MixGMM(B, 2, cov_type="diag", polish=False, source=src),
                        "K=2 diagonal GMM (negative control)",
                        "BIC-rejected (correlated block); clearly worse."),
    "supsel13_gmm2": (lambda: L.SupSelectGMM(13, source=src),
                      "supervised in-fold selection (top-13 marginal LRT on fold labels) + K=2 GMM",
                      "Supervised selection misses jointly-informative block features (R100), so the GMM on the selected "
                      "set is worse than the label-free block (lower CV log-loss quality, lower expected AUC)."),
}
names = a.only or list(CANDIDATES)

s3 = L.s003_pred(src)
ref_post, _, _ = L.polished_posterior(B, 2, "full", 5, 0, 1e-6, src)
ref_test = L.oriented(ref_post, ds.y)[250:]
bayes_ref = L.expected_auc(ref_test, ref_test)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)

rows = []
for name in names:
    make, desc, hyp = CANDIDATES[name]
    cfg = {"name": f"r102_{name}", "description": desc, "model_code": "research/r102_official_candidates.py + research/lib_mix.py",
           "cv": scheme.to_dict(), "test_pred": "fold_mean", "uses_unlabeled_test_features": name != "none"}
    with Run(f"r102_{name}", cfg, kind="experiment", source=src, author=a.author, hypothesis=hyp) as run:
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=a.n_jobs)
        s = run.save_cv(cv)
        tp = cv.test_pred
        extra = {"official_cv": True, **L.cv_extra_metrics(cv, ds.y), **L.boundary_stats(tp, s3),
                 "expected_auc_under_ref": L.expected_auc(tp, ref_test), "bayes_auc_ref": bayes_ref,
                 "max_abs_diff_vs_s003": float(np.abs(tp - s3).max()),
                 "uses_unlabeled_test_features": True}
        run.finish(extra_summary=extra)
        rows.append({"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
                     **{k: extra[k] for k in ("cv_logloss", "cv_brier", "cv_n_wrong_side_mean", "n_boundary_self",
                                              "spearman_boundary", "kendall_boundary", "n_side_flips_all",
                                              "expected_auc_under_ref")}})
        print(pd.DataFrame(rows[-1:]).to_string(index=False), flush=True)

res = pd.DataFrame(rows)
suffix = "" if a.only is None else "_" + "_".join(names)[:60]
res.to_csv(tables / f"r102_candidates{suffix}.csv", index=False)
print("\n" + res.to_string(index=False))
print(f"\nreference Bayes AUC (expected AUC of the reference ranking under itself): {bayes_ref:.6f}")
