"""R155 - candidate submissions suggested by the replica study: converged EM / informative-column mixture.

R150 showed that the TransductiveGMM posterior of S003 (gmm2_hv, sklearn default EM tol=1e-3) differs from
the converged mixture on the same 12-dim subspace by up to 0.013 on boundary rows. Two candidates, each
evaluated with the OFFICIAL protocol (project.yaml official_cv) and archived as an experiment run with test
predictions (so the main session can submit them):
  r155_gmm2_inf_tight  TransductiveGMMTight(columns="informative")  12 label-free identified informative
                       columns (R150), K=2 full, EM tol 1e-8
  r155_gmm2_hv_tight   TransductiveGMMTight(columns="highvar")      the S003 block, EM tol 1e-8
CV AUC is saturated (1.0) for both, so the run also records label-free evidence: log-likelihood/BIC of the
mixture and agreement with the S003-equivalent posterior on the test boundary rows (number of test rows with
0.01 < p < 0.99, Spearman, how many test-row pairs change order). Replica evidence: R153.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen --high-mem -- research/r155_candidate_gmm_tight.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
a = p.parse_args()

scheme = Scheme.from_dict(yaml.safe_load(open(paths.PROJECT_YAML, encoding="utf-8"))["official_cv"])
ds = data.load(a.data)
splits = make_splits(ds.y, scheme)
X_all = unlabeled_features(a.data)
n_tr = len(ds.y)
ref = G.UnsupervisedGMM(X_all, "highvar", 2).fit()          # = ssc TransductiveGMM gmm2_hv (S003) mixture
ref_p = ref.scores(np.arange(n_tr), ds.y)[n_tr:]
tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)

CANDS = {
    "gmm2_inf_tight": dict(columns="informative"),
    "gmm2_hv_tight": dict(columns="highvar"),
}
rows = []
for name, kw in CANDS.items():
    est = G.TransductiveGMMTight(source=a.data, **kw)
    cfg = {"name": f"r155_{name}", "model_code": "research/lib_gen_replica.py:TransductiveGMMTight",
           "params": est.get_params(), "cv": scheme.to_dict(), "test_pred": "fold_mean",
           "uses_unlabeled_test_features": True}
    hyp = ("Same CV AUC as S003 (saturated 1.0); converged EM gives the MLE posterior; on parametric replicas "
           "(R153) the converged/informative-column mixture is at least as good as gmm2_hv on the private test, "
           "so this is a safe alternative submission. It differs from S003 only in the order of a few boundary rows.")
    with Run(f"r155_{name}", cfg, kind="experiment", source=a.data, author=a.author, hypothesis=hyp) as run:
        cv = cross_validate(est, ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=a.n_jobs)
        s = run.save_cv(cv)
        g = est._unsup() if hasattr(est, "_unsup") else None
        tp = cv.test_pred
        bnd = (ref_p > 0.01) & (ref_p < 0.99)
        bnd2 = (tp > 0.01) & (tp < 0.99)
        both = bnd | bnd2
        # pairwise order changes among the boundary rows of either model
        ia, ib = np.triu_indices(int(both.sum()), 1)
        ra, rb = ref_p[both], tp[both]
        changes = int(((np.sign(ra[ia] - ra[ib]) * np.sign(rb[ia] - rb[ib])) < 0).sum())
        extra = {"official_cv": True, "columns": [int(c) for c in g.cols_], "dims": int(g.dims_),
                 "em_converged": g.converged_, "em_n_iter": g.n_iter_, "ref_em_n_iter": ref.n_iter_,
                 "loglik_mixture": g.loglik, "bic_mixture": g.bic,
                 "test_boundary_0.01_0.99": int(bnd2.sum()), "ref_test_boundary_0.01_0.99": int(bnd.sum()),
                 "max_abs_diff_vs_s003_equiv": float(np.abs(tp - ref_p).max()),
                 "spearman_boundary_vs_s003_equiv": float(stats.spearmanr(ra, rb).statistic),
                 "pair_order_changes_boundary": changes, "n_boundary_union": int(both.sum()),
                 "n_side_changes_0.5": int(((tp > 0.5) != (ref_p > 0.5)).sum())}
        run.finish(extra_summary=extra)
        rows.append({"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
                     **{k: v for k, v in extra.items() if k != "columns"}})
        print(name, run.run_id, {k: v for k, v in extra.items() if k != "columns"}, flush=True)
res = pd.DataFrame(rows)
res.to_csv(tables / "r155_candidates.csv", index=False)
print(res.to_string(index=False))
