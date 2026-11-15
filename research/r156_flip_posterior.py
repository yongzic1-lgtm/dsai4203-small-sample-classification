"""R156 - posterior over the label-noise level flip_y and the expected private AUC of S003.

Combines two independent pieces of evidence on a grid of flip_y values:
  (1) the 250 train labels: P(y_i | x_i, flip_y) = (1 - flip_y) * pi_i + flip_y / 2 with pi_i the label-free
      GMM posterior (recomputed here exactly as in R150);
  (2) the public LB score of S003 (0.99923): ABC likelihood P(|public - 0.99923| <= 0.00015 | flip_y) from the
      parametric replica simulation R153 (run 20261022-073058_r153_param, table flip_abc), passed with --abc.
and the R153 posterior predictive of the private AUC given a public score near S003's for each flip level
(--pred, same run, table s003_private_predictive). Prior: uniform over the grid.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen -- research/r156_flip_posterior.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.0025, 0.005, 0.01])
# from 20261022-073058_r153_param: flip_abc.p_near_s003 and s003_private_predictive.private_mean / q05 / q95
p.add_argument("--abc", type=float, nargs="*", default=[0.0025, 0.0179, 0.0216, 0.0172])
p.add_argument("--pred-mean", type=float, nargs="*", default=[0.9999521, 0.9986798, 0.9973724, 0.9949027])
p.add_argument("--source-run", default="20261022-073058_r153_param")
a = p.parse_args()

X = unlabeled_features(a.data)
y = data.load(a.data).y
n_tr = len(y)
tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r156_flip_posterior", "flips": a.flips, "abc_p_near_s003": a.abc, "pred_private_mean": a.pred_mean,
       "abc_source_run": a.source_run, "prior": "uniform over the grid"}
HYP = ("The train labels alone weakly favour flip_y = 0 (no confident contradiction), the S003 public score favours "
       "flip_y > 0; combined, flip_y is most likely 0.0025-0.005 and S003's expected private AUC is ~0.997-0.998.")
with Run("r156_flip_posterior", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    st = G.identify_structure(X)
    post = st["posterior"]
    if post[:n_tr][y == 1].mean() < post[:n_tr][y == 0].mean():
        post = 1 - post
    pi = np.clip(post[:n_tr], 1e-12, 1 - 1e-12)
    ll = np.array([np.sum(np.log(np.where(y == 1, (1 - f) * pi + f / 2, (1 - f) * (1 - pi) + f / 2))) for f in a.flips])
    lik_train = np.exp(ll - ll.max())
    abc = np.asarray(a.abc)
    rows = pd.DataFrame({"flip_y": a.flips, "loglik_train": ll, "lik_train_rel": lik_train, "abc_p_near_s003": abc,
                         "post_train_only": lik_train / lik_train.sum(), "post_public_only": abc / abc.sum()})
    comb = lik_train * abc
    rows["post_combined"] = comb / comb.sum()
    rows["pred_private_mean_given_s003"] = a.pred_mean
    exp_priv = float((rows.post_combined * rows.pred_private_mean_given_s003).sum())
    exp_flip = float((rows.post_combined * rows.flip_y).sum())
    run.save_table("flip_posterior", rows)
    rows.to_csv(tables / "r156_flip_posterior.csv", index=False)
    (tables / "r156_flip_posterior.md").write_text(rows.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    out = {"posterior_combined": dict(zip(map(str, a.flips), rows.post_combined.round(4))),
           "posterior_mean_flip_y": exp_flip, "expected_private_auc_s003": exp_priv,
           "p_flip_gt_0": float(rows.post_combined[rows.flip_y > 0].sum())}
    run.finish(extra_summary=out)
    print(rows.to_string(index=False))
    for k, v in out.items():
        print(f"{k}: {v}")
