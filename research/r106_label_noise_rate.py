"""R106 - how much label noise is consistent with the evidence, and what private AUC does it imply?

R103: under the fitted K=2 generator with clean labels S003's public AUC would be <= 0.99923 with
probability 0.001, but with sklearn-style label noise (flip_y = 0.01 -> 0.5 % wrong labels) with
probability 0.83. Here: a grid of flip rates f (each label replaced by a random label with
probability f, so P(wrong) = f/2). For each of n_rep replicas the S003 procedure is fitted once
(the label noise only touches the mapping through 250 labels, negligible), then many independent
flip draws and random 395 / 19,355 public / private splits of the test rows give
  * the distribution of the public AUC -> likelihood of the observed public score (|pub - obs| <= tol)
  * the private AUC (the quantity that ranks the teams)
  * P(no confidently mislabelled training row | f) = P(0 wrong labels among the 250 rows with
    |posterior - 0.5| > 0.49), matching the R100 observation (0 such rows)
Posterior over f with a uniform prior on the grid -> expected private AUC of S003 (and of every
near-Bayes procedure: R103 shows they differ by < 1e-6).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rmix --high-mem --no-sync -- research/r106_label_noise_rate.py --n-jobs 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

import lib_mix as L
from ssc import data, paths
from ssc.tracking import Run

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--n-rep", type=int, default=16)
p.add_argument("--n-flip-draws", type=int, default=60)
p.add_argument("--public-obs", type=float, default=0.99923)
p.add_argument("--tol", type=float, default=0.0004)
a = p.parse_args()
src = a.data
ds = data.load(src)
NT = L.N_TRAIN
FLIPS = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03]
N_PUB = 395


def one(gen, rep):
    rng = np.random.default_rng(50_000 + rep)
    X, comp_y, bayes = gen.sample(rng)
    cols = list(L.highvar_cols(X))
    Z, _, _ = L.standardize(X[:, cols])
    g = GaussianMixture(2, covariance_type="full", n_init=5, random_state=0, reg_covar=1e-6, max_iter=1000).fit(Z)
    R = g.predict_proba(Z)
    s = (R @ L.component_rates(R[:NT], comp_y[:NT]))[NT:]
    c = comp_y[NT:]
    conf_train = np.abs(bayes[:NT] - 0.5) > 0.49
    rows = []
    for f in FLIPS:
        for k in range(a.n_flip_draws):
            y = c.copy()
            m = rng.uniform(size=len(y)) < f
            y[m] = rng.integers(0, 2, size=m.sum())
            perm = rng.permutation(len(y))
            pub, prv = perm[:N_PUB], perm[N_PUB:]
            if len(np.unique(y[pub])) < 2:
                continue
            rows.append({"rep": rep, "flip": f, "draw": k, "public_auc": roc_auc_score(y[pub], s[pub]),
                         "private_auc": roc_auc_score(y[prv], s[prv]), "n_wrong_public": int((y[pub] != c[pub]).sum()),
                         "p_no_conf_wrong_train": float((1 - f / 2) ** conf_train.sum())})
    return rows


hyp = ("The observed public 0.99923 and the absence of confidently mislabelled training rows are jointly most "
       "consistent with a wrong-label rate of roughly 0.1-0.5 % (flip 0.002-0.01); the implied private AUC of S003 "
       "is ~0.995-0.999, far below the clean-label Bayes AUC 0.99995, and identical for all near-Bayes candidates.")
cfg = {"name": "r106_label_noise_rate", "flips": FLIPS, "n_rep": a.n_rep, "n_flip_draws": a.n_flip_draws,
       "public_obs": a.public_obs, "tol": a.tol, "n_public": N_PUB}
with Run("r106_label_noise_rate", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    gen = L.ReplicaGenerator(L.BLOCK, K=2, source=src, y_train=ds.y)
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(gen, r) for r in range(a.n_rep))
    df = pd.DataFrame([x for r in res for x in r])
    run.save_table("draws", df)
    obs = a.public_obs
    g = df.groupby("flip")
    summ = pd.DataFrame({
        "p_wrong_label": [f / 2 for f in g.groups],
        "public_mean": g["public_auc"].mean().to_numpy(),
        "public_p_le_obs": g["public_auc"].apply(lambda v: (v <= obs).mean()).to_numpy(),
        "public_p_ge_obs": g["public_auc"].apply(lambda v: (v >= obs).mean()).to_numpy(),
        "lik_public_near_obs": g["public_auc"].apply(lambda v: (np.abs(v - obs) <= a.tol).mean()).to_numpy(),
        "lik_train_no_conf_wrong": g["p_no_conf_wrong_train"].mean().to_numpy(),
        "private_mean": g["private_auc"].mean().to_numpy(),
        "private_sd": g["private_auc"].std().to_numpy(),
    }, index=list(g.groups)).rename_axis("flip").reset_index()
    summ["joint_lik"] = summ["lik_public_near_obs"] * summ["lik_train_no_conf_wrong"]
    summ["posterior_uniform_grid"] = summ["joint_lik"] / summ["joint_lik"].sum()
    run.save_table("flip_summary", summ)
    tables = paths.tables_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    summ.to_csv(tables / "r106_flip_summary.csv", index=False)
    post = summ["posterior_uniform_grid"].to_numpy()
    out = {"posterior_mean_flip": float((summ["flip"] * post).sum()),
           "posterior_mean_wrong_label_rate": float((summ["p_wrong_label"] * post).sum()),
           "expected_private_auc_s003": float((summ["private_mean"] * post).sum()),
           "private_auc_clean": float(summ.loc[summ.flip == 0, "private_mean"].iloc[0]),
           "map_flip": float(summ.loc[summ["posterior_uniform_grid"].idxmax(), "flip"])}
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(out)
    run.finish(extra_summary=out)
