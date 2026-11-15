"""R103 - parametric bootstrap: which procedure ranks the test rows best when the truth is known?

Generator (lib_mix.ReplicaGenerator, fitted to the real 20,000 x 150 features): K=2 full
Gaussian mixture on the 12-dim PCA coordinates of the 13-feature block (component = class,
mapped by the training labels), every other column a class-independent Gaussian linear
regression on those coordinates with a full residual covariance (keeps the 7 exact
redundancies and the 130 independent noise columns), real means/sds, rounded to 6 decimals.
  G0  clean labels (label = component)
  G1  make_classification-style label noise: each label replaced by a random label with
      probability flip (default 0.01, sklearn's flip_y default) -> 0.5 % wrong labels
Each replica = 20,000 rows; rows 0..249 are 'train' (labels visible), 250.. are 'test'.
Every procedure is run exactly as on the real data (its own label-free steps on the
replica's features, the 250 replica labels) and scored by test AUC on the 19,750 rows.
Also: the generator's Bayes posterior (ceiling), and the public-LB distribution of S003
(395 random test rows) to ask how unusual the real public 0.99923 is under G0 / G1.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rmix --high-mem --no-sync -- research/r103_replica_bootstrap.py --n-rep 40 --n-jobs 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

import lib_mix as L
from ssc import data, paths
from ssc.tracking import Run

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--n-rep", type=int, default=40)
p.add_argument("--flip", type=float, default=0.01)
p.add_argument("--all150-reps", type=int, default=8, help="run the 142-dim GMM only on the first N replicas")
p.add_argument("--public-draws", type=int, default=400)
p.add_argument("--public-obs", type=float, default=0.99923)
p.add_argument("--reg-small", type=float, default=1e-8)
a = p.parse_args()
src = a.data
ds = data.load(src)
NT = L.N_TRAIN


def s003_proc(X, yt):
    cols = list(L.highvar_cols(X))
    Z, _, _ = L.standardize(X[:, cols])
    g = GaussianMixture(2, covariance_type="full", n_init=5, random_state=0, reg_covar=1e-6, max_iter=1000).fit(Z)
    R = g.predict_proba(Z)
    return R @ L.component_rates(R[:NT], yt), cols


def polish_proc(U, yt, reg=1e-6):
    f = L.gmm_fit_arr(U, 2, "full", 5, 0, reg)
    return L.class1_prob(f["post"], np.arange(NT), yt), f


def procedures(X, yt, rep):
    out, info = {}, {}
    t0 = time.time()
    out["s003"], cols = s003_proc(X, yt)
    info["highvar_is_block"] = tuple(cols) == L.BLOCK
    Ub = L.basis_arr(X, cols)
    out["block_polish"], fb = polish_proc(Ub, yt)
    out["block_polish_reg_small"], _ = polish_proc(Ub, yt, a.reg_small)
    S_hat, hist = L.discover_arr(X, start=cols, max_iter=4)
    info["discover_n_added"] = len(S_hat) - len(cols)
    out["discover_polish"] = out["block_polish"] if len(S_hat) == len(cols) else polish_proc(L.basis_arr(X, S_hat), yt)[0]
    acc = 0
    for s in range(10):
        g = GaussianMixture(2, covariance_type="full", n_init=1, random_state=s, reg_covar=1e-6, max_iter=1000).fit(Ub)
        R = g.predict_proba(Ub)
        acc = acc + R @ L.component_rates(R[:NT], yt)
    out["seed10"] = acc / 10
    out["bag10"] = L.class1_prob(L.bagged_post_arr(Ub, fb, 10, rep), np.arange(NT), yt)
    p0 = out["block_polish"]
    cc = np.array([0, 1]) if L.component_rates(fb["post"][:NT], yt)[1] > 0.5 else np.array([1, 0])
    for name, eps in (("semisup_eps01", 0.01), ("semisup_hard", 0.0)):
        f = L.em(Ub, fb["post"], "full", 1e-6, tol=1e-10, label_idx=np.arange(NT), label_cls=yt, comp_class=cc, eps=eps)
        out[name] = f["post"][:, cc == 1].sum(1)
    f4 = L.gmm_fit_arr(Ub, 4, "full", 5, 0, 1e-6, max_iter=300)
    rate4 = L.component_rates(f4["post"][:NT], yt)
    cc4 = np.zeros(4, int)
    cc4[np.argsort(rate4)[::-1][:2]] = 1
    f = L.em(Ub, f4["post"], "full", 1e-6, max_iter=300, tol=1e-10, label_idx=np.arange(NT), label_cls=yt, comp_class=cc4, eps=0.01)
    out["semisup_2pc"] = f["post"][:, cc4 == 1].sum(1)
    out["gmm4_rates"] = f4["post"] @ rate4
    out["selftrain_qda99"] = L.selftrain_qda_arr(Ub, p0, np.arange(NT), yt, 0.99, 5)
    out["quadlogit_pseudo"] = L.quad_logit_arr(Ub, p0, np.arange(NT), yt, 1.0)
    Z_all, _, _ = L.standardize(X)
    res = L.class_dependence(Z_all[:NT], yt.astype(float), None)
    sel = sorted(int(j) for j in np.argsort(-res["lrt"])[:13])
    info["supsel_overlap_block"] = len(set(sel) & set(cols))
    out["supsel13"] = polish_proc(L.basis_arr(X, sel), yt)[0]
    Zb, _, _ = L.standardize(X[:, cols])
    out["sup_qda"] = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(Zb[:NT], yt).predict_proba(Zb)[:, 1]
    g = GaussianMixture(2, covariance_type="tied", n_init=5, random_state=0, reg_covar=1e-6).fit(Ub)
    R = g.predict_proba(Ub)
    out["tied"] = R @ L.component_rates(R[:NT], yt)
    if rep < a.all150_reps:
        Ua = L.basis_arr(X, list(range(150)))
        g = GaussianMixture(2, covariance_type="full", n_init=1, random_state=0, reg_covar=1e-6, max_iter=1000).fit(Ua)
        R = g.predict_proba(Ua)
        out["all150"] = R @ L.component_rates(R[:NT], yt)
    info["seconds"] = time.time() - t0
    return out, info


def one_replica(gen, gname, rep, flip):
    rng = np.random.default_rng(10_000 * (1 + (gname == "G1")) + rep)
    X, comp_y, bayes = gen.sample(rng)
    y = comp_y.copy()
    if flip > 0:
        m = rng.uniform(size=len(y)) < flip
        y[m] = rng.integers(0, 2, size=m.sum())
    yt = y[:NT]
    if len(np.unique(yt)) < 2:
        return []
    scores, info = procedures(X, yt, rep)
    scores["bayes"] = bayes
    yte, cte, bte = y[NT:], comp_y[NT:], bayes[NT:]
    bnd = (bte > 0.1) & (bte < 0.9)
    rows = []
    for name, s in scores.items():
        st = s[NT:]
        rows.append({"gen": gname, "rep": rep, "proc": name, "test_auc": roc_auc_score(yte, st),
                     "test_auc_vs_component": roc_auc_score(cte, st),
                     "expected_auc_under_bayes": L.expected_auc(st, bte),
                     "kendall_bayes_boundary": float(kendalltau(st[bnd], bte[bnd])[0]) if bnd.sum() > 2 else np.nan,
                     "n_boundary_bayes": int(bnd.sum()), "n_wrong_labels_test": int((yte != cte).sum()), **info})
    # public-LB distribution of S003 on random 395-row subsets of the test rows
    pr = np.random.default_rng(rep + 777)
    s3 = scores["s003"][NT:]
    pub = []
    for _ in range(a.public_draws):
        ix = pr.choice(len(yte), 395, replace=False)
        if len(np.unique(yte[ix])) == 2:
            pub.append(roc_auc_score(yte[ix], s3[ix]))
    pub = np.array(pub)
    for r in rows:
        if r["proc"] == "s003":
            r.update({"public_mean": float(pub.mean()), "public_p_le_obs": float((pub <= a.public_obs).mean()),
                      "public_p_eq_1": float((pub >= 1 - 1e-12).mean())})
    return rows


hyp = ("Under the fitted generator the S003 procedure is within 2e-5 test AUC of the Bayes ceiling; refinements "
       "(polishing, restarts, bagging, semi-supervised EM, discovery) differ from S003 by < 1e-5 (not detectable on a "
       "19,355-row private LB); self-training QDA, supervised selection, tied covariance, supervised QDA and the 142-dim "
       "mixture are measurably worse. Without label noise a public AUC <= 0.99923 is rare; with 0.5 % wrong labels "
       "it is typical, i.e. the residual public error is label noise, not boundary ranking.")
cfg = {"name": "r103_replica_bootstrap", "generator": "lib_mix.ReplicaGenerator(BLOCK, K=2)", "n_rep": a.n_rep,
       "flip": a.flip, "all150_reps": a.all150_reps, "public_draws": a.public_draws, "public_obs": a.public_obs,
       "reg_small": a.reg_small}
with Run("r103_replica_bootstrap", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    gen = L.ReplicaGenerator(L.BLOCK, K=2, source=src, y_train=ds.y)
    out = {"generator_weights": gen.fit["weights"].tolist(), "generator_comp_class": gen.comp_class.tolist()}
    jobs = [("G0", r, 0.0) for r in range(a.n_rep)] + [("G1", r, a.flip) for r in range(a.n_rep)]
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_replica)(gen, g, r, f) for g, r, f in jobs)
    rows = [x for r in res for x in r]
    df = pd.DataFrame(rows)
    run.save_table("replica_scores", df)
    out["wall_seconds"] = time.time() - t0
    # paired differences vs s003 per generator
    summ = []
    for gname, g in df.groupby("gen"):
        base = g[g.proc == "s003"].set_index("rep")
        for proc, gp in g.groupby("proc"):
            gp = gp.set_index("rep")
            common = gp.index.intersection(base.index)
            for metric in ("test_auc", "expected_auc_under_bayes"):
                dlt = gp.loc[common, metric] - base.loc[common, metric]
                summ.append({"gen": gname, "proc": proc, "metric": metric, "n_rep": len(common),
                             "mean": float(gp.loc[common, metric].mean()), "mean_diff_vs_s003": float(dlt.mean()),
                             "se_diff": float(dlt.std(ddof=1) / np.sqrt(len(dlt))) if len(dlt) > 1 else np.nan,
                             "frac_better": float((dlt > 0).mean() + 0.5 * (dlt == 0).mean()),
                             "kendall_bayes_boundary": float(gp.loc[common, "kendall_bayes_boundary"].mean())})
    summ = pd.DataFrame(summ).sort_values(["gen", "metric", "mean"], ascending=[True, True, False])
    run.save_table("replica_summary", summ)
    tables = paths.tables_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    summ.to_csv(tables / "r103_replica_summary.csv", index=False)
    pubs = df[df.proc == "s003"].groupby("gen")[["public_mean", "public_p_le_obs", "public_p_eq_1"]].mean()
    run.save_table("public_lb_s003", pubs.reset_index())
    pubs.reset_index().to_csv(tables / "r103_public_lb_s003.csv", index=False)
    out["public_lb_s003"] = pubs.reset_index().to_dict("records")
    out["highvar_is_block_frac"] = float(df[df.proc == "s003"]["highvar_is_block"].mean())
    out["discover_n_added_mean"] = float(df[df.proc == "s003"]["discover_n_added"].mean())
    out["supsel_overlap_block_mean"] = float(df[df.proc == "s003"]["supsel_overlap_block"].mean())
    out["n_boundary_bayes_mean"] = float(df[df.proc == "s003"]["n_boundary_bayes"].mean())
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(pubs.to_string())
    print({k: v for k, v in out.items() if not isinstance(v, list)})
    run.finish(extra_summary=out)
