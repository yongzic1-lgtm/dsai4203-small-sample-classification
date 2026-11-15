"""R921 (verifier "gen") - independent re-computation of "S003 is ~2-3e-7 AUC below the Bayes AUC" on replicas.

Claims checked: gen.md (R153, 100 parametric replicas: regret 3.1e-7, sd 7.5e-7) and mix.md (R103, 40 replicas: 1.7e-7).
Own code only (no lib_gen_replica, no lib_mix):

  param  parametric replica of the real data: label-free K=2 full GMM (converged) in the 12-dim PCA basis of the
         linearly structured columns (|corr| > 0.1); all structured columns rebuilt exactly from the latent coordinates,
         the other columns iid N(real mean, real sd^2); 10,000 + 10,000 rows, rounded to 6 decimals, random 250 train /
         19,750 test rows, random 395 public rows. Truth = the fitted model, so the true Bayes log-odds is known.
  mc     own re-implementation of sklearn make_classification(n_informative=12, n_redundant=8, n_repeated=0,
         n_clusters_per_class=1, hypercube=True, class_sep=0.71) with the centroid pair conditioned on hamming
         distance 4 (as in the real data), + per-column affine map a~U(0.5,3), b~U(-5,5), column permutation, rounding.

Procedure = S003: high-variance block (std > 1.5 x median std), standardised with train+test statistics, sklearn
GaussianMixture(2, full, n_init=5, random_state=0, reg_covar=1e-6, max_iter=1000, default tol), score = R @ smoothed
component rates from the 250 train labels (the fold mean of S003 is a monotone transform of the same R, so it ranks
identically); "s003_r8" also rounds the score to 8 decimals as in the submitted csv (creates the large tie groups).

Per replica (private part, 19,355 rows): observed AUC with clean labels and with flip_y = 0.01 labels (one draw), and
the label-free "expected AUC" (Rao-Blackwellised over the labels: E[#concordant] / E[n1 n0] with P(class 1) = true
posterior), which removes the label-sampling noise from the paired regret. MC standard errors = sd / sqrt(n_replicas).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vgen --high-mem -- research/r921_bayes_regret_replicas.py --n-param 400 --n-mc 400
"""

from __future__ import annotations

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--n-param", type=int, default=400)
p.add_argument("--n-mc", type=int, default=400)
p.add_argument("--class-sep", type=float, default=0.71)
p.add_argument("--flip", type=float, default=0.01)
p.add_argument("--quick", action="store_true")
a = p.parse_args()
if a.quick:
    a.n_param, a.n_mc = 2, 2
N_TR, N_PUB, NK = 250, 395, 10000


def mvn_logpdf(X, mu, cov):
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (sol ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * np.log(2 * np.pi)


def auc_ties(y, s):
    y = np.asarray(y, bool)
    r = stats.rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def expected_auc(pi, s):
    """E[concordant pairs] / E[n1 n0] over labels ~ Bern(pi) independently, ties count 1/2, i != j."""
    o = np.argsort(s, kind="stable")
    s, pi = s[o], pi[o]
    new = np.r_[True, s[1:] != s[:-1]]
    st = np.flatnonzero(new)
    P = np.add.reduceat(pi, st)
    Q = np.add.reduceat(1 - pi, st)
    PQ = np.add.reduceat(pi * (1 - pi), st)
    below = np.cumsum(Q) - Q
    num = (P * below).sum() + 0.5 * ((P * Q).sum() - PQ.sum())
    den = pi.sum() * (1 - pi).sum() - (pi * (1 - pi)).sum()
    return float(num / den)


def s003_scores(X, y_tr):
    std = X.std(0)
    hv = np.where(std > 1.5 * np.median(std))[0]
    Z = (X[:, hv] - X[:, hv].mean(0)) / X[:, hv].std(0)
    gm = GaussianMixture(2, covariance_type="full", n_init=5, random_state=0, reg_covar=1e-6, max_iter=1000).fit(Z)
    R = gm.predict_proba(Z)
    Rt = R[:N_TR]
    p0 = y_tr.mean()
    rates = (Rt.T @ y_tr + p0) / (Rt.sum(0) + 1.0)
    return np.clip(R @ rates, 0, 1), hv, int(gm.n_iter_)


# ----------------------------------------------------------------------------- generators
def fit_param(X_all, y_train):
    mu, sd = X_all.mean(0), X_all.std(0)
    Z = (X_all - mu) / sd
    C = (Z.T @ Z) / len(Z)
    off = np.abs(C - np.diag(np.diag(C))).max(1)
    S = np.where(off > 0.1)[0]
    ev, V = np.linalg.eigh(C[np.ix_(S, S)])
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    r = int((ev > 1e-9 * ev[0]).sum())
    V = V[:, :r]
    W = Z[:, S] @ V
    gm = GaussianMixture(2, covariance_type="full", n_init=1 if a.quick else 10, tol=1e-10, reg_covar=1e-10,
                         max_iter=20000, random_state=1).fit(W)
    lp = gm._estimate_weighted_log_prob(W)
    k1 = 1 if lp[:N_TR][y_train == 1, 1].mean() - lp[:N_TR][y_train == 1, 0].mean() > 0 else 0
    recon = np.abs(Z[:, S] - W @ V.T).max()
    return {"mu": mu, "sd": sd, "S": S, "V": V, "r": r, "means": gm.means_[[1 - k1, k1]],
            "covs": gm.covariances_[[1 - k1, k1]], "recon_max_abs": float(recon),
            "noise": np.setdiff1d(np.arange(X_all.shape[1]), S)}


def gen_param(seed, P):
    rng = np.random.default_rng([921, 1, int(seed)])
    r = P["r"]
    W = np.vstack([rng.multivariate_normal(P["means"][k], P["covs"][k], size=NK, method="cholesky") for k in (0, 1)])
    cl = np.r_[np.zeros(NK, int), np.ones(NK, int)]
    lo = mvn_logpdf(W, P["means"][1], P["covs"][1]) - mvn_logpdf(W, P["means"][0], P["covs"][0])
    X = np.empty((2 * NK, len(P["mu"])))
    X[:, P["S"]] = P["mu"][P["S"]] + P["sd"][P["S"]] * (W @ P["V"].T)
    X[:, P["noise"]] = P["mu"][P["noise"]] + P["sd"][P["noise"]] * rng.standard_normal((2 * NK, len(P["noise"])))
    del r
    return X, cl, lo, rng, {"structured_cols": P["S"]}


def gen_mc(seed, cs):
    rng = np.random.default_rng([921, 2, int(seed)])
    r, nred, p = 12, 8, 150
    v0 = rng.choice([-1.0, 1.0], r)
    v1 = v0.copy()
    v1[rng.choice(r, 4, replace=False)] *= -1
    cent = cs * np.vstack([v0, v1])
    A = [2 * rng.random((r, r)) - 1 for _ in range(2)]
    U = np.vstack([rng.standard_normal((NK, r)) @ A[k] + cent[k] for k in (0, 1)])
    cl = np.r_[np.zeros(NK, int), np.ones(NK, int)]
    lo = mvn_logpdf(U, cent[1], A[1].T @ A[1]) - mvn_logpdf(U, cent[0], A[0].T @ A[0])
    Bm = 2 * rng.random((r, nred)) - 1
    X = np.hstack([U, U @ Bm, rng.standard_normal((2 * NK, p - r - nred))])
    perm = rng.permutation(p)
    X = X[:, perm] * rng.uniform(0.5, 3.0, p) + rng.uniform(-5, 5, p)
    # column roles after permutation: perm[j] = original index of column j
    vecs = np.hstack([np.eye(r), Bm])           # each structured column as a vector in u-space
    return X, cl, lo, rng, {"perm": perm, "vecs": vecs}


def one(variant, seed, P):
    t0 = time.time()
    if variant == "param":
        X, cl, lo, rng, info = gen_param(seed, P)
    else:
        X, cl, lo, rng, info = gen_mc(seed, a.class_sep)
    X = np.round(X, 6)
    rp = rng.permutation(len(cl))
    X, cl, lo = X[rp], cl[rp], lo[rp]
    pi = 1 / (1 + np.exp(-np.clip(lo, -700, 700)))
    s, hv, nit = s003_scores(X, cl[:N_TR])
    s8 = np.round(s, 8)
    te = slice(N_TR, None)
    n_te = len(cl) - N_TR
    priv = np.ones(n_te, bool)
    priv[rng.choice(n_te, N_PUB, replace=False)] = False
    c_te, lo_te, pi_te, s_te, s8_te = cl[te], lo[te], pi[te], s[te], s8[te]
    flipm = rng.random(n_te) < a.flip
    y_f = c_te.copy()
    y_f[flipm] = rng.integers(0, 2, flipm.sum())
    row = {"variant": variant, "seed": int(seed), "n_highvar": int(len(hv)), "gmm_n_iter": nit}
    if variant == "param":
        row["block_equals_real13"] = bool(set(hv.tolist()) == set(P["real_hv"]))
        row["n_noise_in_block"] = int(len(set(hv.tolist()) - set(P["S"].tolist())))
        row["block_spans"] = True
    else:
        perm = info["perm"]
        orig = perm[hv]
        struct = orig[orig < 20]
        row["n_noise_in_block"] = int((orig >= 20).sum())
        row["block_spans"] = bool(np.linalg.matrix_rank(info["vecs"][:, struct]) == 12) if len(struct) else False
    for name, sc in (("bayes", lo_te), ("s003", s_te), ("s003_r8", s8_te)):
        row[f"auc_clean_{name}"] = auc_ties(c_te[priv], sc[priv])
        row[f"auc_flip_{name}"] = auc_ties(y_f[priv], sc[priv])
        row[f"eauc_{name}"] = expected_auc(pi_te[priv], sc[priv])
    row["n1n0_private"] = float(c_te[priv].sum() * (1 - c_te[priv]).sum())
    row["n_ties_min_r8"] = int((s8_te == s8_te.min()).sum())
    row["seconds"] = time.time() - t0
    return row


cfg = {"name": "r921_bayes_regret_replicas", "n_param": a.n_param, "n_mc": a.n_mc, "class_sep_mc": a.class_sep,
       "hamming_mc": 4, "flip_for_noisy_auc": a.flip, "procedure": "S003 (highvar block, GMM K=2 full, n_init 5, "
       "reg 1e-6, default tol, smoothed rates)", "independent_of": ["research/lib_gen_replica.py", "research/lib_mix.py"]}
HYP = ("On parametric replicas the paired clean-label regret Bayes - S003 is a few 1e-7 (gen: 3.1e-7 +- 0.75e-7 SE, "
       "mix: 1.7e-7); with 400 replicas the MC SE is ~4e-8 for observed AUC and smaller for the expected AUC, so a "
       "positive regret of that size is real but ~1000x below the label-noise sd of the private AUC (~5e-4). On generic "
       "make_classification replicas the mean regret is larger and heavy-tailed (block misses an informative direction "
       "or contains noise columns). With flip_y = 0.01 the paired regret is swamped by label noise.")
with Run("r921_bayes_regret_replicas", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    ds = data.load(a.data)
    X_all = np.asarray(unlabeled_features(a.data))
    P = fit_param(X_all, ds.y)
    std = X_all.std(0)
    P["real_hv"] = np.where(std > 1.5 * np.median(std))[0].tolist()
    print(f"param fit: structured {len(P['S'])} cols, rank {P['r']}, recon max abs {P['recon_max_abs']:.2e}, "
          f"real block {P['real_hv']}", flush=True)
    jobs = [delayed(one)("param", s, P) for s in range(a.n_param)] + [delayed(one)("mc", s, None) for s in range(a.n_mc)]
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(jobs)
    D = pd.DataFrame(rows)
    for kind_ in ("clean", "flip"):
        D[f"regret_{kind_}"] = D[f"auc_{kind_}_bayes"] - D[f"auc_{kind_}_s003"]
        D[f"regret_{kind_}_r8"] = D[f"auc_{kind_}_bayes"] - D[f"auc_{kind_}_s003_r8"]
    D["regret_eauc"] = D.eauc_bayes - D.eauc_s003
    D["regret_eauc_r8"] = D.eauc_bayes - D.eauc_s003_r8
    D["regret_clean_pairs"] = D.regret_clean * D.n1n0_private
    run.save_table("replicas", D)
    tables = paths.tables_dir(a.data)
    tables.mkdir(parents=True, exist_ok=True)
    D.to_csv(tables / "r921_replicas.csv", index=False)

    def summ(g):
        out = {"n": len(g)}
        for c in ("auc_clean_bayes", "auc_flip_bayes", "eauc_bayes"):
            out[c + "_mean"] = g[c].mean()
        for c in ("regret_clean", "regret_clean_r8", "regret_eauc", "regret_eauc_r8", "regret_flip", "regret_flip_r8",
                  "regret_clean_pairs"):
            v = g[c].to_numpy()
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            out[c + "_mean"] = v.mean()
            out[c + "_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            out[c + "_se"] = se
            out[c + "_t"] = v.mean() / se if se and se > 0 else np.nan
            out[c + "_median"] = np.median(v)
            out[c + "_frac_pos"] = float((v > 0).mean())
        return pd.Series(out)

    groups = {"param": D[D.variant == "param"], "mc_all": D[D.variant == "mc"],
              "mc_spans_no_noise": D[(D.variant == "mc") & D.block_spans & (D.n_noise_in_block == 0)],
              "mc_spans_with_noise": D[(D.variant == "mc") & D.block_spans & (D.n_noise_in_block > 0)],
              "mc_not_spanning": D[(D.variant == "mc") & ~D.block_spans]}
    SM = pd.DataFrame({k: summ(g) for k, g in groups.items() if len(g)}).T.reset_index().rename(columns={"index": "group"})
    run.save_table("summary", SM)
    SM.to_csv(tables / "r921_summary.csv", index=False)
    cols = ["group", "n", "auc_clean_bayes_mean", "regret_clean_mean", "regret_clean_se", "regret_clean_t",
            "regret_eauc_mean", "regret_eauc_se", "regret_eauc_t", "regret_eauc_r8_mean", "regret_flip_mean",
            "regret_flip_se", "regret_clean_pairs_mean"]
    (tables / "r921_summary.md").write_text(SM[cols].to_markdown(index=False, floatfmt=".3g"), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(SM[cols].to_string(), flush=True)
    pr = SM.set_index("group")
    out = {"param_recon_max_abs": P["recon_max_abs"],
           "param_block_equals_real13_rate": float(D[D.variant == "param"].block_equals_real13.mean()) if a.n_param else None,
           "summary": pr[cols[1:]].to_dict(orient="index"), "seconds_per_replica": float(D.seconds.mean())}
    run.finish(extra_summary=out)
    print(out)
