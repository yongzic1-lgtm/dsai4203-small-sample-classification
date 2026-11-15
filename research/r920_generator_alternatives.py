"""R920 (verifier "gen") - can other make_classification settings explain the real data equally well?

Claim under test (research/reports/gen.md, R150/R152): make_classification(n_informative=12, n_redundant=8, n_repeated=0,
n_clusters_per_class=1, hypercube=True, class_sep ~ 0.71, shuffle) + per-column affine map; informative
{6,15,30,46,52,61,64,80,100,112,142,144}, redundant {8,17,26,49,57,67,77,98}. Own code; labels are used only where stated.

  A  rank / exactness: eigenvalues of the structured block vs the 6-decimal rounding floor; residual sd of the 8
     dependencies vs the sd predicted by rounding alone; how many 12-subsets of the 20 columns are full rank (i.e. how
     little "exact linear combination" alone says about WHICH columns are informative).
  B  informative-basis identification without labels: Wishart profile likelihood of each 12-subset (A_k rows iid
     U(-1,1)^12 => within-class Gram ~ Wishart(12, I/3) up to the unknown per-column scales, which are profiled out):
     crit = 6 (log|S0| + log|S1|) - 12 sum_i log((S0_ii + S1_ii)/2)  [nats], all 125,970 subsets; margin to the
     runner-up; row-bootstrap stability; recovery rate on make_classification replicas with known truth.
  C  n_clusters_per_class = 2 vs 1: LR = 2 (loglik_K4 - loglik_K2) of converged full GMMs in the 12-dim latent space,
     real data vs replicas under H0 (parametric fit, own make_classification with 1 cluster/class) and H1 (own
     make_classification with 2 clusters/class, class_sep 0.71).
  D  hypercube True vs False: the 8 zero-difference z-values (T1 = sum z^2) and the spread of the 4 active standardised
     differences (T2 = max/min), simulated under both centroid processes of sklearn (hypercube=False multiplies the
     vertices by U(0,1) per cluster and per coordinate) with ABC matching of the active effect size.
  E  class_sep: exact likelihood from the 8 scale-free ratios w_ki = dx_i^2 / var_k(x_i) = 4 cs^2 / V_ki on the 4 active
     coordinates, V_ki = sum of 12 U(-1,1)^2 (independent across k, i).
  F  noise columns: class-mean and log-variance-ratio z for the 130 unstructured columns (should be N(0,1)).
  G  row order (shuffle) and class balance (weights=None => exactly 10,000 / 10,000 clean rows).
  (flip_y does not change X at all, so it is identifiable only from labels: see R922.)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vgen --high-mem -- research/r920_generator_alternatives.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import itertools
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
p.add_argument("--n-rep-lr", type=int, default=12, help="replicas per hypothesis for the cluster LR test")
p.add_argument("--n-boot", type=int, default=20)
p.add_argument("--n-calib", type=int, default=24, help="mc replicas for basis-recovery calibration")
p.add_argument("--n-sim", type=int, default=4_000_000, help="Monte Carlo draws for D and E")
p.add_argument("--quick", action="store_true")
a = p.parse_args()
if a.quick:
    a.n_rep_lr, a.n_boot, a.n_calib, a.n_sim = 1, 1, 1, 100_000
GEN_INF = (6, 15, 30, 46, 52, 61, 64, 80, 100, 112, 142, 144)
GEN_RED = (8, 17, 26, 49, 57, 67, 77, 98)
NK = 10000


def wmom(X, w):
    w = w / w.sum()
    m = w @ X
    Xc = X - m
    return m, (Xc * w[:, None]).T @ Xc


def latent_basis(X):
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    C = (Z.T @ Z) / len(Z)
    off = np.abs(C - np.diag(np.diag(C))).max(1)
    r2 = 1.0 - 1.0 / np.diag(np.linalg.inv(C + 1e-9 * np.eye(len(C))))
    S = np.where((off > 0.05) | (r2 > 0.5))[0]
    ev, V = np.linalg.eigh(C[np.ix_(S, S)])
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    r = int((ev > 1e-9 * ev[0]).sum())
    return S, ev, Z[:, S] @ V[:, :r], off, C


def gmm(W, K, n_init=5, seed=0, tol=1e-8, max_iter=5000):
    return GaussianMixture(K, covariance_type="full", n_init=n_init, tol=tol, reg_covar=1e-10, max_iter=max_iter,
                           random_state=seed).fit(W)


def basis_search(S0, S1, cols, top=10, chunk=20000):
    """Wishart profile log-likelihood (nats, up to a constant) of every 12-subset of `cols` as the informative basis."""
    n = 12
    combos = list(itertools.combinations(range(len(cols)), 12))
    res = np.empty(len(combos))
    for st in range(0, len(combos), chunk):
        idx = np.array(combos[st:st + chunk])
        A0 = S0[idx[:, :, None], idx[:, None, :]]
        A1 = S1[idx[:, :, None], idx[:, None, :]]
        s0, l0 = np.linalg.slogdet(A0)
        s1, l1 = np.linalg.slogdet(A1)
        dg = np.log((np.diagonal(A0, axis1=1, axis2=2) + np.diagonal(A1, axis1=1, axis2=2)) / 2).sum(1)
        v = n / 2 * (l0 + l1) - n * dg
        res[st:st + chunk] = np.where((s0 > 0) & (s1 > 0), v, -np.inf)
    o = np.argsort(-res)
    best = [tuple(int(cols[j]) for j in combos[i]) for i in o[:top]]
    return best, res[o[:top]], res


def class_moments(X, post, cols):
    m1, C1 = wmom(X[:, cols], post)
    m0, C0 = wmom(X[:, cols], 1 - post)
    return m0, C0, m1, C1, (1 - post).sum(), post.sum()


# ----------------------------------------------------------------------------- generators (own code)
def mc_latent(rng, cs=0.71, n_clusters_per_class=1, hamming=4, r=12):
    """Informative coordinates u of make_classification (hypercube=True); returns u, class, cluster params."""
    nc = 2 * n_clusters_per_class
    if n_clusters_per_class == 1 and hamming is not None:
        v0 = rng.choice([-1.0, 1.0], r)
        v1 = v0.copy()
        v1[rng.choice(r, hamming, replace=False)] *= -1
        V = np.vstack([v0, v1])
    else:
        codes = rng.choice(2 ** r, nc, replace=False)
        V = np.array([[1.0 if (c >> b) & 1 else -1.0 for b in range(r)] for c in codes])
    cent = cs * V
    sizes = [2 * NK // nc] * nc
    A = [2 * rng.random((r, r)) - 1 for _ in range(nc)]
    U = np.vstack([rng.standard_normal((sizes[k], r)) @ A[k] + cent[k] for k in range(nc)])
    cls = np.concatenate([np.full(sizes[k], k % 2) for k in range(nc)])
    return U, cls, cent, A


def mc_full(rng, cs=0.71, hamming=4):
    U, cls, cent, A = mc_latent(rng, cs, 1, hamming)
    Bm = 2 * rng.random((12, 8)) - 1
    X = np.hstack([U, U @ Bm, rng.standard_normal((2 * NK, 130))])
    perm = rng.permutation(150)
    X = np.round(X[:, perm] * rng.uniform(0.5, 3.0, 150) + rng.uniform(-5, 5, 150), 6)
    inv = np.argsort(perm)              # inv[orig] = new position
    return X, cls, {"informative": sorted(inv[:12].tolist()), "redundant": sorted(inv[12:20].tolist()),
                    "active": sorted(inv[np.flatnonzero(cent[0] != cent[1])].tolist())}


# ----------------------------------------------------------------------------- C: cluster LR worker
def lr_worker(kind, seed, par):
    rng = np.random.default_rng([920, 3, {"H0_param": 1, "H0_mc": 2, "H1_mc_2pc": 3}[kind], int(seed)])
    if kind == "H0_param":
        W = np.vstack([rng.multivariate_normal(par["means"][k], par["covs"][k], size=NK, method="cholesky") for k in (0, 1)])
    elif kind == "H0_mc":
        W = mc_latent(rng, 0.71, 1, 4)[0]
    else:
        W = mc_latent(rng, 0.71, 2, None)[0]
    out = {"kind": kind, "seed": int(seed)}
    ll = {}
    for K in (1, 2, 3, 4):
        g = gmm(W, K, n_init=1 if K == 1 else 4, seed=seed, tol=1e-6, max_iter=1500)
        ll[K] = float(g.score(W) * len(W))
        out[f"bic_K{K}"] = float(g.bic(W))
    out["LR_K3_vs_K2"] = 2 * (ll[3] - ll[2])
    out["LR_K4_vs_K2"] = 2 * (ll[4] - ll[2])
    out["LR_K2_vs_K1"] = 2 * (ll[2] - ll[1])
    return out


def v_density(nsim, seed=92002):
    """Histogram density of log V, V = sum of 12 U(-1,1)^2 (diagonal of A^T A for one column)."""
    rs = np.random.default_rng(seed)
    Vs = np.concatenate([(rs.uniform(-1, 1, (min(nsim, 1_000_000), 12)) ** 2).sum(1)
                         for _ in range(max(1, nsim // 1_000_000))])
    lv = np.log(Vs)
    hb = np.linspace(lv.min() - 1e-9, lv.max() + 1e-9, 801)
    hd, _ = np.histogram(lv, bins=hb, density=True)
    return Vs, (hb[:-1] + hb[1:]) / 2, hd


CSG = np.round(np.arange(0.30, 1.6001, 0.001), 3)


def cs_loglik(wk, hc, hd):
    """log-likelihood of class_sep on CSG from ratios w = dx^2 / var_k(x) = 4 cs^2 / V (V independent)."""
    def logf_V(v):
        x = np.log(v)
        d = np.interp(x, hc, hd, left=0, right=0)
        return np.log(np.maximum(d, 1e-300)) - x
    return np.array([sum(logf_V(4 * c ** 2 / w) + np.log(4 * c ** 2 / w ** 2) for w in wk) for c in CSG])


def calib_worker(seed, hc, hd, cs_true=0.71):
    """Basis recovery, zero-set recovery and class_sep likelihood coverage on a make_classification replica."""
    rng = np.random.default_rng([920, 4, int(seed)])
    X, cls, truth = mc_full(rng, cs=cs_true)
    S, ev, W, _, _ = latent_basis(X)
    if len(S) > 22:
        return {"seed": int(seed), "n_structured": len(S), "rank": int(W.shape[1]), "basis_recovered": False,
                "margin_nats": np.nan, "zero_set_recovered": False}
    g = gmm(W, 2, n_init=3, seed=seed)
    post = g.predict_proba(W)[:, 1]
    if np.corrcoef(post, cls)[0, 1] < 0:
        post = 1 - post
    m0, C0, m1, C1, n0, n1 = class_moments(X, post, S)
    best, vals, _ = basis_search(C0, C1, S, top=2)
    z = (m1 - m0) / np.sqrt(np.diag(C1) / n1 + np.diag(C0) / n0)
    found = sorted(int(S[j]) for j in np.flatnonzero(np.abs(z) < 4))
    true_zero = sorted(set(truth["informative"]) - set(truth["active"]))
    loc = {c: j for j, c in enumerate(S)}
    act = [loc[c] for c in truth["active"] if c in loc]
    dx = (m1 - m0)[act]
    wk = np.r_[dx ** 2 / np.diag(C0)[act], dx ** 2 / np.diag(C1)[act]]
    LL = cs_loglik(wk, hc, hd)
    im = int(np.argmax(LL))
    ci = CSG[LL >= LL[im] - 1.92]
    return {"seed": int(seed), "n_structured": len(S), "rank": int(W.shape[1]),
            "basis_recovered": sorted(best[0]) == truth["informative"], "margin_nats": float(vals[0] - vals[1]),
            "zero_set_recovered": found == true_zero, "true_zero_subset_of_found": set(true_zero) <= set(found),
            "n_zero_found": len(found), "cs_mle": float(CSG[im]), "cs_ci_lo": float(ci.min()), "cs_ci_hi": float(ci.max()),
            "cs_covered": bool(ci.min() <= cs_true <= ci.max())}


# ----------------------------------------------------------------------------- main
tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r920_generator_alternatives", "n_rep_lr": a.n_rep_lr, "n_boot": a.n_boot, "n_calib": a.n_calib,
       "n_sim": a.n_sim, "claim_informative": list(GEN_INF), "claim_redundant": list(GEN_RED), "quick": a.quick}
HYP = ("The structural parameters are pinned down far beyond 'equally good' alternatives: rank 12 / 8 exact dependencies "
       "fix n_informative = 12, n_redundant = 8 deterministically; K = 4 vs K = 2 LR on the real data lies inside the "
       "H0 replica range and orders of magnitude below H1 (2 clusters/class); hypercube=False is disfavoured by a large "
       "factor through the 8 exact zero mean differences; class_sep is ~0.7 with a 95 % interval roughly 0.6-0.85, so "
       "0.6-0.9 are NOT all equally supported (0.9 is clearly worse); flip_y is not identifiable from features at all.")
with Run("r920_generator_alternatives", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t0 = time.time()
    ds = data.load(a.data)
    y_tr = ds.y
    n_tr = len(y_tr)
    X = np.asarray(unlabeled_features(a.data))
    S, ev, W, offmax, Cfull = latent_basis(X)
    out = {"structured": S.tolist(), "n_structured": int(len(S)), "rank": int(W.shape[1]),
           "max_abs_corr_outside_structured": float(offmax[np.setdiff1d(np.arange(150), S)].max()),
           "structured_equals_claim": sorted(S.tolist()) == sorted(GEN_INF + GEN_RED)}
    # ---- A. eigenvalues vs rounding floor, dependency residuals, full-rank subsets
    sdS = X[:, S].std(0)
    floor = (1e-12 / 12) / sdS ** 2                      # variance of 6-decimal rounding error, standardised units
    out["eig_lambda1"] = float(ev[0])
    out["eig_lambda12"] = float(ev[11])
    out["eig_lambda13"] = float(ev[12])
    out["eig_lambda20"] = float(ev[-1])
    out["eig_small8_over_rounding_floor_mean"] = float(ev[12:].mean() / floor.mean())
    pd.DataFrame({"k": np.arange(1, len(ev) + 1), "eigenvalue": ev}).to_csv(tables / "r920_block_eigenvalues.csv", index=False)
    run.save_table("block_eigenvalues", pd.DataFrame({"k": np.arange(1, len(ev) + 1), "eigenvalue": ev}))
    I, Rd = list(GEN_INF), list(GEN_RED)
    XI = np.c_[np.ones(len(X)), X[:, I]]
    dep = []
    for j in Rd:
        beta, *_ = np.linalg.lstsq(XI, X[:, j], rcond=None)
        res = X[:, j] - XI @ beta
        pred = np.sqrt(1e-12 / 12 * (1 + (beta[1:] ** 2).sum()))
        dep.append({"redundant_col": j, "r2": 1 - res.var() / X[:, j].var(), "resid_sd": res.std(), "rounding_sd_pred": pred,
                    "ratio_obs_over_pred": res.std() / pred, "max_abs_resid": np.abs(res).max(),
                    "resid_kurtosis_excess": float(stats.kurtosis(res))})
    DEP = pd.DataFrame(dep)
    run.save_table("dependency_residuals", DEP)
    DEP.to_csv(tables / "r920_dependency_residuals.csv", index=False)
    print(DEP.to_string(), flush=True)
    # ---- label-free posterior (orientation by train labels only for reporting)
    g2 = gmm(W, 2, n_init=2 if a.quick else 10, seed=1, tol=1e-10)
    post = g2.predict_proba(W)[:, 1]
    if post[:n_tr][y_tr == 1].mean() < post[:n_tr][y_tr == 0].mean():
        post = 1 - post
    m0, C0, m1, C1, n0, n1 = class_moments(X, post, S)
    z = (m1 - m0) / np.sqrt(np.diag(C1) / n1 + np.diag(C0) / n0)
    zt = pd.DataFrame({"col": S, "z_mean_diff": z, "abs_z": np.abs(z), "claimed_role": ["informative" if c in GEN_INF else
                       "redundant" for c in S], "var_ratio_1_over_0": np.diag(C1) / np.diag(C0)}).sort_values("abs_z")
    run.save_table("z_mean_diff", zt)
    zt.to_csv(tables / "r920_z_mean_diff.csv", index=False)
    zero = zt.col[zt.abs_z < 4].tolist()
    out["zero_diff_cols"] = sorted(int(c) for c in zero)
    out["zero_diff_all_claimed_informative"] = all(c in GEN_INF for c in zero)
    # full-rank subsets
    best, vals, allv = basis_search(C0, C1, S, top=10)
    out["n_subsets"] = int(len(allv))
    out["n_subsets_full_rank"] = int(np.isfinite(allv).sum())
    Rw = (C0 + C1) / 2
    dsd = np.sqrt(np.diag(Rw))
    out["basis_best"] = sorted(best[0])
    out["basis_best_equals_claim"] = sorted(best[0]) == sorted(GEN_INF)
    out["basis_margin_nats"] = float(vals[0] - vals[1])
    out["basis_runner_up_swap"] = {"out": sorted(set(best[0]) - set(best[1])), "in": sorted(set(best[1]) - set(best[0]))}
    # hypercube-constrained search: the zero-difference columns must be informative
    zpos = [int(np.flatnonzero(S == c)[0]) for c in zero]
    cons = np.array([all(j in cb for j in zpos) for cb in itertools.combinations(range(len(S)), 12)])
    vc = np.where(cons, allv, -np.inf)
    oc = np.argsort(-vc)
    out["n_subsets_hypercube_constrained"] = int(cons.sum())
    out["basis_margin_constrained_nats"] = float(vc[oc[0]] - vc[oc[1]])
    BT = pd.DataFrame({"rank": np.arange(1, 11), "basis": [" ".join(map(str, sorted(b))) for b in best], "crit_nats": vals})
    run.save_table("basis_top10", BT)
    BT.to_csv(tables / "r920_basis_top10.csv", index=False)
    print(BT.to_string(), flush=True)
    # bootstrap stability (row resampling, posterior kept)
    rng = np.random.default_rng(92001)
    boot = []
    for b in range(a.n_boot):
        w = rng.multinomial(len(X), np.full(len(X), 1 / len(X))).astype(float)
        _, B0 = wmom(X[:, S], w * (1 - post))
        _, B1 = wmom(X[:, S], w * post)
        bb, bv, _ = basis_search(B0, B1, S, top=2)
        boot.append({"boot": b, "same_as_claim": sorted(bb[0]) == sorted(GEN_INF), "margin": float(bv[0] - bv[1])})
    BO = pd.DataFrame(boot)
    run.save_table("basis_bootstrap", BO)
    out["basis_bootstrap_same_rate"] = float(BO.same_as_claim.mean())
    out["basis_bootstrap_margin_min"] = float(BO.margin.min())
    print(f"A/B done {time.time() - t0:.0f} s: {out}", flush=True)

    # ---- C. cluster LR (real + replicas), B calibration (replicas) in parallel
    par = {"means": g2.means_, "covs": g2.covariances_}
    Vs, hc, hd = v_density(a.n_sim)
    jobs = [delayed(lr_worker)(k, s, par) for k in ("H0_param", "H0_mc", "H1_mc_2pc") for s in range(a.n_rep_lr)]
    jobs += [delayed(calib_worker)(s, hc, hd) for s in range(a.n_calib)]
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(jobs)
    LRr = pd.DataFrame([r for r in res if "kind" in r])
    CAL = pd.DataFrame([r for r in res if "basis_recovered" in r])
    real = {"kind": "REAL", "seed": -1}
    ll = {}
    for K in (1, 2, 3, 4):
        g = g2 if K == 2 else gmm(W, K, n_init=1 if (K == 1 or a.quick) else 8, seed=1, tol=1e-7, max_iter=3000)
        ll[K] = float(g.score(W) * len(W))
        real[f"bic_K{K}"] = float(g.bic(W))
    real["LR_K3_vs_K2"] = 2 * (ll[3] - ll[2])
    real["LR_K4_vs_K2"] = 2 * (ll[4] - ll[2])
    real["LR_K2_vs_K1"] = 2 * (ll[2] - ll[1])
    LRr = pd.concat([pd.DataFrame([real]), LRr], ignore_index=True)
    run.save_table("cluster_lr", LRr)
    LRr.to_csv(tables / "r920_cluster_lr.csv", index=False)
    lrs = LRr.groupby("kind")[["LR_K3_vs_K2", "LR_K4_vs_K2", "LR_K2_vs_K1"]].agg(["min", "median", "max"])
    print(lrs.to_string(), flush=True)
    run.save_table("basis_calibration", CAL)
    out["cluster_LR_real_K4_vs_K2"] = real["LR_K4_vs_K2"]
    out["cluster_LR_H0_param_K4_vs_K2_range"] = LRr[LRr.kind == "H0_param"].LR_K4_vs_K2.agg(["min", "max"]).tolist()
    out["cluster_LR_H0_mc_K4_vs_K2_range"] = LRr[LRr.kind == "H0_mc"].LR_K4_vs_K2.agg(["min", "max"]).tolist()
    out["cluster_LR_H1_K4_vs_K2_range"] = LRr[LRr.kind == "H1_mc_2pc"].LR_K4_vs_K2.agg(["min", "max"]).tolist()
    out["basis_calibration_recovery_rate"] = float(CAL.basis_recovered.mean())
    out["basis_calibration_zero_set_rate"] = float(CAL.zero_set_recovered.mean())
    out["basis_calibration_margin_median"] = float(CAL.margin_nats.median())
    out["basis_calibration_margin_min"] = float(CAL.margin_nats.min())
    out["zero_set_calibration_true_subset_rate"] = float(CAL.true_zero_subset_of_found.mean())
    out["class_sep_calibration_mle_mean"] = float(CAL.cs_mle.mean())
    out["class_sep_calibration_mle_sd"] = float(CAL.cs_mle.std())
    out["class_sep_calibration_coverage95"] = float(CAL.cs_covered.mean())

    # ---- D/E. hypercube and class_sep from the 4 active coordinates
    inf_order = [c for c in zt.col if c in GEN_INF]
    act = [c for c in zt.col.iloc[::-1] if c in GEN_INF][:4]                # 4 largest |z| among informative
    zer = [c for c in inf_order if c not in act]
    loc = {c: j for j, c in enumerate(S)}
    T1_obs = float(sum(z[loc[c]] ** 2 for c in zer))
    dx = np.array([m1[loc[c]] - m0[loc[c]] for c in act])
    v0 = np.array([C0[loc[c], loc[c]] for c in act])
    v1 = np.array([C1[loc[c], loc[c]] for c in act])
    rpool = np.abs(dx) / np.sqrt((v0 + v1) / 2)
    T2_obs = float(rpool.max() / rpool.min())
    wk = np.r_[dx ** 2 / v0, dx ** 2 / v1]                                   # 8 ratios = 4 cs^2 / V
    out["active_cols"] = act
    out["zero_cols_informative"] = zer
    out["T1_obs_sum_z2_zero"] = T1_obs
    out["T1_chi2_8_cdf"] = float(stats.chi2.cdf(T1_obs, 8))
    out["T2_obs_maxmin_active"] = T2_obs
    out["active_r_pooled"] = rpool.tolist()
    csg = CSG
    LLc = cs_loglik(wk, hc, hd)
    imax = int(np.argmax(LLc))
    ci = csg[LLc >= LLc[imax] - 1.92]
    post_cs = np.exp(LLc - LLc.max())
    post_cs /= post_cs.sum()
    cdf = np.cumsum(post_cs)
    CS = pd.DataFrame({"class_sep": csg, "loglik": LLc, "rel_lik": np.exp(LLc - LLc.max())})
    run.save_table("class_sep_likelihood", CS)
    CS.to_csv(tables / "r920_class_sep_likelihood.csv", index=False)
    named = {f"LR_cs{c:g}_vs_mle": float(np.exp(LLc[np.argmin(np.abs(csg - c))] - LLc[imax]))
             for c in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)}
    out.update({"class_sep_mle": float(csg[imax]), "class_sep_95ci_LR": [float(ci.min()), float(ci.max())],
                "class_sep_post_mean_flat": float((csg * post_cs).sum()),
                "class_sep_post_95": [float(csg[np.searchsorted(cdf, .025)]), float(csg[np.searchsorted(cdf, .975)])],
                **named})
    # hypercube simulation: SE of z in u-units uses V's; ABC on the mean active effect r
    def sim_hyper(hyper, n, chunk):
        r_ = np.random.default_rng([92003, int(hyper), int(chunk)])
        cs = r_.uniform(0.2, 6.0, n)
        Vk = r_.choice(Vs, (2, n, 12))                                     # (class, draw, coord): sum over 12 A rows
        se = np.sqrt(Vk[0] / n0 + Vk[1] / n1)
        vbar = (Vk[0] + Vk[1]) / 2
        if hyper:
            d_agree = np.zeros((n, 8))
            d_act = 2 * cs[:, None] * np.ones((n, 4))
        else:
            s0, s1 = r_.random(n), r_.random(n)
            t = r_.random((n, 12))
            d_agree = (s1 - s0)[:, None] * t[:, :8] * cs[:, None]
            d_act = (s1 + s0)[:, None] * t[:, 8:] * cs[:, None]
        zz = d_agree / se[:, :8] + r_.standard_normal((n, 8))
        rr = np.abs(d_act) / np.sqrt(vbar[:, 8:]) * (1 + 0.02 * r_.standard_normal((n, 4)))
        T1 = (zz ** 2).sum(1)
        T2 = rr.max(1) / rr.min(1)
        mr = rr.mean(1)
        return T1, T2, mr

    nh = max(a.n_sim // 8, 10000)
    hrows = []
    for hyper in (True, False):
        T1s, T2s, mrs = [], [], []
        for ch in range(8):
            t1, t2, mr = sim_hyper(hyper, nh // 8 if not a.quick else nh, ch)
            T1s.append(t1); T2s.append(t2); mrs.append(mr)
        T1s, T2s, mrs = map(np.concatenate, (T1s, T2s, mrs))
        acc = np.abs(mrs / rpool.mean() - 1) < 0.05                          # ABC on the active effect size
        p_t1 = float((T1s[acc] <= T1_obs).mean())
        p_t2 = float((T2s[acc] <= T2_obs).mean())
        p_joint = float(((T1s <= T1_obs) & (T2s <= T2_obs))[acc].mean())
        # local density (likelihood) of the observed (T1, T2) among accepted draws
        near = acc & (np.abs(T1s - T1_obs) < 1.0) & (np.abs(T2s - T2_obs) < 0.05)
        hrows.append({"hypercube": hyper, "n_draws": len(T1s), "n_abc_accepted": int(acc.sum()),
                      "P_T1_le_obs": p_t1, "P_T2_le_obs": p_t2, "P_joint_le_obs": p_joint,
                      "local_density_rel": float(near.sum() / max(len(T1s), 1)),
                      "abc_accept_rate": float(acc.mean())})
    HY = pd.DataFrame(hrows)
    run.save_table("hypercube_test", HY)
    HY.to_csv(tables / "r920_hypercube_test.csv", index=False)
    print(HY.to_string(), flush=True)
    dh = HY.set_index("hypercube")
    nf = float(dh.loc[False, "n_draws"])
    out["hypercube_BF_true_vs_false_local"] = float(dh.loc[True, "local_density_rel"] /
                                                    max(dh.loc[False, "local_density_rel"], 3.0 / nf))
    out["hypercube_BF_is_lower_bound"] = bool(dh.loc[False, "local_density_rel"] == 0)
    out["hypercube_false_P_joint_le_obs"] = float(dh.loc[False, "P_joint_le_obs"])

    # ---- F. noise columns
    N = np.setdiff1d(np.arange(150), S)
    mN1, CN1 = wmom(X[:, N], post)
    mN0, CN0 = wmom(X[:, N], 1 - post)
    vN1, vN0 = np.diag(CN1), np.diag(CN0)
    zN = (mN1 - mN0) / np.sqrt(vN1 / n1 + vN0 / n0)
    zV = np.log(vN1 / vN0) / np.sqrt(2 / n1 + 2 / n0)
    NZ = pd.DataFrame({"col": N, "z_mean": zN, "z_logvar": zV})
    run.save_table("noise_columns", NZ)
    out.update({"noise_z_mean_KS_p": float(stats.kstest(zN, "norm").pvalue),
                "noise_z_logvar_KS_p": float(stats.kstest(zV, "norm").pvalue),
                "noise_max_abs_z": float(max(np.abs(zN).max(), np.abs(zV).max())),
                "noise_bonferroni_p_min": float(min(1.0, 2 * stats.norm.sf(max(np.abs(zN).max(), np.abs(zV).max())) * 260)),
                "noise_cols_std_KS_U0.5_3_p": float(stats.kstest(X[:, N].std(0), "uniform", args=(0.5, 2.5)).pvalue),
                "noise_cols_mean_KS_U-5_5_p": float(stats.kstest(X[:, N].mean(0), "uniform", args=(-5, 10)).pvalue)})
    # ---- G. row order and balance
    ids = np.r_[ds.ids, ds.ids_test]
    te = slice(n_tr, None)
    lab = post > 0.5
    runs = 1 + int((lab[te][1:] != lab[te][:-1]).sum())
    n_a, n_b = int(lab[te].sum()), int((~lab[te]).sum())
    mu_r = 1 + 2 * n_a * n_b / (n_a + n_b)
    sd_r = np.sqrt(2 * n_a * n_b * (2 * n_a * n_b - n_a - n_b) / ((n_a + n_b) ** 2 * (n_a + n_b - 1)))
    out.update({"spearman_id_post_test": float(stats.spearmanr(ids[te], post[te])[0]),
                "spearman_id_post_train": float(stats.spearmanr(ids[:n_tr], post[:n_tr])[0]),
                "runs_test_z_test_rows": float((runs - mu_r) / sd_r),
                "lag1_autocorr_post_test": float(np.corrcoef(post[te][1:], post[te][:-1])[0, 1]),
                "sum_post_all": float(post.sum()), "sd_sum_post": float(np.sqrt((post * (1 - post)).sum())),
                "z_balance_vs_10000": float((post.sum() - 10000) / np.sqrt((post * (1 - post)).sum())),
                "train_y_sum": int(y_tr.sum()), "train_post_sum": float(post[:n_tr].sum())})
    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].semilogy(np.arange(1, len(ev) + 1), ev, "o-"); ax[0].axhline(floor.mean(), color="r", ls="--", label="rounding floor")
        ax[0].set_xlabel("eigenvalue index"); ax[0].set_title("20 structured columns"); ax[0].legend()
        for kind_, g_ in LRr.groupby("kind"):
            ax[1].scatter([kind_] * len(g_), g_.LR_K4_vs_K2, label=kind_)
        ax[1].set_yscale("symlog"); ax[1].set_title("LR K=4 vs K=2"); ax[1].tick_params(axis="x", rotation=20)
        ax[2].plot(csg, np.exp(LLc - LLc.max())); ax[2].set_xlabel("class_sep"); ax[2].set_title("class_sep likelihood")
        fig.tight_layout()
        run.save_figure("r920_generator", fig)
        fig.savefig(figs / "r920_generator.png", dpi=150, bbox_inches="tight")
    except Exception as e:  # noqa: BLE001
        print("figure failed:", e)
    out["seconds"] = time.time() - t0
    run.finish(extra_summary=out)
    for k_, v_ in out.items():
        print(f"{k_}: {v_}")
