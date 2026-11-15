"""R101 - label-free model choice for the mixture on the 20,000 rows: BIC, restarts, regularisation.

Everything here uses only the features of the 20,000 train+test rows (the training labels
are used only to orient components to classes for the S003 comparison).

  1. BIC grid on the block (non-degenerate PCA basis, 12 dims): K = 1..6 x covariance type
     {full, tied, diag, spherical} x reg_covar {1e-6, 1e-4, 1e-3, 1e-2}
  2. joint 142-dim model comparison (same variables for all models: 12 block PCA coordinates
     + the 130 isolated standardised features; the 8 exact redundancies are deterministic):
       A  K=2 full GMM on all 142 dims (what R003 gmm2_all approximates)
       B  K=2 full GMM on the 12 block dims x one full-covariance Gaussian on the 130 (independent)
       C  K=2 full GMM on the 12 block dims x independent N(m_j, s_j) per isolated feature
  3. restart stability: 30 seeds x n_init = 1 for K = 2, 3, 4 (full): log-likelihood spread,
     posterior differences, Kendall tau on the S003 boundary rows; ensemble of the 30 restarts
  4. 'expected AUC' of every K=2 variant's test ranking if test labels were drawn from the
     reference posterior (polished K=2 full block GMM) - a label-free proxy for the private LB;
     for the reference itself this is the model's own Bayes AUC

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rmix --high-mem --no-sync -- research/r101_bic_model_choice.py --n-jobs 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from sklearn.mixture import GaussianMixture

import lib_mix as L
from ssc import data, paths
from ssc.tracking import Run
from ssc.transductive import unlabeled_features

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--n-seeds", type=int, default=30)
a = p.parse_args()
src = a.data

ds = data.load(src)
y = ds.y
X_all = unlabeled_features(src)
n = len(X_all)
logn = np.log(n)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
BLOCK = L.BLOCK
*_, U = L.basis(BLOCK, src)
d = U.shape[1]
s3 = L.s003_pred(src)
b3 = (s3 > 0.1) & (s3 < 0.9)

ref_post, ref_ll, ref_fit = L.polished_posterior(BLOCK, 2, "full", 5, 0, 1e-6, src)
ref = L.oriented(ref_post, y)
ref_test = ref[250:]


def k2_stats(post, name):
    r = L.oriented(post, y)
    t = r[250:]
    return {"variant": name, "kendall_s003_boundary": float(kendalltau(t[b3], s3[b3])[0]),
            "kendall_ref_on_s003_boundary": float(kendalltau(t[b3], ref_test[b3])[0]),
            "max_abs_post_diff_vs_ref": float(np.abs(t - ref_test).max()),
            "n_boundary": int(((t > 0.1) & (t < 0.9)).sum()),
            "expected_auc_under_ref": L.expected_auc(t, ref_test)}


def fit_one(K, cov, reg, n_init=5, seed=0):
    g = GaussianMixture(K, covariance_type=cov, n_init=n_init, random_state=seed, reg_covar=reg,
                        max_iter=5000).fit(U)
    return {"K": K, "cov": cov, "reg": reg, "seed": seed, "n_init": n_init, "bic": float(g.bic(U)),
            "aic": float(g.aic(U)), "mean_loglik": float(g.score(U)), "n_params": int(g._n_parameters()),
            "converged": bool(g.converged_), "n_iter": int(g.n_iter_), "post": g.predict_proba(U)}


hyp = ("BIC on the 20,000 rows picks K = 2 full covariance with negligible regularisation; tied/diag/spherical are "
       "far worse (the classes differ in covariance). The joint 142-dim comparison prefers 'block mixture x independent "
       "noise' over a 142-dim mixture. Restarts of K = 2 all reach the same optimum, so restart ensembles cannot change "
       "the ranking of boundary rows; the expected-AUC proxy differences between K=2 variants are < 1e-4.")
cfg = {"name": "r101_bic_model_choice", "basis": "non-degenerate PCA of the standardised 13-feature block (12 dims)",
       "grid": {"K": list(range(1, 7)), "cov": ["full", "tied", "diag", "spherical"], "reg": [1e-6, 1e-4, 1e-3, 1e-2]},
       "n_seeds_restarts": a.n_seeds}
with Run("r101_bic_model_choice", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    out = {"ref_mean_loglik": ref_ll, "ref_bic": float(-2 * ref_ll * n + L.n_params(2, d, "full") * logn)}

    # 1. grid
    jobs = [(K, cov, reg) for K in range(1, 7) for cov in ("full", "tied", "diag", "spherical")
            for reg in (1e-6, 1e-4, 1e-3, 1e-2)]
    res = Parallel(n_jobs=a.n_jobs)(delayed(fit_one)(K, cov, reg) for K, cov, reg in jobs)
    grid = pd.DataFrame([{k: v for k, v in r.items() if k != "post"} for r in res])
    k2 = []
    for r in res:
        if r["K"] == 2:
            k2.append({**k2_stats(r["post"], f"K2_{r['cov']}_reg{r['reg']:g}"), "bic": r["bic"]})
    grid = grid.sort_values("bic")
    run.save_table("bic_grid", grid)
    grid.to_csv(tables / "r101_bic_grid.csv", index=False)
    best = grid.iloc[0]
    out["bic_best"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    print(grid.head(15).to_string(index=False))
    piv = grid[grid["reg"] == 1e-6].pivot(index="K", columns="cov", values="bic")
    print(piv.to_string())
    piv.reset_index().to_csv(tables / "r101_bic_K_by_cov.csv", index=False)

    # 2. joint 142-dim comparison
    Z, _, _ = L.standardize(X_all)
    iso = np.array([j for j in range(150) if j not in set([6, 8, 15, 17, 26, 30, 46, 49, 52, 57, 61, 64, 67, 77,
                                                             80, 98, 100, 112, 142, 144])])
    Zn = Z[:, iso]
    J = np.c_[U, Zn]
    gA = GaussianMixture(2, covariance_type="full", n_init=2, random_state=0, reg_covar=1e-6, max_iter=2000).fit(J)
    llA = gA.score(J) * n
    kA = gA._n_parameters()
    gB_ll = ref_ll * n
    Cn = np.cov(Zn, rowvar=False, bias=True)
    sign, logdet = np.linalg.slogdet(Cn)
    q = Zn.shape[1]
    llB = gB_ll + (-0.5 * n * (q * np.log(2 * np.pi) + logdet + q))
    kB = L.n_params(2, d, "full") + q + q * (q + 1) // 2
    v = Zn.var(0)
    llC = gB_ll + (-0.5 * n * (np.log(2 * np.pi * v) + 1)).sum()
    kC = L.n_params(2, d, "full") + 2 * q
    joint = pd.DataFrame([
        {"model": "A: K=2 full GMM on all 142 dims", "loglik": llA, "n_params": kA, "bic": -2 * llA + kA * logn},
        {"model": "B: K=2 block GMM x full Gaussian on 130 isolated", "loglik": llB, "n_params": kB, "bic": -2 * llB + kB * logn},
        {"model": "C: K=2 block GMM x independent Gaussians on 130 isolated", "loglik": llC, "n_params": kC, "bic": -2 * llC + kC * logn}])
    run.save_table("joint_142_models", joint)
    joint.to_csv(tables / "r101_joint_142_models.csv", index=False)
    print(joint.to_string(index=False))
    out["joint_142"] = joint.to_dict("records")
    rA = L.oriented(gA.predict_proba(J), y)[250:]
    out["model_A_vs_ref"] = {**L.boundary_stats(rA, s3), "expected_auc_under_ref": L.expected_auc(rA, ref_test)}

    # 3. restarts
    rs_jobs = [(K, s) for K in (2, 3, 4) for s in range(a.n_seeds)]
    rs = Parallel(n_jobs=a.n_jobs)(delayed(fit_one)(K, "full", 1e-6, 1, s) for K, s in rs_jobs)
    rrows = []
    ens = np.zeros(len(ref_test))
    for r in rs:
        row = {k: v for k, v in r.items() if k != "post"}
        if r["K"] == 2:
            st = k2_stats(r["post"], f"seed{r['seed']}")
            row.update({k: v for k, v in st.items() if k != "variant"})
            ens += L.oriented(r["post"], y)[250:]
        rrows.append(row)
    rst = pd.DataFrame(rrows)
    run.save_table("restarts", rst)
    rst.to_csv(tables / "r101_restarts.csv", index=False)
    summ = rst.groupby("K").agg(loglik_min=("mean_loglik", "min"), loglik_max=("mean_loglik", "max"),
                                loglik_sd=("mean_loglik", "std"), n_iter_mean=("n_iter", "mean")).reset_index()
    run.save_table("restarts_summary", summ)
    print(summ.to_string(index=False))
    ens /= a.n_seeds
    k2.append({**k2_stats(np.c_[1 - np.r_[ref[:250], ens], np.r_[ref[:250], ens]], f"ensemble_{a.n_seeds}_restarts"), "bic": np.nan})
    k2.append({**k2_stats(ref_post, "reference_polished"), "bic": out["ref_bic"]})
    s3_full = np.r_[ref[:250], s3]
    k2.append({**k2_stats(np.c_[1 - s3_full, s3_full], "S003"), "bic": np.nan})
    k2 = pd.DataFrame(k2).sort_values("expected_auc_under_ref", ascending=False)
    run.save_table("k2_variants_vs_ref", k2)
    k2.to_csv(tables / "r101_k2_variants.csv", index=False)
    print(k2.to_string(index=False))
    out["expected_auc_ref_self (model Bayes AUC on test)"] = L.expected_auc(ref_test, ref_test)
    out["expected_auc_s003_under_ref"] = L.expected_auc(s3, ref_test)
    out["restart_K2_loglik_range"] = float(summ.loc[summ.K == 2, "loglik_max"].iloc[0] - summ.loc[summ.K == 2, "loglik_min"].iloc[0])
    print({k: v for k, v in out.items() if not isinstance(v, (list, dict))})
    run.finish(extra_summary=out)
