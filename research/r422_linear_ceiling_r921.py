"""R422 (research round 2, agent "r2_noise") - R154's linear-projection ceiling (population AUC 0.947), Bayes error
(0.33 %) and boundary-row count (~186 test rows with 0.1 < p < 0.9), recomputed with R921's parametric generator.

R154 used lib_gen_replica.fit_parametric (K=2 full GMM on the 12 standardised informative columns, reg 1e-6,
tol 1e-6, n_init 5). R921 (verify_gen) wrote an independent generator: a converged K=2 full GMM (n_init 10,
tol 1e-10, reg 1e-10) in the 12-dim PCA basis of the 20 linearly structured columns, all structured columns rebuilt
exactly, noise columns iid. Both bases span the same 12-dim space and AUCs of linear scores are invariant to
invertible linear maps, so the two fits must agree up to the fitting details. r921 cannot be imported (it runs on
import), so fit_param / gen_param / mvn_logpdf are copied verbatim; R154's fit is recomputed alongside.

  1. Population quantities of both fits (closed form + Monte Carlo, 400k rows per class, same seed as R154):
     best linear projection AUC Phi(sqrt(dmu' (S0+S1)^-1 dmu)), LDA direction, mean-only, covariance-only and
     quadratic Bayes AUC, Bayes error, expected fraction / count of test rows with p in (0.1, 0.9), (0.05, 0.95),
     (0.01, 0.99).
  2. Boundary rows on the REAL test rows under several definitions that circulate in the reports: converged
     posterior (R922: 189), S003's single-fit mixture posterior (r321: 203), S003's submitted fold-mean score
     (r932: 204).
  3. Parametric bootstrap with R921's generator (--n-rep replicas of 20,000 rows, 6-decimal rounding, random
     250/19,750 split): per replica the TRUE boundary count on the test rows, the count under a refitted converged
     posterior (as computed on the real data), the refitted linear ceiling / Bayes error (sampling sd of the
     estimates), and the test AUC of two linear models trained on the 250 train rows (shrinkage LDA and L2 logistic
     regression on the high-variance block) to show how far finite-sample linear models fall below the ceiling.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2noise --high-mem -- research/r422_linear_ceiling_r921.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import TransductiveGMM, unlabeled_features  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--n-rep", type=int, default=200)
p.add_argument("--mc-rows", type=int, default=400_000)
p.add_argument("--quick", action="store_true")
a = p.parse_args()
if a.quick:
    a.n_rep, a.mc_rows = 4, 20_000
N_TR, N_PUB, NK = 250, 395, 10000
N_TEST = 2 * NK - N_TR
BANDS = [(0.1, 0.9), (0.05, 0.95), (0.01, 0.99)]


# ----------------------------------------------------------------------------- R921 code (copied verbatim)
def mvn_logpdf(X, mu, cov):
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (sol ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * np.log(2 * np.pi)


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


# ----------------------------------------------------------------------------- new code
def population(m0, S0, m1, S1, n, seed):
    """Population AUCs / Bayes error / boundary fractions of two equal-weight Gaussians (R154's quantities)."""
    rng = np.random.default_rng(seed)
    dmu = m1 - m0
    Sp = 0.5 * (S0 + S1)
    w_best = np.linalg.solve(S0 + S1, dmu)
    w_lda = np.linalg.solve(Sp, dmu)
    Z0 = m0 + rng.standard_normal((n, len(m0))) @ np.linalg.cholesky(S0).T
    Z1 = m1 + rng.standard_normal((n, len(m1))) @ np.linalg.cholesky(S1).T
    lab = np.r_[np.zeros(n), np.ones(n)]
    Zc = np.vstack([Z0, Z1])
    lo = mvn_logpdf(Zc, m1, S1) - mvn_logpdf(Zc, m0, S0)
    post = 1 / (1 + np.exp(-np.clip(lo, -700, 700)))
    mid = 0.5 * (m0 + m1)
    Zm0 = mid + rng.standard_normal((n, len(m0))) @ np.linalg.cholesky(S0).T
    Zm1 = mid + rng.standard_normal((n, len(m1))) @ np.linalg.cholesky(S1).T
    Zmc = np.vstack([Zm0, Zm1])
    lo_cov = mvn_logpdf(Zmc, mid, S1) - mvn_logpdf(Zmc, mid, S0)
    Zp0 = m0 + rng.standard_normal((n, len(m0))) @ np.linalg.cholesky(Sp).T
    Zp1 = m1 + rng.standard_normal((n, len(m1))) @ np.linalg.cholesky(Sp).T
    out = {"auc_best_linear_formula": float(stats.norm.cdf(np.sqrt(dmu @ np.linalg.solve(S0 + S1, dmu)))),
           "auc_lda_direction_formula": float(stats.norm.cdf((w_lda @ dmu) / np.sqrt(w_lda @ (S0 + S1) @ w_lda))),
           "auc_best_linear_mc": G.auc_fast(lab, Zc @ w_best), "auc_lda_direction_mc": G.auc_fast(lab, Zc @ w_lda),
           "auc_bayes_quadratic_mc": G.auc_fast(lab, lo),
           "auc_mean_only_pooled_cov_mc": G.auc_fast(lab, np.vstack([Zp0, Zp1]) @ w_lda),
           "auc_cov_only_same_mean_mc": G.auc_fast(lab, lo_cov),
           "mahalanobis_pooled": float(np.sqrt(dmu @ np.linalg.solve(Sp, dmu))),
           "bayes_error_rate": float(np.mean((lo > 0) != (lab == 1)))}
    for lo_b, hi_b in BANDS:
        fr = float(((post > lo_b) & (post < hi_b)).mean())
        out[f"frac_post_{lo_b:g}_{hi_b:g}"] = fr
        out[f"expected_test_rows_{lo_b:g}_{hi_b:g}"] = fr * N_TEST
    return out


def refit_posterior(X_all):
    """R922 / R921 converged posterior refitted on a matrix (label-free; orientation fixed later by the truth)."""
    mu, sd = X_all.mean(0), X_all.std(0)
    Z = (X_all - mu) / sd
    Cm = (Z.T @ Z) / len(Z)
    off = np.abs(Cm - np.diag(np.diag(Cm))).max(1)
    S = np.where(off > 0.1)[0]
    ev, V = np.linalg.eigh(Cm[np.ix_(S, S)])
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    r = int((ev > 1e-9 * ev[0]).sum())
    W = Z[:, S] @ V[:, :r]
    gm = GaussianMixture(2, covariance_type="full", n_init=1 if a.quick else 10, tol=1e-10, reg_covar=1e-10,
                         max_iter=20000, random_state=1).fit(W)
    lp = gm._estimate_weighted_log_prob(W)
    return lp[:, 1] - lp[:, 0], gm.means_, gm.covariances_, len(S), r


def one_rep(seed, P):
    t0 = time.time()
    X, cl, lo, rng, _ = gen_param(seed, P)
    X = np.round(X, 6)
    rp = rng.permutation(len(cl))
    X, cl, lo = X[rp], cl[rp], lo[rp]
    te = slice(N_TR, None)
    pi = 1 / (1 + np.exp(-np.clip(lo, -700, 700)))
    row = {"seed": int(seed)}
    lo_f, mf, cf, n_struct, rank = refit_posterior(X)
    if np.corrcoef(lo_f, lo)[0, 1] < 0:                 # orient the refitted components by the truth
        lo_f, mf, cf = -lo_f, mf[::-1], cf[::-1]
    pf = 1 / (1 + np.exp(-np.clip(lo_f, -700, 700)))
    for lo_b, hi_b in BANDS:
        row[f"true_test_{lo_b:g}_{hi_b:g}"] = int(((pi[te] > lo_b) & (pi[te] < hi_b)).sum())
        row[f"refit_test_{lo_b:g}_{hi_b:g}"] = int(((pf[te] > lo_b) & (pf[te] < hi_b)).sum())
    row["true_bayes_errors_test"] = int(((lo[te] > 0) != (cl[te] == 1)).sum())
    row["expected_bayes_error_test_true"] = float(np.minimum(pi[te], 1 - pi[te]).mean())
    row["expected_bayes_error_test_refit"] = float(np.minimum(pf[te], 1 - pf[te]).mean())
    dmu = mf[1] - mf[0]
    row["refit_best_linear_ceiling"] = float(stats.norm.cdf(np.sqrt(dmu @ np.linalg.solve(cf[0] + cf[1], dmu))))
    row["refit_n_structured"], row["refit_rank"] = n_struct, rank
    # finite-sample linear models on the 250 train rows (high-variance block, train+test standardisation as S003)
    hv = G.highvar_columns(X)
    Zb = StandardScaler().fit_transform(X[:, hv])
    ytr, yte = cl[:N_TR], cl[N_TR:]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Zb[:N_TR], ytr)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(Zb[:N_TR], ytr)
    row["n_highvar"] = int(len(hv))
    row["test_auc_lda_shrink_250"] = G.auc_fast(yte, lda.decision_function(Zb[N_TR:]))
    row["test_auc_logreg_l2_250"] = G.auc_fast(yte, lr.decision_function(Zb[N_TR:]))
    # population-optimal linear direction applied to this replica's test rows (finite-test-set ceiling)
    Wr = ((X - P["mu"]) / P["sd"])[:, P["S"]] @ P["V"]
    w_best = np.linalg.solve(P["covs"][0] + P["covs"][1], P["means"][1] - P["means"][0])
    row["test_auc_true_best_linear"] = G.auc_fast(yte, Wr[N_TR:] @ w_best)
    row["test_auc_bayes"] = G.auc_fast(yte, lo[te])
    row["seconds"] = time.time() - t0
    return row


tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r422_linear_ceiling_r921", "generator": "R921 fit_param / gen_param (copied verbatim)",
       "compared_with": "R154 = lib_gen_replica.fit_parametric (run 20261022-065605_r154_class_structure)",
       "mc_rows_per_class": a.mc_rows, "mc_seed": 154, "n_rep": a.n_rep, "bands": BANDS, "quick": a.quick}
HYP = ("Written before running. R921's converged fit and R154's fit describe the same 12-dim Gaussian pair, so the "
       "population best-linear AUC is 0.947 +- 0.002 under both (R154: formula 0.9468, MC 0.9466), the quadratic Bayes "
       "AUC 0.99995, the Bayes error 0.33 % +- 0.02 %, and ~186 +- 5 test rows are expected with 0.1 < p < 0.9. On the "
       "real test rows the converged posterior gives 189 (R922), the S003 mixture ~203 (r321) and S003's fold-mean "
       "score ~204 (r932): the definitions differ by ~10 %, not the model. Across R921 replicas the true count has sd "
       "~13 (binomial), so 189 is typical; the refitted posterior's count is within a few rows of the true one. The "
       "refitted ceiling has sampling sd < 0.005. Linear models trained on 250 rows reach ~0.85-0.92 test AUC.")

with Run("r422_linear_ceiling_r921", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t_start = time.time()
    ds = data.load(a.data)
    y_tr = np.asarray(ds.y).astype(int)
    X_all = np.asarray(unlabeled_features(a.data), dtype=float)
    P = fit_param(X_all, y_tr)
    print(f"R921 fit: structured {len(P['S'])} cols, rank {P['r']}, recon {P['recon_max_abs']:.2e}", flush=True)
    st = G.identify_structure(X_all)
    ps = G.fit_parametric(X_all, y_tr, struct=st)
    pop = {"R921_converged_pca": population(P["means"][0], P["covs"][0], P["means"][1], P["covs"][1], a.mc_rows, 154),
           "R154_lib_gen_replica": population(ps.means[0], ps.covs[0], ps.means[1], ps.covs[1], a.mc_rows, 154)}
    POP = pd.DataFrame(pop).reset_index().rename(columns={"index": "quantity"})
    POP["R154_archived"] = POP.quantity.map({
        "auc_best_linear_formula": 0.9468428993444679, "auc_best_linear_mc": 0.94662692549375,
        "auc_lda_direction_mc": 0.94662692549375, "auc_bayes_quadratic_mc": 0.999949316725,
        "auc_mean_only_pooled_cov_mc": 0.947035300975, "auc_cov_only_same_mean_mc": 0.9998030571,
        "mahalanobis_pooled": 2.283932004869551, "bayes_error_rate": 0.003265, "frac_post_0.1_0.9": 0.0094225,
        "expected_test_rows_0.1_0.9": 186.094375})
    run.save_table("population", POP)
    POP.to_csv(tables / "r422_population.csv", index=False)
    (tables / "r422_population.md").write_text(POP.to_markdown(index=False, floatfmt=".6g"), encoding="utf-8")
    print(POP.to_string(index=False), flush=True)

    # ---- real test rows: boundary counts under the circulating definitions
    n_tr = len(y_tr)
    lo_real, *_ = refit_posterior(X_all)
    if lo_real[:n_tr][y_tr == 1].mean() < lo_real[:n_tr][y_tr == 0].mean():
        lo_real = -lo_real
    pi_te = 1 / (1 + np.exp(-np.clip(lo_real[n_tr:], -700, 700)))
    tg = TransductiveGMM("highvar", 2, source=a.data).fit(ds.X, y_tr)
    post_single = tg.predict_proba(ds.X_test)[:, 1]
    scheme = Scheme.from_dict(yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))["official_cv"])
    if a.quick:
        scheme = Scheme(kind="kfold", n_splits=5, n_repeats=1, seed=2026)
    cvr = cross_validate(TransductiveGMM("highvar", 2, source=a.data), ds.X, y_tr, make_splits(y_tr, scheme), scheme,
                         X_test=ds.X_test, n_jobs=1)
    s003 = np.array([float(f"{v:.8f}") for v in cvr.test_pred])
    real = []
    for name, v, src in (("converged posterior (R922 fit_posterior)", pi_te, "R922: 189 / 554"),
                         ("S003 mixture, single fit on all 250 labels (r321)", post_single, "r321: 203"),
                         ("S003 submitted fold-mean score, %.8f (r932)", s003, "r932: 204")):
        r = {"definition": name, "archived": src}
        for lo_b, hi_b in BANDS:
            r[f"n_{lo_b:g}_{hi_b:g}"] = int(((v > lo_b) & (v < hi_b)).sum())
        real.append(r)
    REAL = pd.DataFrame(real)
    run.save_table("real_boundary_counts", REAL)
    REAL.to_csv(tables / "r422_real_boundary_counts.csv", index=False)
    print(REAL.to_string(index=False), flush=True)

    # ---- parametric bootstrap with R921's generator
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_rep)(s, P) for s in range(a.n_rep))
    D = pd.DataFrame(rows)
    run.save_table("replicas", D)
    D.to_csv(tables / "r422_replicas.csv", index=False)
    summ = D.drop(columns=["seed"]).describe(percentiles=[0.05, 0.5, 0.95]).T.reset_index().rename(
        columns={"index": "quantity"})
    real_conv = REAL.iloc[0]
    summ["real_value"] = summ.quantity.map({f"refit_test_{lo_b:g}_{hi_b:g}": real_conv[f"n_{lo_b:g}_{hi_b:g}"]
                                            for lo_b, hi_b in BANDS})
    for lo_b, hi_b in BANDS:
        col = f"refit_test_{lo_b:g}_{hi_b:g}"
        summ.loc[summ.quantity == col, "real_percentile_in_replicas"] = float(
            (D[col] < real_conv[f"n_{lo_b:g}_{hi_b:g}"]).mean() + 0.5 * (D[col] == real_conv[f"n_{lo_b:g}_{hi_b:g}"]).mean())
    run.save_table("replica_summary", summ)
    summ.to_csv(tables / "r422_replica_summary.csv", index=False)
    (tables / "r422_replica_summary.md").write_text(summ.to_markdown(index=False, floatfmt=".5g"), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summ.to_string(index=False), flush=True)
    d_cnt = D["refit_test_0.1_0.9"] - D["true_test_0.1_0.9"]
    out = {"R921_fit": {"n_structured": int(len(P["S"])), "rank": int(P["r"]), "recon_max_abs": P["recon_max_abs"]},
           "population_R921": pop["R921_converged_pca"], "population_R154_recomputed": pop["R154_lib_gen_replica"],
           "real_boundary_counts": REAL.set_index("definition").drop(columns="archived").to_dict(orient="index"),
           "rep_true_count_0.1_0.9_mean_sd": [float(D["true_test_0.1_0.9"].mean()), float(D["true_test_0.1_0.9"].std())],
           "rep_refit_minus_true_count_mean_sd": [float(d_cnt.mean()), float(d_cnt.std())],
           "rep_refit_ceiling_mean_sd": [float(D.refit_best_linear_ceiling.mean()), float(D.refit_best_linear_ceiling.std())],
           "rep_lda_shrink_250_mean_sd": [float(D.test_auc_lda_shrink_250.mean()), float(D.test_auc_lda_shrink_250.std())],
           "rep_logreg_250_mean_sd": [float(D.test_auc_logreg_l2_250.mean()), float(D.test_auc_logreg_l2_250.std())],
           "seconds": time.time() - t_start}
    run.finish(extra_summary=out)
    for k, v in out.items():
        print(f"{k}: {v}")
