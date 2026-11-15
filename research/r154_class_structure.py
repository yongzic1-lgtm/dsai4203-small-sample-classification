"""R154 - per-class Gaussian structure in the informative subspace and why linear models stall at ~0.88.

Label-free fit (research/lib_gen_replica.fit_parametric: 2-component full-covariance GMM on the 12
informative columns of all 20,000 rows; train labels only orient the components). Reports
  * class means (latent = standardised informative columns, and in unscaled make_classification units
    using the R150 scale estimates a_i), class covariance eigenvalues, generalised eigenvalues of
    Sigma0^-1 Sigma1 (how differently the classes are shaped);
  * population AUCs implied by the fitted Gaussians (Monte Carlo, 400k rows per class):
      Bayes (quadratic) AUC; best linear projection AUC = Phi(sqrt(dmu' (S0+S1)^-1 dmu));
      LDA-direction AUC; mean-only AUC (both classes with the pooled covariance);
      covariance-only AUC (both classes with the pooled mean, own covariances);
  * Bayes error rate and the expected number of test rows in the overlap region.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen --high-mem -- research/r154_class_structure.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--mc-rows", type=int, default=400_000)
a = p.parse_args()

X = unlabeled_features(a.data)
y = data.load(a.data).y
tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)


def qlogodds(Z, m0, S0, m1, S1):
    return G._mvn_logpdf(Z - m1, S1) - G._mvn_logpdf(Z - m0, S0)


def mc_auc(m0, S0, m1, S1, score, n, rng):
    Z0 = m0 + rng.standard_normal((n, len(m0))) @ np.linalg.cholesky(S0).T
    Z1 = m1 + rng.standard_normal((n, len(m1))) @ np.linalg.cholesky(S1).T
    s = np.r_[score(Z0), score(Z1)]
    lab = np.r_[np.zeros(n), np.ones(n)]
    return G.auc_fast(lab, s), Z0, Z1


cfg = {"name": "r154_class_structure", "mc_rows_per_class": a.mc_rows}
HYP = ("One Gaussian per class in the 12-dim informative space; the classes differ mainly in covariance: the best "
       "linear projection reaches only ~0.88-0.92 AUC (explaining LDA/logreg/L1 in R003) while the quadratic Bayes "
       "rule reaches ~0.9999; a covariance-only version of the fitted model is still near-perfectly separable.")
with Run("r154_class_structure", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    st = G.identify_structure(X)
    ps = G.fit_parametric(X, y, struct=st)
    I = list(ps.cols_I)
    m0, m1 = ps.means
    S0, S1 = ps.covs
    a_I = st["a_informative"]                      # scale estimates, same order as I (sorted)
    sd_I = ps.sd_I
    # latent L = (x - mu)/sd ; unscaled u = (x - b)/a -> du = dL * sd / a
    du = (m1 - m0) * sd_I / a_I
    means = pd.DataFrame({"informative": I, "mean_class0_latent": m0, "mean_class1_latent": m1, "diff_latent": m1 - m0,
                          "diff_unscaled_u": du, "active": [c in st["active"] for c in I],
                          "var_class0_latent": np.diag(S0), "var_class1_latent": np.diag(S1),
                          "var_class0_unscaled": np.diag(S0) * (sd_I / a_I) ** 2, "var_class1_unscaled": np.diag(S1) * (sd_I / a_I) ** 2})
    run.save_table("class_means", means)
    means.to_csv(tables / "r154_class_means.csv", index=False)
    ev0, ev1 = np.sort(np.linalg.eigvalsh(S0))[::-1], np.sort(np.linalg.eigvalsh(S1))[::-1]
    gev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(S0, S1))))[::-1]
    eig = pd.DataFrame({"k": np.arange(1, len(I) + 1), "eig_cov_class0": ev0, "eig_cov_class1": ev1, "eig_S0inv_S1": gev})
    run.save_table("class_cov_eigs", eig)
    eig.to_csv(tables / "r154_class_cov_eigs.csv", index=False)

    rng = np.random.default_rng(154)
    n = a.mc_rows
    dmu = m1 - m0
    Sp = 0.5 * (S0 + S1)
    w_best = np.linalg.solve(S0 + S1, dmu)
    w_lda = np.linalg.solve(Sp, dmu)
    auc_lin_formula = float(stats.norm.cdf(np.sqrt(dmu @ np.linalg.solve(S0 + S1, dmu))))
    res = {}
    res["bayes_quadratic"], Z0, Z1 = mc_auc(m0, S0, m1, S1, lambda Z: qlogodds(Z, m0, S0, m1, S1), n, rng)
    lo = np.r_[qlogodds(Z0, m0, S0, m1, S1), qlogodds(Z1, m0, S0, m1, S1)]
    lab = np.r_[np.zeros(n), np.ones(n)]
    bayes_err = float(np.mean((lo > 0) != (lab == 1)))
    post = 1 / (1 + np.exp(-np.clip(lo, -700, 700)))
    res["best_linear_mc"] = G.auc_fast(lab, np.r_[Z0 @ w_best, Z1 @ w_best])
    res["lda_direction_mc"] = G.auc_fast(lab, np.r_[Z0 @ w_lda, Z1 @ w_lda])
    mid = 0.5 * (m0 + m1)
    res["mean_only_pooled_cov"], *_ = mc_auc(m0, Sp, m1, Sp, lambda Z: Z @ np.linalg.solve(Sp, dmu), n, rng)
    res["cov_only_same_mean"], *_ = mc_auc(mid, S0, mid, S1, lambda Z: qlogodds(Z, mid, S0, mid, S1), n, rng)
    aucs = pd.DataFrame([{"quantity": k, "auc": v} for k, v in res.items()] +
                        [{"quantity": "best_linear_formula", "auc": auc_lin_formula}])
    run.save_table("population_aucs", aucs)
    aucs.to_csv(tables / "r154_population_aucs.csv", index=False)
    (tables / "r154_population_aucs.md").write_text(aucs.to_markdown(index=False, floatfmt=".6f"), encoding="utf-8")
    out = {"informative": I, "active": st["active"], "diff_unscaled_active": [round(float(v), 4) for v, c in zip(du, I) if c in st["active"]],
           "diff_unscaled_inactive_absmax": float(max(abs(v) for v, c in zip(du, I) if c not in st["active"])),
           "mahalanobis_pooled": float(np.sqrt(dmu @ np.linalg.solve(Sp, dmu))),
           "gen_eig_S0inv_S1_range": [float(gev.min()), float(gev.max())],
           "log_det_ratio_S1_S0": float(np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1]),
           **{f"auc_{k}": float(v) for k, v in res.items()}, "auc_best_linear_formula": auc_lin_formula,
           "bayes_error_rate": bayes_err, "frac_post_0.1_0.9": float(((post > 0.1) & (post < 0.9)).mean()),
           "expected_test_rows_0.1_0.9": float(((post > 0.1) & (post < 0.9)).mean() * 19750),
           "gmm_weights": [float(w) for w in ps.weights]}
    run.finish(extra_summary=out)
    for k, v in out.items():
        print(f"{k}: {v}")
    print(means.to_string(index=False))
    print(eig.to_string(index=False))
    print(aucs.to_string(index=False))
