"""R254 - variational Dirichlet-process Gaussian mixture as an alternative transductive model.

sklearn BayesianGaussianMixture (variational inference, stick-breaking Dirichlet-process prior on the
weights, Gauss-Wishart prior on the components, truncation T) fitted on the 20,000 unlabeled
train+test rows of the label-free high-variance block. Labels only map components to classes
(same Beta smoothing as ssc.transductive.TransductiveGMM).

Part A, analysis run r254_dpgmm_evidence (label-free evidence; CV AUC is saturated at 1.0):
  * effective number of components vs the concentration gamma (0.001 ... 100) and truncation T,
    ELBO, plug-in log-likelihood / BIC of the active components in the PCA(12) basis of the block
    (the 13-column block has rank 12, so densities in the raw block are degenerate), next to
    finite GMMs K = 1..6;
  * per-component train-label rates (all 250 labels);
  * full-train test scores vs the S003 reference (recomputed TransductiveGMM K=2 block):
    boundary-row agreement;
  * replica simulation (parametric bootstrap): 20,000 rows sampled from (G1) the fitted GMM K=2 and
    (G2) the fitted DP mixture, labels from the generator, first 250 rows labelled; GMM K=2 and the
    DP mixture are refitted on the simulated features and scored on the 19,750 simulated test rows,
    next to the supervised Bayesian QDA of R255 (250 simulated labels only) and a GMM+BQDA average.
Part B, experiment runs (official CV, fold_mean test predictions): dpgmm_hv, dpgmm_pca12
  (gamma = 1, T = 10, pre-registered).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm --no-sync -- research/r254_dp_gmm.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from ssc import paths
from ssc.tracking import Run
from ssc.transductive import _basis, _mixture, resolve_columns, unlabeled_features
from ssc.validation import cross_validate

import lib_fm
from lib_fm import TransductiveBGMM, mixture_bic, plugin_mixture_loglik

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--skip-analysis", action="store_true")
p.add_argument("--skip-cv", action="store_true")
p.add_argument("--replicas", type=int, default=3)
a = p.parse_args()
src = a.data
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)

ds, scheme, splits = lib_fm.official_setup(src)
X_all = unlabeled_features(src)
hv = resolve_columns("highvar", X_all)
key = tuple(int(c) for c in hv)
n_tr = len(ds.y)
print(f"block={list(key)} cpus={os.cpu_count()} n_jobs={n_jobs}", flush=True)

t0 = time.time()
ref_cv = lib_fm.r003_reference("r003_gmm2_hv", ds, splits, scheme, src)
s003 = ref_cv.test_pred
print(f"S003 reference recomputed in {time.time() - t0:.0f}s: n_boundary={int(((s003 > .1) & (s003 < .9)).sum())}, "
      f"mean={s003.mean():.6f}", flush=True)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
GAMMAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def bgmm_cfgs():
    out = [("pca12", 12, 10, g, "dirichlet_process") for g in GAMMAS]
    out += [("pca12", 12, 20, 1.0, "dirichlet_process"), ("pca12", 12, 10, 0.01, "dirichlet_distribution"),
            ("hv", None, 10, 1.0, "dirichlet_process"), ("hv", None, 10, 0.01, "dirichlet_process")]
    return out


def fit_bgmm_cfg(basis, pca, T, gamma, prior_type):
    *_, Z = _basis(src, key, pca)
    t = time.time()
    m = BayesianGaussianMixture(n_components=T, covariance_type="full", weight_concentration_prior_type=prior_type,
                                weight_concentration_prior=gamma, n_init=3, max_iter=3000, tol=1e-4,
                                init_params="kmeans", random_state=0).fit(Z)
    return m, time.time() - t


def fit_gmm_k(pca, k):
    *_, Z = _basis(src, key, pca)
    return GaussianMixture(k, covariance_type="full", n_init=5, random_state=0, reg_covar=1e-6, max_iter=1000).fit(Z)


def label_rates(R, y, alpha=1.0):
    p0 = y.mean()
    return (R.T @ y + alpha * p0) / (R.sum(0) + alpha)


if not a.skip_analysis:
    cfg = {"name": "r254_dpgmm_evidence", "kind": "analysis", "bgmm_configs": [list(c) for c in bgmm_cfgs()],
           "inputs": "train+test features (label-free fits); 250 train labels only to map components",
           "code_sha": lib_fm.code_hash(__file__)}
    with Run("r254_dpgmm_evidence", cfg, kind="analysis", source=src, author=a.author,
             hypothesis="With a DP prior the variational mixture keeps about 2 effective components for small "
                        "gamma (one per class) and splits a class into more components only for large gamma; "
                        "its plug-in BIC is not better than the finite K=2 GMM, and its test ranking agrees with "
                        "S003 on > 90% of the ~200 boundary rows (sides of 0.5). On replica data with known labels "
                        "the transductive GMM K=2 beats the supervised Bayesian QDA (250 labels) on the GMM's "
                        "boundary rows, and averaging the two does not beat the GMM.") as run:
        *_, Z12 = _basis(src, key, 12)
        Ztr12 = Z12[:n_tr]
        # finite GMMs, K = 1..6 in the PCA(12) basis
        gm = Parallel(n_jobs=n_jobs)(delayed(fit_gmm_k)(12, k) for k in range(1, 7))
        ev = []
        for k, g in zip(range(1, 7), gm):
            ll = float(g.score(Z12) * len(Z12))
            ev.append({"model": f"GMM K={k}", "basis": "pca12", "T": k, "gamma": None, "prior": "finite-ML",
                       "elbo": None, "n_eff_1pct": k, "n_active": k, "loglik_plugin": ll, "bic_plugin": float(g.bic(Z12)),
                       "converged": bool(g.converged_), "n_iter": int(g.n_iter_), "weights": np.sort(g.weights_)[::-1].round(4).tolist()})
        # DP / Dirichlet variational mixtures
        fits = Parallel(n_jobs=n_jobs)(delayed(fit_bgmm_cfg)(*c) for c in bgmm_cfgs())
        comp_rows = []
        full_preds = {}
        for (basis, pca, T, gamma, ptype), (m, secs) in zip(bgmm_cfgs(), fits):
            *_, Z = _basis(src, key, pca)
            w = m.weights_
            row = {"model": f"BGMM {ptype} T={T} gamma={gamma}", "basis": basis, "T": T, "gamma": gamma, "prior": ptype,
                   "elbo": float(m.lower_bound_), "n_eff_1pct": int((w > 0.01).sum()), "n_active": int((w > 1e-3).sum()),
                   "converged": bool(m.converged_), "n_iter": int(m.n_iter_), "fit_seconds": secs,
                   "weights": np.sort(w)[::-1].round(4).tolist()}
            if basis == "pca12":
                ll, ka = plugin_mixture_loglik(Z, w, m.means_, m.covariances_, min_weight=1e-3)
                row.update({"loglik_plugin": ll, "bic_plugin": mixture_bic(ll, ka, Z.shape[1], len(Z))})
            R = m.predict_proba(Z[:n_tr])
            rates = label_rates(R, ds.y)
            hard = R.argmax(1)
            for k in np.argsort(-w):
                if w[k] > 1e-3:
                    comp_rows.append({"model": row["model"], "basis": basis, "component": int(k), "weight": float(w[k]),
                                      "n_train_hard": int((hard == k).sum()), "train_pos_hard": int(ds.y[hard == k].sum()),
                                      "rate_smoothed": float(rates[k])})
            pred = m.predict_proba(Z[n_tr:]) @ rates
            agr = lib_fm.reference_agreement(pred, s003)
            row.update({f"s003_{k}": v for k, v in agr.items()})
            full_preds[row["model"] + " " + basis] = pred
            ev.append(row)
            print(f"{row['model']:45s} {basis:5s} n_eff={row['n_eff_1pct']} active={row['n_active']} "
                  f"elbo={row['elbo']:.1f} bic={row.get('bic_plugin', float('nan')):.1f} "
                  f"side_agree_bnd={agr.get('side_agree_boundary', float('nan')):.3f} ({secs:.0f}s)", flush=True)
        # the S003 configuration itself, full-train mapping (same as above, for an apples-to-apples row)
        g2 = _mixture(src, key, None, 2, "full", 5, 0, 1e-6)
        *_, Zhv = _basis(src, key, None)
        R = g2.predict_proba(Zhv[:n_tr])
        pred = g2.predict_proba(Zhv[n_tr:]) @ label_rates(R, ds.y)
        ev.append({"model": "GMM K=2 (S003 config, full-train mapping)", "basis": "hv",
                   **{f"s003_{k}": v for k, v in lib_fm.reference_agreement(pred, s003).items()}})
        evdf = pd.DataFrame(ev)
        run.save_table("evidence", evdf)
        evdf.to_csv(tables / "r254_dpgmm_evidence.csv", index=False)
        cdf = pd.DataFrame(comp_rows)
        run.save_table("components", cdf)
        cdf.to_csv(tables / "r254_dpgmm_components.csv", index=False)
        run.save_array("s003_reference_recomputed", s003)
        print(evdf[["model", "basis", "n_eff_1pct", "n_active", "elbo", "bic_plugin", "s003_side_agree_boundary",
                    "s003_spearman_boundary", "s003_pseudo_auc_confident"]].to_string(index=False), flush=True)

        # ---- replica simulation (parametric bootstrap), PCA(12) basis
        g2_12 = _mixture(src, key, 12, 2, "full", 5, 0, 1e-6)
        dp = [f for c, f in zip(bgmm_cfgs(), fits) if c == ("pca12", 12, 10, 1.0, "dirichlet_process")][0][0]

        def generator(name):
            if name == "G1_gmm2":
                R = g2_12.predict_proba(Ztr12)
                return g2_12.weights_, g2_12.means_, g2_12.covariances_, label_rates(R, ds.y, alpha=0.0 + 1e-9)
            keep = dp.weights_ > 1e-3
            R = dp.predict_proba(Ztr12)[:, keep]
            w = dp.weights_[keep] / dp.weights_[keep].sum()
            return w, dp.means_[keep], dp.covariances_[keep], label_rates(R, ds.y, alpha=1.0)

        def replica(gen, rep):
            rng = np.random.default_rng(1000 + rep)
            w, mu, cov, rate = generator(gen)
            comp = rng.choice(len(w), size=len(Z12), p=w)
            Zs = np.empty_like(Z12)
            for k in range(len(w)):
                idx = np.where(comp == k)[0]
                Zs[idx] = rng.multivariate_normal(mu[k], cov[k], size=len(idx))
            ys = (rng.random(len(Zs)) < rate[comp]).astype(int)
            out = []
            fitted = {
                "GMM K=2": GaussianMixture(2, covariance_type="full", n_init=5, random_state=0, reg_covar=1e-6,
                                           max_iter=1000).fit(Zs),
                "DP-BGMM T=10 g=1": BayesianGaussianMixture(n_components=10, covariance_type="full",
                                                            weight_concentration_prior_type="dirichlet_process",
                                                            weight_concentration_prior=1.0, n_init=3, max_iter=3000,
                                                            tol=1e-4, random_state=0).fit(Zs)}
            scores = {}
            for mn, m in fitted.items():
                R = m.predict_proba(Zs[:n_tr])
                scores[mn] = (m.predict_proba(Zs[n_tr:]) @ label_rates(R, ys[:n_tr]), int((m.weights_ > 0.01).sum()))
            # supervised Bayesian QDA on the 250 simulated labelled rows only (R255 model), and a GMM+BQDA average
            bq = lib_fm.BayesQDA().fit(Zs[:n_tr], ys[:n_tr])
            scores["BayesQDA (250 labels only)"] = (bq.predict_proba(Zs[n_tr:])[:, 1], -1)
            scores["mean(GMM K=2, BayesQDA)"] = ((scores["GMM K=2"][0] + scores["BayesQDA (250 labels only)"][0]) / 2, -1)
            yt = ys[n_tr:]
            bayes = rate[comp][n_tr:]  # generator's own P(y=1 | component) as an oracle reference
            ref_b = (scores["GMM K=2"][0] > 0.1) & (scores["GMM K=2"][0] < 0.9)  # boundary rows of the GMM fit
            for mn, (sc, neff) in scores.items():
                b = (sc > 0.1) & (sc < 0.9)
                out.append({"generator": gen, "replica": rep, "model": mn, "test_auc": roc_auc_score(yt, sc),
                            "oracle_component_auc": roc_auc_score(yt, bayes) if len(np.unique(bayes)) > 1 else float("nan"),
                            "n_boundary": int(b.sum()),
                            "boundary_auc": roc_auc_score(yt[b], sc[b]) if b.sum() > 5 and len(np.unique(yt[b])) > 1 else float("nan"),
                            "auc_on_gmm_boundary_rows": roc_auc_score(yt[ref_b], sc[ref_b])
                            if ref_b.sum() > 5 and len(np.unique(yt[ref_b])) > 1 else float("nan"),
                            "n_gmm_boundary": int(ref_b.sum()), "n_eff_1pct": neff})
            return out

        jobs = [(g, r) for g in ("G1_gmm2", "G2_dpgmm") for r in range(a.replicas)]
        rep_rows = sum(Parallel(n_jobs=n_jobs)(delayed(replica)(g, r) for g, r in jobs), [])
        rdf = pd.DataFrame(rep_rows)
        run.save_table("replica_auc", rdf)
        rdf.to_csv(tables / "r254_replica_auc.csv", index=False)
        summ = rdf.groupby(["generator", "model"])[["test_auc", "oracle_component_auc", "auc_on_gmm_boundary_rows",
                                                    "n_gmm_boundary", "boundary_auc", "n_eff_1pct"]].mean().reset_index()
        summ.to_csv(tables / "r254_replica_auc_summary.csv", index=False)
        print(summ.to_string(index=False), flush=True)
        run.finish(extra_summary={"official_cv": False, "evidence": ev, "replica_summary": summ.to_dict("records")})

if not a.skip_cv:
    CANDS = {"dpgmm_hv": (None, "DP-BGMM T=10 gamma=1 on the standardised block"),
             "dpgmm_pca12": (12, "DP-BGMM T=10 gamma=1 on the PCA(12) basis of the block")}
    rows = []
    refs = {"r003_gmm2_hv": ref_cv}
    for name, (pca, desc) in CANDS.items():
        cfg = {"name": f"r254_{name}", "description": desc, "model_code": "research/r254_dp_gmm.py + research/lib_fm.py",
               "cv": scheme.to_dict(), "test_pred": "fold_mean", "uses_unlabeled_test_features": True,
               "params": {"n_components": 10, "gamma": 1.0, "pca_dims": pca, "n_init": 3, "alpha": 1.0},
               "code_sha": lib_fm.code_hash(__file__)}
        with Run(f"r254_{name}", cfg, kind="experiment", source=src, author=a.author,
                 hypothesis="A DP mixture fitted on the 20,000 unlabeled rows is as good as the K=2 GMM on CV (AUC "
                            "1.0, saturated) and differs from S003 only on the ~200 boundary rows; CV log-loss within "
                            "10% of the GMM K=2 reference.") as run:
            model = TransductiveBGMM("highvar", 10, gamma=1.0, pca_dims=pca, n_init=3, source=src)
            cv = cross_validate(model, ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=1)
            s = run.save_cv(cv)
            cal = lib_fm.oof_calibration(cv.oof, ds.y)
            agr = lib_fm.reference_agreement(cv.test_pred, s003)
            mix = lib_fm.fit_bgmm(src, key, pca, 10, 1.0, 3, 0)
            mixinfo = {"weights": np.sort(mix.weights_)[::-1].round(5).tolist(), "n_eff_1pct": int((mix.weights_ > 0.01).sum()),
                       "elbo": float(mix.lower_bound_), "converged": bool(mix.converged_)}
            pc = lib_fm.paired_table({name: cv}, refs)[0]
            rows.append({"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], **cal, **mixinfo,
                         **{f"s003_{k}": v for k, v in agr.items()}})
            run.finish(extra_summary={"official_cv": True, **cal, "mixture": mixinfo, "agreement_with_s003": agr,
                                      "paired_vs_r003_gmm2_hv": pc, "uses_unlabeled_test_features": True})
            print(f"{name:12s} auc={s['auc_mean']:.5f} logloss={cal['cv_logloss']:.4f} brier={cal['cv_brier']:.5f} "
                  f"n_eff={mixinfo['n_eff_1pct']} side_agree_bnd={agr.get('side_agree_boundary', float('nan')):.3f} "
                  f"run={run.run_id}", flush=True)
    cal = lib_fm.oof_calibration(ref_cv.oof, ds.y)
    rows.append({"candidate": "[ref] r003_gmm2_hv (recomputed)", "run_id": "20261022-050708_r003_gmm2_hv",
                 "auc_mean": float(ref_cv.fold_scores["auc"].mean()), **cal})
    pd.DataFrame(rows).to_csv(tables / "r254_dpgmm_candidates.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
