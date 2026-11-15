"""R257 - follow-up to R253: why did the 4-chain horseshoe fit on all 150 features not mix
(R-hat 4.6 on beta, ESS 2), and is the posterior inclusion interpretable on an identifiable design?

Written AFTER seeing R253 (reported as a follow-up). Hypothesis: the 8 exact linear dependencies among the
150 features (R001) leave beta unidentified along 8 directions; the horseshoe then has several sparse
representations of the same linear predictor, and chains settle in different ones. Predictions are
unaffected: the R-hat of the linear predictor eta = alpha + X beta on the training rows is close to 1.

Part A, analysis run r257_hs_identifiability:
  * null space of the standardised 20,000 x 150 feature matrix (label-free): features with loadings;
  * all-250 refit of R253's hs_all (4 chains x 1000/1000): R-hat of beta vs R-hat of eta, per-chain means;
  * the same model on an identifiable design (DropCollinear: QR with column pivoting on the fitting rows,
    labels unused) with 4 chains: R-hat, ESS, posterior inclusion table.
Part B, experiment run r257_hs_indep: StandardScaler -> DropCollinear -> horseshoe logistic, official CV.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm --no-sync -- research/r257_hs_identifiability.py --n-jobs 12
"""

from __future__ import annotations

import argparse
import os
import time

import lib_fm

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=["raw", "synthetic"])
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--skip-cv", action="store_true")
a = p.parse_args()

if lib_fm.ON_VM:
    py, venv_info = lib_fm.ensure_venv("numpyro", ["numpyro", "jax"])
    lib_fm.reexec_in(py)
else:
    venv_info = {}
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 " \
                          "--xla_force_host_platform_device_count=4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.features import abs_correlation  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import resolve_columns, unlabeled_features  # noqa: E402
from ssc.validation import cross_validate  # noqa: E402

from lib_fm import DropCollinear, HorseshoeLogReg  # noqa: E402

src = a.data
ds, scheme, splits = lib_fm.official_setup(src)
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)
X, y = ds.X.to_numpy(), ds.y
X_all = unlabeled_features(src)
hv = [int(c) for c in resolve_columns("highvar", X_all)]
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
versions = lib_fm.freeze_versions()


def chains_rhat(arr):
    """split R-hat per element of arr with shape (chains, draws, ...)."""
    from numpyro.diagnostics import split_gelman_rubin
    return np.asarray(split_gelman_rubin(arr))


cfg = {"name": "r257_hs_identifiability", "kind": "analysis", "inputs": "all 250 labelled rows; label-free null space",
       "mcmc": {"warmup": 1000, "samples": 1000, "chains": 4}, "versions": versions, "code_sha": lib_fm.code_hash(__file__)}
with Run("r257_hs_identifiability", cfg, kind="analysis", source=src, author=a.author,
         hypothesis="beta is not identified along the 8 exact linear dependencies, so chains disagree on beta (R-hat >> 1) "
                    "but agree on the linear predictor (R-hat(eta) < 1.1); on a QR-reduced identifiable design the chains "
                    "mix (R-hat(beta) < 1.1) and CV AUC is unchanged within 2 split-SE.") as run:
    # ---- label-free null space
    Z = (X_all - X_all.mean(0)) / X_all.std(0)
    ev, V = np.linalg.eigh(Z.T @ Z / len(Z))
    null = V[:, ev < 1e-8 * ev.max()]
    load = np.abs(null).max(1)
    involved = np.where(load > 0.05)[0]
    ns = pd.DataFrame({"feature": np.arange(150), "max_abs_null_loading": load, "in_block": np.isin(np.arange(150), hv)})
    ns = ns.sort_values("max_abs_null_loading", ascending=False)
    ns.to_csv(tables / "r257_null_space_loadings.csv", index=False)
    run.save_table("null_space_loadings", ns)
    print(f"null space dimension {null.shape[1]}; features with |loading| > 0.05: {involved.tolist()} "
          f"(in block: {[int(j) for j in involved if j in hv]})", flush=True)
    print("rank of the 250 train rows (standardised):", np.linalg.matrix_rank((X - X.mean(0)) / X.std(0)), flush=True)

    out = {"null_dim": int(null.shape[1]), "null_features": involved.tolist()}
    for label, pipe in (("full150", Pipeline([("scale", StandardScaler()), ("model", HorseshoeLogReg(num_chains=4, num_warmup=1000, num_samples=1000))])),
                        ("indep", Pipeline([("scale", StandardScaler()), ("drop", DropCollinear()),
                                            ("model", HorseshoeLogReg(num_chains=4, num_warmup=1000, num_samples=1000))]))):
        t0 = time.time()
        pipe.fit(X, y)
        hs = pipe[-1]
        Xd = pipe[:-1].transform(X)
        B = hs.beta_.reshape(4, -1, hs.beta_.shape[1])
        A = hs.alpha_.reshape(4, -1)
        eta = np.einsum("cdp,np->cdn", B, Xd) + A[:, :, None]
        rb, re_ = chains_rhat(B), chains_rhat(eta)
        cols = pipe.named_steps["drop"].columns_ if "drop" in pipe.named_steps else np.arange(150)
        dropped = pipe.named_steps["drop"].dropped_.tolist() if "drop" in pipe.named_steps else []
        info = {"seconds": time.time() - t0, "rhat_beta_max": float(np.nanmax(rb)), "rhat_beta_median": float(np.nanmedian(rb)),
                "rhat_eta_max": float(np.nanmax(re_)), "rhat_eta_median": float(np.nanmedian(re_)),
                "ess_min_reported": hs.ess_min_, "divergences": hs.n_divergent_, "n_columns": int(len(cols)),
                "dropped_columns": dropped, "m_eff": float(hs.shrinkage().sum())}
        out[label] = info
        print(f"{label}: {info}", flush=True)
        corr = abs_correlation(X[:, cols], y)
        inc = pd.DataFrame({"feature": cols, "in_block": np.isin(cols, hv), "beta_mean": hs.beta_.mean(0),
                            "q05": np.quantile(hs.beta_, 0.05, 0), "q95": np.quantile(hs.beta_, 0.95, 0),
                            "shrinkage_1mk": hs.shrinkage(), "rhat_beta": rb, "abs_corr_train": corr,
                            **{f"chain{c}_mean": B[c].mean(0) for c in range(4)}})
        inc["ci90_excludes_0"] = (inc["q05"] > 0) | (inc["q95"] < 0)
        inc["in_null_space"] = np.isin(cols, involved)
        inc = inc.sort_values("shrinkage_1mk", ascending=False)
        inc.to_csv(tables / f"r257_inclusion_{label}.csv", index=False)
        run.save_table(f"inclusion_{label}", inc)
        print(inc.head(15)[["feature", "in_block", "in_null_space", "beta_mean", "q05", "q95", "shrinkage_1mk",
                            "rhat_beta", "chain0_mean", "chain1_mean", "chain2_mean", "chain3_mean"]].to_string(index=False), flush=True)
    run.finish(extra_summary={"official_cv": False, **out})

if not a.skip_cv:
    refs = {rn: lib_fm.r003_reference(rn, ds, splits, scheme, src, n_jobs=n_jobs)
            for rn in ("r003_sup_l1_all", "r003_sup_logreg_hv", "r003_gmm2_hv")}
    s003 = refs["r003_gmm2_hv"].test_pred
    cfg = {"name": "r257_hs_indep", "description": "regularised-horseshoe logistic on a QR-reduced identifiable design of the 150 features",
           "model_code": "research/r257_hs_identifiability.py + research/lib_fm.py", "cv": scheme.to_dict(), "test_pred": "fold_mean",
           "uses_unlabeled_test_features": False, "prior": {"type": "regularised horseshoe", "p0": 10, "slab_scale": 2.0, "slab_df": 4.0},
           "mcmc": {"sampler": "NUTS", "warmup": 500, "samples": 500, "chains": 1}, "versions": versions,
           "code_sha": lib_fm.code_hash(__file__)}
    with Run("r257_hs_indep", cfg, kind="experiment", source=src, author=a.author,
             hypothesis="Removing the exactly dependent columns changes the parametrisation, not the linear predictor space: "
                        "CV AUC within 2 split-SE of r253_hs_all (0.926).") as run:
        model = Pipeline([("scale", StandardScaler()), ("drop", DropCollinear()), ("model", HorseshoeLogReg())])
        cv = cross_validate(model, ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
        s = run.save_cv(cv)
        cal = lib_fm.oof_calibration(cv.oof, y)
        agr = lib_fm.reference_agreement(cv.test_pred, s003)
        pcs = lib_fm.paired_table({"hs_indep": cv}, refs)
        run.save_table("paired_vs_references", pd.DataFrame(pcs))
        run.finish(extra_summary={"official_cv": True, **cal, "agreement_with_s003": agr, "paired_vs_references": pcs,
                                  "uses_unlabeled_test_features": False})
        print(f"hs_indep auc={s['auc_mean']:.5f} +/- {s['auc_std_repeat']:.5f} logloss={cal['cv_logloss']:.4f} "
              f"vsL1={pcs[0]['mean_diff']:+.4f} (se {pcs[0]['se_split']:.4f}) run={run.run_id}", flush=True)
