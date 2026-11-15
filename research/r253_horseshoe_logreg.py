"""R253 - Bayesian sparse logistic regression with a regularised horseshoe prior (NumPyro NUTS).

Supervised only (250 labelled rows), official CV (project.yaml official_cv), test predictions =
fold_mean of the posterior-mean probabilities. Candidates (experiment runs):

  hs_all      regularised horseshoe logistic on all 150 standardised features
              (p0 = 10 expected relevant features -> global scale tau0, Piironen & Vehtari 2017)
  hs_quadhv   the same prior on [150 linear terms + 91 squares/products of the 13 block columns]
              (block = label-free high-variance rule; lets a sparse linear model express the
              covariance difference between the classes)

References (recomputed R003 configurations, identical splits, not archived again):
L1 logistic all features (0.8469), L2 logistic on the block (0.8977), QDA block (0.9958), GMM (S003).

Analysis run r253_hs_inclusion: full fit on all 250 rows, 4 chains x (1000 warmup + 1000 draws), per
feature posterior summaries and shrinkage 1 - kappa_j ("posterior inclusion"), R-hat / ESS /
divergences; compared with the block and with the L1 solution.

Runs in /content/venv_numpyro (pinned project packages + jax CPU + numpyro), built by lib_fm.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm --no-sync -- research/r253_horseshoe_logreg.py --n-jobs 12
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
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--p0", type=float, default=10.0)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--samples", type=int, default=500)
p.add_argument("--skip-analysis", action="store_true")
p.add_argument("--skip-cv", action="store_true")
a = p.parse_args()

if lib_fm.ON_VM:
    py, venv_info = lib_fm.ensure_venv("numpyro", ["numpyro", "jax"])
    lib_fm.reexec_in(py)
else:
    venv_info = {}
# CPU, one XLA thread per process (folds run in parallel processes); inherited by joblib workers
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 " \
                          "--xla_force_host_platform_device_count=4"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.features import abs_correlation  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import resolve_columns, unlabeled_features  # noqa: E402
from ssc.validation import cross_validate  # noqa: E402

from lib_fm import HorseshoeLogReg, QuadBlock  # noqa: E402

src = a.data
ds, scheme, splits = lib_fm.official_setup(src)
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)
hv = [int(c) for c in resolve_columns("highvar", unlabeled_features(src))]
versions = lib_fm.freeze_versions()
print(f"cpus={os.cpu_count()} n_jobs={n_jobs} block={hv} versions={versions}", flush=True)
HS = dict(p0=a.p0, num_warmup=a.warmup, num_samples=a.samples, num_chains=1, target_accept=0.95, seed=0)

CANDS = {
    "hs_all": (lambda: Pipeline([("scale", StandardScaler()), ("model", HorseshoeLogReg(**HS))]),
               "regularised-horseshoe logistic, all 150 standardised features", False),
    "hs_quadhv": (lambda: Pipeline([("quad", QuadBlock(block=tuple(hv))), ("model", HorseshoeLogReg(**HS))]),
                  "regularised-horseshoe logistic, 150 linear + 91 block squares/products", True),
}
names = a.only or list(CANDS)
tables, figs = paths.tables_dir(src), paths.figures_dir(src)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
X, y = ds.X.to_numpy(), ds.y

if not a.skip_cv:
    t0 = time.time()
    refs = {rn: lib_fm.r003_reference(rn, ds, splits, scheme, src, n_jobs=n_jobs)
            for rn in ("r003_sup_l1_all", "r003_sup_logreg_hv", "r003_sup_qda_hv", "r003_gmm2_hv")}
    s003 = refs["r003_gmm2_hv"].test_pred
    print(f"references in {time.time() - t0:.0f}s: " +
          ", ".join(f"{k}={v.fold_scores['auc'].mean():.4f}" for k, v in refs.items()), flush=True)
    # timing / sanity probe on one fold (not archived)
    t0 = time.time()
    sp = splits[0]
    m = CANDS[names[0]][0]().fit(X[sp.train_idx], y[sp.train_idx])
    print(f"probe {names[0]}: one fold {time.time() - t0:.1f}s, divergences={m[-1].n_divergent_}, "
          f"tau mean={m[-1].tau_.mean():.4f}", flush=True)

    rows, cvs = [], {}
    for name in names:
        make, desc, uses_unl = CANDS[name]
        cfg = {"name": f"r253_{name}", "description": desc, "model_code": "research/r253_horseshoe_logreg.py + research/lib_fm.py",
               "cv": scheme.to_dict(), "test_pred": "fold_mean", "uses_unlabeled_test_features": uses_unl,
               "prior": {"type": "regularised horseshoe", "p0": a.p0, "slab_scale": 2.0, "slab_df": 4.0},
               "mcmc": {"sampler": "NUTS", "warmup": a.warmup, "samples": a.samples, "chains": 1, "target_accept": 0.95},
               "versions": versions, "venv": {k: v for k, v in venv_info.items() if k != "freeze"},
               "code_sha": lib_fm.code_hash(__file__)}
        hyp = ("Horseshoe shrinkage separates the few relevant linear features from noise better than L1 on 200 rows: "
               "predicted CV AUC above L1 logistic (0.847) by > 2 split-SE but below L2 on the block (0.898)."
               if name == "hs_all" else
               "With squares/products of the block the sparse model can express the class covariance difference: "
               "predicted CV AUC 0.97-0.995, i.e. close to but below QDA on the block (0.996).")
        with Run(f"r253_{name}", cfg, kind="experiment", source=src, author=a.author, hypothesis=hyp) as run:
            t0 = time.time()
            cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
            secs = time.time() - t0
            s = run.save_cv(cv)
            cvs[name] = cv
            cal = lib_fm.oof_calibration(cv.oof, y)
            agr = lib_fm.reference_agreement(cv.test_pred, s003)
            pcs = lib_fm.paired_table({name: cv}, refs)
            run.save_table("paired_vs_references", pd.DataFrame(pcs))
            pc = {r["reference"]: r for r in pcs}
            rows.append({"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
                         "oof_auc_mean": s.get("oof_auc_mean"), **cal, "cv_seconds": secs,
                         **{f"diff_vs_{k}": v["mean_diff"] for k, v in pc.items()},
                         **{f"se_vs_{k}": v["se_split"] for k, v in pc.items()},
                         **{f"p_vs_{k}": v["corrected_p"] for k, v in pc.items()},
                         **{f"s003_{k}": v for k, v in agr.items()}, "uses_unlabeled_test_features": uses_unl})
            run.finish(extra_summary={"official_cv": True, **cal, "cv_seconds": secs, "agreement_with_s003": agr,
                                      "paired_vs_references": pcs, "uses_unlabeled_test_features": uses_unl})
            print(f"{name:10s} auc={s['auc_mean']:.5f} +/- {s['auc_std_repeat']:.5f} logloss={cal['cv_logloss']:.4f} "
                  f"vsL1={pc['r003_sup_l1_all']['mean_diff']:+.4f} (se {pc['r003_sup_l1_all']['se_split']:.4f}) "
                  f"vsQDA={pc['r003_sup_qda_hv']['mean_diff']:+.4f} pseudoAUC={agr['pseudo_auc_confident']:.4f} "
                  f"({secs:.0f}s) run={run.run_id}", flush=True)
    for rn, rcv in refs.items():
        rows.append({"candidate": f"[ref] {rn}", "run_id": "(recomputed R003 config)",
                     "auc_mean": float(rcv.fold_scores["auc"].mean()), "auc_std_repeat": rcv.summary()["auc_std_repeat"],
                     **lib_fm.oof_calibration(rcv.oof, y),
                     **{f"s003_{k}": v for k, v in lib_fm.reference_agreement(rcv.test_pred, s003).items()}})
    pd.DataFrame(rows).to_csv(tables / "r253_hs_candidates.csv", index=False)
    pd.DataFrame(lib_fm.paired_table(cvs, refs)).to_csv(tables / "r253_paired_vs_references.csv", index=False)
    print(pd.DataFrame(rows)[["candidate", "auc_mean", "auc_std_repeat", "cv_logloss", "s003_pseudo_auc_confident"]]
          .to_string(index=False), flush=True)

if not a.skip_analysis:
    cfg = {"name": "r253_hs_inclusion", "kind": "analysis", "inputs": "all 250 labelled rows",
           "mcmc": {"warmup": 1000, "samples": 1000, "chains": 4}, "p0": a.p0, "code_sha": lib_fm.code_hash(__file__)}
    with Run("r253_hs_inclusion", cfg, kind="analysis", source=src, author=a.author,
             hypothesis="The horseshoe posterior concentrates its inclusion mass on a few block features and on the "
                        "target-correlated features outside the block found in R002; the L1 solution selects "
                        "a superset with many noise features.") as run:
        corr = abs_correlation(X, y)
        l1 = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.1, l1_ratio=1.0, solver="liblinear",
                                                                          max_iter=5000, random_state=0))]).fit(X, y)
        l1coef = l1[-1].coef_.ravel()
        diag = {}
        for name, (make, desc, _) in CANDS.items():
            if name not in names:
                continue
            pipe = make()
            pipe.set_params(model__num_chains=4, model__num_warmup=1000, model__num_samples=1000)
            t0 = time.time()
            pipe.fit(X, y)
            hs = pipe[-1]
            B = hs.beta_
            n_feat = B.shape[1]
            labels = [str(j) for j in range(150)]
            if name == "hs_quadhv":
                iu = np.triu_indices(len(hv))
                labels += [f"{hv[i]}x{hv[j]}" for i, j in zip(*iu)]
            inc = pd.DataFrame({"term": labels[:n_feat], "beta_mean": B.mean(0), "beta_sd": B.std(0),
                                "q05": np.quantile(B, 0.05, 0), "q95": np.quantile(B, 0.95, 0),
                                "p_positive": (B > 0).mean(0), "shrinkage_1mk": hs.shrinkage(),
                                "p_abs_gt_0.1": (np.abs(B) > 0.1).mean(0)})
            inc["ci90_excludes_0"] = (inc["q05"] > 0) | (inc["q95"] < 0)
            inc["in_block"] = [t in {str(c) for c in hv} or "x" in t for t in inc["term"]]
            if name == "hs_all":
                inc["abs_corr_train"] = corr
                inc["l1_coef_C0.1"] = l1coef
            inc = inc.sort_values("shrinkage_1mk", ascending=False)
            run.save_table(f"inclusion_{name}", inc)
            inc.to_csv(tables / f"r253_inclusion_{name}.csv", index=False)
            diag[name] = {"seconds": time.time() - t0, "rhat_max": getattr(hs, "rhat_max_", None),
                          "ess_min": getattr(hs, "ess_min_", None), "divergences": hs.n_divergent_,
                          "m_eff": float(hs.shrinkage().sum()), "tau_mean": float(hs.tau_.mean()), "tau0": float(hs.tau0_),
                          "n_ci90_excl0": int(inc["ci90_excludes_0"].sum()),
                          "top10": inc["term"].head(10).tolist()}
            if name == "hs_all":
                top = inc.head(20)
                diag[name].update({
                    "block_in_top13": int(inc.head(13)["in_block"].sum()),
                    "l1_nonzero": int((np.abs(l1coef) > 1e-8).sum()),
                    "l1_nonzero_in_block": int(((np.abs(l1coef) > 1e-8) & np.isin(np.arange(150), hv)).sum()),
                    "spearman_shrink_vs_abscorr": float(pd.Series(inc["shrinkage_1mk"].to_numpy()).corr(
                        pd.Series(inc["abs_corr_train"].to_numpy()), method="spearman"))})
                fig, ax = plt.subplots(figsize=(12, 3.5))
                order = np.arange(150)
                sh = inc.set_index("term").loc[[str(j) for j in order], "shrinkage_1mk"].to_numpy()
                ax.bar(order, sh, color=["#c0392b" if j in hv else "#7f8c8d" for j in order], width=0.8)
                ax.set_xlabel("feature index (red = label-free high-variance block)")
                ax.set_ylabel("posterior mean 1 - kappa")
                ax.set_title("Regularised horseshoe logistic regression, all 250 rows: posterior inclusion (shrinkage) per feature")
                run.save_figure("inclusion_all", fig)
                fig.savefig(figs / "r253_hs_inclusion_all.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print("top 20 by inclusion:\n" + top[["term", "beta_mean", "q05", "q95", "shrinkage_1mk", "in_block",
                                                      "abs_corr_train", "l1_coef_C0.1"]].to_string(index=False), flush=True)
            else:
                print("quad model top 20:\n" + inc.head(20)[["term", "beta_mean", "q05", "q95", "shrinkage_1mk"]]
                      .to_string(index=False), flush=True)
            print(f"{name} diagnostics: {diag[name]}", flush=True)
        pd.DataFrame([{"model": k, **v} for k, v in diag.items()]).to_csv(tables / "r253_hs_diagnostics.csv", index=False)
        run.finish(extra_summary={"official_cv": False, "diagnostics": diag})
