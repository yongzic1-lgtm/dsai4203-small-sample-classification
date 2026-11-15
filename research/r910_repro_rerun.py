"""R910 - reproducibility check (verifier `repro`): re-create three archived models with TODAY's code and
re-run them with the OFFICIAL CV on a fresh Colab VM.

  s003     TransductiveGMM("highvar", K=2, full), n_jobs 2   = run 20261022-050708_r003_gmm2_hv (submission S003)
  e3final  E3 honest pipeline: StandardScaler + CorrelationTopK(20, pearson) + L2 logistic C=0.1 (liblinear),
           test_pred = refit on all 250 rows                 = run 20261022-063908_e3_leakage (final model)
           (CV = the honest k_feat=20 / 5-fold cell of fold_scores_grid.csv; the leaky CV is re-run too)
  r201qda  lib_sup.Tuned(HighVarColumns + StandardScaler + QDA, nested reg_param), n_jobs -1
                                                              = run 20261022-071234_r201_qda_block

Each model is rebuilt from the code it was archived with: the archived config.yaml of these runs only
describes the model (it names model_code), so the estimator comes from research/r003 (constructor copied
below, r003 has no import guard), extensions/e3_leakage/run.py (load_config / with_selector / leaky_columns)
and research/r201_sup_quadratic.candidates(). The E3 fields the archived config.yaml holds are asserted here
(E3_ARCHIVED) and compared with the archived file by research/r911_repro_compare.py.

Runs written
  r910_s003_gmm2_hv, r910_e3_final_k20, r910_r201_qda_block   kind=experiment, official CV, test_pred.npy
  r910_repro_env                                              kind=analysis: environment (every pinned
      package, BLAS / threadpool info, CPU model) + BLAS-thread sensitivity: every model again with
      threadpoolctl.threadpool_limits(1) and joblib n_jobs=1, compared in-VM with the default-thread run.
The comparison with the ARCHIVED arrays (experiments/ is not synced to the VM) is research/r911_repro_compare.py.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vrepro -- research/r910_repro_rerun.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from importlib import metadata

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_sup  # noqa: E402
import r201_sup_quadratic as R201  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.models import build_pipeline  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import TransductiveGMM  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

ARCHIVED = {"s003": "20261022-050708_r003_gmm2_hv", "e3final": "20261022-063908_e3_leakage",
            "r201qda": "20261022-071234_r201_qda_block"}
OFFICIAL_SIG = "5708ce5c3828"
# fields of experiments/20261022-063908_e3_leakage/config.yaml that define the final model
E3_ARCHIVED = {"pipeline": {"scaler": "standard",
                            "model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 0.0, "solver": "liblinear"}},
                            "random_state": 0},
               "selection": {"method": "pearson"}, "leaderboard": {"k_feat": 20, "k_fold_reference": 5},
               "cv": {"n_repeats": 20, "seed": 2026, "stratify": True}}

HYP = ("Today's code (bundle hash in meta.json) reproduces the archived run bit for bit: all model seeds are "
       "fixed (GMM random_state 0, liblinear random_state 0, inner CV seed 0, QDA deterministic), the package "
       "pins are identical (numpy 2.4.6, scipy 1.17.1, sklearn 1.9.1) and the computational code is unchanged "
       "(only plot titles / an unused `scoring` argument differ from the archived bundles). Prediction: every "
       "fold AUC identical; test_pred max |diff| < 1e-12, Spearman 1.0. Possible source of a difference: "
       "OpenBLAS kernel / thread count of another VM type (summation order) -> diffs ~1e-15 in probabilities; "
       "for the nested QDA a flipped inner-CV tie could change a chosen reg_param in a fold (visible in "
       "fold_fit_info.csv). Refuted if any fold AUC differs by > 1e-9 or test Spearman < 0.999999.")


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()[:16]


def load_e3():
    spec = importlib.util.spec_from_file_location("e3_run", paths.ROOT / "extensions" / "e3_leakage" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def pinned_versions() -> pd.DataFrame:
    rows = []
    for line in (paths.ROOT / "colab" / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, pin = line.split("==", 1)
        try:
            inst = metadata.version(name)
        except metadata.PackageNotFoundError:
            inst = None
        rows.append({"package": name, "pinned": pin, "installed": inst, "match": inst == pin})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="raw", choices=paths.SOURCES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--skip-threads", action="store_true", help="skip the threadpool_limits(1) repeats")
    ap.add_argument("--max-folds", type=int, default=None, help="smoke test only (synthetic data): first N folds")
    ap.add_argument("--only", nargs="*", default=None, choices=["s003", "e3final", "r201qda"], help="subset of models")
    a = ap.parse_args()
    if a.max_folds and a.data == "raw":
        raise SystemExit("--max-folds is for smoke tests on --data synthetic only")
    src = a.data
    lib_sup.CURRENT["source"] = src

    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
    scheme = Scheme.from_dict(official)
    ds = data.load(src)
    splits = make_splits(ds.y, scheme)
    if a.max_folds:
        splits = splits[:a.max_folds]
    X, y, Xt = ds.X.to_numpy(dtype=float), ds.y, ds.X_test.to_numpy(dtype=float)

    # ---- model builders (today's code) ------------------------------------------------------------
    e3 = load_e3()
    e3cfg = e3.load_config(e3.DEFAULT_CONFIG, quick=False)
    for k, v in E3_ARCHIVED.items():
        got = {kk: e3cfg[k].get(kk) for kk in v} if k == "cv" else e3cfg[k]
        if got != v:
            raise SystemExit(f"E3 config `{k}` today {got} != archived {v}")
    k_lb, method = int(e3cfg["leaderboard"]["k_feat"]), e3cfg["selection"]["method"]
    e3_honest_spec = e3.with_selector(e3cfg["pipeline"], {"name": "corr_topk", "k": k_lb, "method": method})
    e3_leaky_cols = [int(c) for c in e3.leaky_columns(X, y, k_lb, method)]
    e3_leaky_spec = e3.with_selector(e3cfg["pipeline"], {"name": "fixed_columns", "columns": e3_leaky_cols})
    qda = next(c for c in R201.candidates() if c.name == "qda_block")
    gmm_cols = "highvar" if src == "raw" else "all"  # the synthetic sandbox data have no high-variance block

    models = {
        "s003": dict(name="r910_s003_gmm2_hv", n_jobs=2, test_pred="fold_mean", X=ds.X, Xt=ds.X_test,
                     make=lambda d: TransductiveGMM(gmm_cols, 2, source=src),
                     desc="R910 re-run of 20261022-050708_r003_gmm2_hv (S003): TransductiveGMM('highvar', 2), "
                          "constructor as in research/r003_transductive_vs_supervised.py",
                     model_code="ssc.transductive.TransductiveGMM('highvar', 2, source=src) [= r003 CANDIDATES['gmm2_hv']]",
                     uses_unlabeled=True),
        "e3final": dict(name="r910_e3_final_k20", n_jobs=1, test_pred="refit", X=X, Xt=Xt,
                        make=lambda d: build_pipeline(e3_honest_spec), pipeline=e3_honest_spec,
                        desc="R910 re-run of the E3 final model (20261022-063908_e3_leakage): scaler + "
                             "CorrelationTopK(20, pearson) + logreg C=0.1; honest CV; test_pred = refit on 250 rows",
                        model_code="extensions/e3_leakage/run.py (load_config, with_selector) + ssc.models.build_pipeline",
                        uses_unlabeled=False),
        "r201qda": dict(name="r910_r201_qda_block", n_jobs=-1, test_pred="fold_mean", X=ds.X, Xt=ds.X_test,
                        make=qda.make,
                        desc="R910 re-run of 20261022-071234_r201_qda_block: " + qda.desc,
                        model_code="research/r201_sup_quadratic.py candidates()['qda_block'] + research/lib_sup.py",
                        uses_unlabeled=False, tuning=qda.tuning),
    }

    if a.only:
        models = {k: v for k, v in models.items() if k in a.only}
    results, fit_infos = {}, {}
    print(f"[r910] cpus={os.cpu_count()} cpu='{cpu_model()}' python={platform.python_version()}", flush=True)
    for key, m in models.items():
        cfg = {"name": m["name"], "description": m["desc"], "model_code": m["model_code"],
               "reproduces_run": ARCHIVED[key], "cv": scheme.to_dict(), "test_pred": m["test_pred"],
               "n_jobs": m["n_jobs"], "uses_unlabeled_test_features": m["uses_unlabeled"]}
        if "pipeline" in m:
            cfg["pipeline"] = m["pipeline"]
        if "tuning" in m:
            cfg["tuning"] = m["tuning"]
        log_dir = tempfile.mkdtemp(prefix=f"r910_{key}_")
        with Run(m["name"], cfg, kind="experiment", source=src, author=a.author, hypothesis=HYP) as run:
            t0 = time.perf_counter()
            cv = cross_validate(m["make"](log_dir), m["X"], y, splits, scheme, X_test=m["Xt"],
                                test_pred=m["test_pred"], n_jobs=m["n_jobs"])
            secs = time.perf_counter() - t0
            s = run.save_cv(cv)
            extra = {"official_cv": cv.split_signature == OFFICIAL_SIG, "reproduces_run": ARCHIVED[key],
                     "test_pred_sha16": sha(cv.test_pred), "oof_sha16": sha(cv.oof),
                     "fold_auc_sha16": sha(cv.fold_scores["auc"].to_numpy()), "wall_seconds": round(secs, 1),
                     "cpus": os.cpu_count(), "cpu_model": cpu_model()}
            if key == "r201qda":
                fi = lib_sup.collect_logs(log_dir, ds.X, splits)
                if fi is not None:
                    run.save_table("fold_fit_info", fi)
                    fit_infos[key] = fi
                    extra["chosen_params_top"] = {k: int(v) for k, v in fi["best_params"].value_counts().head(6).items()}
                extra.update(lib_sup.prob_metrics(cv.oof, y))
            s = run.finish(extra_summary=extra)
            results[key] = {"run_id": run.run_id, "cv": cv}
            print(f"[r910] {m['name']:24s} auc={s['auc_mean']:.6f} sig={cv.split_signature} "
                  f"test_sha={extra['test_pred_sha16']} {secs:.1f}s run={run.run_id}", flush=True)

    # ---- analysis run: environment + BLAS-thread sensitivity ------------------------------------------
    acfg = {"name": "r910_repro_env", "description": "environment of the R910 re-run + threadpool_limits(1) repeats",
            "reruns": {k: v["run_id"] for k, v in results.items()}, "archived": ARCHIVED}
    with Run("r910_repro_env", acfg, kind="analysis", source=src, author=a.author,
             hypothesis="BLAS thread count does not change any result (limits=1 vs default: max |diff| < 1e-12, "
                        "identical fold AUCs); every pinned package is installed at its pin.") as run:
        pv = pinned_versions()
        run.save_table("pinned_vs_installed", pv)
        tp = pd.DataFrame(threadpool_info())
        run.save_table("threadpool_info", tp)
        try:
            npcfg = np.show_config(mode="dicts")
        except Exception as e:  # noqa: BLE001
            npcfg = {"error": repr(e)}
        (run.dir / "numpy_config.json").write_text(json.dumps(npcfg, indent=2, default=str), encoding="utf-8")
        leaky = cross_validate(build_pipeline(e3_leaky_spec), X, y, splits, scheme, test_pred="none", n_jobs=1)
        run.save_table("e3_leaky_fold_scores", leaky.fold_scores)
        rows = []
        if not a.skip_threads:
            for key, m in models.items():
                log_dir = tempfile.mkdtemp(prefix=f"r910t_{key}_")
                t0 = time.perf_counter()
                with threadpool_limits(limits=1):
                    cv1 = cross_validate(m["make"](log_dir), m["X"], y, splits, scheme, X_test=m["Xt"],
                                         test_pred=m["test_pred"], n_jobs=1)
                secs = time.perf_counter() - t0
                cv0 = results[key]["cv"]
                run.save_array(f"test_pred_t1_{key}", cv1.test_pred)
                run.save_table(f"fold_scores_t1_{key}", cv1.fold_scores)
                d_auc = np.abs(cv1.fold_scores["auc"].to_numpy() - cv0.fold_scores["auc"].to_numpy())
                d_tp = np.abs(cv1.test_pred - cv0.test_pred)
                rows.append({"model": key, "rerun": results[key]["run_id"], "seconds_t1": round(secs, 1),
                             "fold_auc_max_abs_diff": float(d_auc.max()), "n_folds_differ": int((d_auc > 0).sum()),
                             "test_max_abs_diff": float(d_tp.max()), "test_n_differ": int((d_tp > 0).sum()),
                             "test_spearman": float(spearmanr(cv1.test_pred, cv0.test_pred).statistic),
                             "test_bit_identical": bool(np.array_equal(cv1.test_pred, cv0.test_pred)),
                             "oof_max_abs_diff": float(np.nanmax(np.abs(cv1.oof - cv0.oof)))})
                print(f"[r910] threads=1 {key}: {rows[-1]}", flush=True)
        th = pd.DataFrame(rows)
        run.save_table("thread_sensitivity", th)
        out = paths.tables_dir(src)
        out.mkdir(parents=True, exist_ok=True)
        th.to_csv(out / "r910_thread_sensitivity.csv", index=False)
        pv.to_csv(out / "r910_pinned_vs_installed.csv", index=False)
        run.finish(extra_summary={
            "python": platform.python_version(), "cpus": os.cpu_count(), "cpu_model": cpu_model(),
            "all_pins_match": bool(pv["match"].all()), "pin_mismatches": pv.loc[~pv["match"]].to_dict("records"),
            "blas": [{k: r.get(k) for k in ("internal_api", "version", "num_threads", "architecture", "threading_layer")}
                     for r in threadpool_info()],
            "e3_leaky_auc_mean": float(leaky.fold_scores["auc"].mean()),
            "e3_leaky_columns": e3_leaky_cols,
            "thread_sensitivity": rows,
        })
        print(f"[r910] env: pins match={bool(pv['match'].all())}; leaky E3 auc={leaky.fold_scores['auc'].mean():.6f}; "
              f"run={run.run_id}", flush=True)
        print(pv.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
