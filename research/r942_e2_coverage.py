"""R942 (verifier "ext") - are E2's extrapolation intervals honest? A coverage simulation.

E2 (extensions/e2_learning_curve/run.py, unmodified functions) is re-run end to end on M parametric
replicas of the real data (research/lib_gen_replica.py): each replica is a fresh 250-row labelled
sample from the stand-in population. For every replica the E2 protocol (10 stratified subsets per size
n = 50..250, 5-fold x 4 CV inside each, inverse power law fit weighted by the SE, bootstrap over the
subset scores, log-linear / unweighted / drop-50 sensitivity fits, envelope) gives a point and a
90% interval for AUC at 500 training rows. Coverage = share of replicas whose interval contains the
population's expected AUC at 500 rows (from R941, passed with --truth-*).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vext --high-mem -- research/r942_e2_coverage.py --n-jobs -1 --truth-l1 X --truth-qda Y
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

p = argparse.ArgumentParser()
p.add_argument("--n-rep", type=int, default=100, help="number of replica 250-row samples")
p.add_argument("--n-boot", type=int, default=500, help="bootstrap refits (E2 uses 2000; 500 for speed)")
p.add_argument("--sens-n-boot", type=int, default=200)
p.add_argument("--models", nargs="*", default=["l1_all", "qda_hv"])
p.add_argument("--truth-l1", type=float, default=None)
p.add_argument("--truth-qda", type=float, default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--smoke", action="store_true", help="2 subsets, 1 CV repeat (local smoke test only)")
a = p.parse_args()

HYP = ("E2's intervals resample subsets of ONE 250-row sample, so as statements about the population AUC at 500 rows "
       "they should under-cover: the 90% bootstrap interval of the power law covers the replica truth in clearly fewer "
       "than 90% of replicas (expected 30-60%), the envelope in more. The power-law point is roughly unbiased if the "
       "back-test (R308) is representative.")

_MOD = {}


def e2mod():
    if "m" not in _MOD:
        spec = importlib.util.spec_from_file_location("e2_run_r942", ROOT / "extensions/e2_learning_curve/run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD["m"] = mod
    return _MOD["m"]


def one(ps, r: int, model: str, n_boot: int, sens_n_boot: int, smoke: bool = False) -> dict:
    import lib_gen_replica as G
    mod = e2mod()
    cfg = mod.load_config(ROOT / "extensions/e2_learning_curve/config.yaml", False)
    cfg = copy.deepcopy(cfg)
    cfg["fit"]["n_boot"] = n_boot
    cfg["fit"]["sensitivity"]["n_boot"] = sens_n_boot
    if smoke:
        cfg["n_subsets"], cfg["cv"]["n_repeats"] = 2, 1
    rep = G.sample_parametric(ps, seed=70000 + r)
    X, y = rep.X_train, rep.y_train
    if model == "qda_hv":
        est = G.make_supervised("qda_hv", rep.X_all)
        mod.build_pipeline = lambda spec, _e=est: __import__("sklearn.base", fromlist=["clone"]).clone(_e)
    else:
        from ssc.models import build_pipeline
        mod.build_pipeline = build_pipeline
    cfg["out_of_subset"] = False
    rows = [mod.score_subset(n, i, cfg, X, y)["row"] for n in cfg["sizes"] for i in range(cfg["n_subsets"])]
    sub = pd.DataFrame(rows)
    fit = mod.fit_and_bootstrap(sub, cfg)
    ext = fit["extrapolation"]
    lo_c, hi_c = fit["interval_cols"]
    out = {"replica": r, "model": model}
    for name in ("power_law", "log_linear", "power_law_unweighted", "power_law_drop_50", "envelope"):
        e = ext[(ext["model"] == name) & (ext["x"] == 500.0)]
        if len(e):
            e = e.iloc[0]
            out.update({f"{name}_point": float(e["point"]), f"{name}_lo": float(e[lo_c]), f"{name}_hi": float(e[hi_c])})
    m = sub.groupby("size")["cv_auc"].mean()
    out.update({f"cv_mean_n{int(k)}": float(v) for k, v in m.items()})
    out["frac_constant_n50"] = float(sub[sub["size"] == 50]["frac_constant_folds"].mean())
    out["gain_pl_median"] = fit["gains"]["pl"]["median"]
    out["gain_pl_frac_positive"] = fit["gains"]["pl"]["frac_positive"]
    return out


from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
import lib_gen_replica as G  # noqa: E402

with Run("r942_e2_coverage", {"n_rep": a.n_rep, "n_boot": a.n_boot, "sens_n_boot": a.sens_n_boot, "models": a.models,
                              "truth": {"l1_all": a.truth_l1, "qda_hv": a.truth_qda},
                              "replica": "sample_parametric(fit_parametric(X_all, y)), seed 70000 + r",
                              "e2_code": "extensions/e2_learning_curve/run.py (score_subset, fit_and_bootstrap; out_of_subset off)",
                              "model_code": "research/r942_e2_coverage.py"},
         kind="analysis", source="raw", hypothesis=HYP) as run:
    ps = G.fit_parametric(unlabeled_features("raw"), data.load("raw").y)
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(ps, r, m, a.n_boot, a.sens_n_boot, a.smoke)
                                    for m in a.models for r in range(a.n_rep))
    df = pd.DataFrame(res)
    run.save_table("replica_intervals", df)
    truth = {"l1_all": a.truth_l1, "qda_hv": a.truth_qda}
    cov = []
    for m, g in df.groupby("model"):
        t = truth.get(m)
        row = {"model": m, "n": len(g), "truth_500": t,
               "pl_point_mean": g["power_law_point"].mean(), "pl_point_sd": g["power_law_point"].std(),
               "pl_width_mean": (g["power_law_hi"] - g["power_law_lo"]).mean(),
               "env_width_mean": (g["envelope_hi"] - g["envelope_lo"]).mean(),
               "cv_mean_n250_mean": g.get("cv_mean_n250", pd.Series(dtype=float)).mean(),
               "cv_mean_n250_sd": g.get("cv_mean_n250", pd.Series(dtype=float)).std(),
               "gain_frac_positive_mean": g["gain_pl_frac_positive"].mean()}
        if t is not None:
            for name in ("power_law", "log_linear", "power_law_unweighted", "power_law_drop_50", "envelope"):
                if f"{name}_lo" in g:
                    row[f"cover_{name}"] = float(((g[f"{name}_lo"] <= t) & (t <= g[f"{name}_hi"])).mean())
                    row[f"bias_{name}_point"] = float(g[f"{name}_point"].mean() - t)
                    row[f"rmse_{name}_point"] = float(np.sqrt(((g[f"{name}_point"] - t) ** 2).mean()))
        cov.append(row)
    cov = pd.DataFrame(cov)
    run.save_table("coverage", cov)
    paths.tables_dir("raw").mkdir(parents=True, exist_ok=True)
    cov.to_csv(paths.tables_dir("raw") / "r942_e2_coverage.csv", index=False)
    run.finish(extra_summary={"coverage": cov.to_dict(orient="records")})
    pd.set_option("display.width", 250)
    print(cov.round(4).T.to_string())
    print("run:", run.run_id)
