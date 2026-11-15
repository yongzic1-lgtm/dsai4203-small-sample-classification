"""R308 - back-test of the E2 extrapolation on the real data: does the learning-curve fit,
made from the small sizes only, predict the larger sizes we actually observed?

E2 extrapolates from 200 effective training rows to 500 (2.5x). We cannot observe 500 rows,
but we can hide the largest sizes: fit the same inverse power law (and its log-linear and
unweighted variants, E2 run.py's fit_and_bootstrap with the E2 config) on sizes 50-150
(x = 40..120) and predict x = 160 and 200; then fit on 50-200 and predict x = 200. The
subset scores are recomputed with E2's own score_subset (same seeds, so they equal the E2
runs 20261022-070248_e2_learning_curve and 20261022-070339_e2_learning_curve_qda_block).
Reported: prediction error (point - observed mean) and whether the 90 % bootstrap interval
and the envelope cover the observed mean.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext12 --high-mem -- research/r308_e2_backtest.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib_ext12
from ssc import data, paths
from ssc.tracking import Run

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--n-boot", type=int, default=None, help="smoke test only")
a = p.parse_args()

spec = importlib.util.spec_from_file_location("e2_run_r308", ROOT / "extensions/e2_learning_curve/run.py")
e2 = importlib.util.module_from_spec(spec)  # not in sys.modules: joblib pickles its functions by value
spec.loader.exec_module(e2)
e2.build_pipeline = lib_ext12.build_pipeline

VARIANTS = {"l1_all": ROOT / "extensions/e2_learning_curve/config.yaml",
            "qda_hv": ROOT / "research/r302_e2_qda_block.yaml"}
HYP = ("Fitted on the small sizes only, the E2 power law under-covers the larger observed sizes: its 90 % "
       "bootstrap interval misses the held-out means for the L1 baseline (the curve shape at small n is "
       "dominated by collapsed L1 fits), while the envelope of curve forms covers them; prediction errors "
       "grow with the extrapolation ratio, so the 2.5x extrapolation to 500 rows is less reliable still.")

ds = data.load(a.data)
X, y = ds.X.to_numpy(float), ds.y
rows, all_sub = [], []
for key, cfg_path in VARIANTS.items():
    cfg = e2.load_config(cfg_path, quick=False)
    if a.n_boot:
        cfg["fit"]["n_boot"] = a.n_boot
        cfg["fit"]["sensitivity"]["n_boot"] = a.n_boot
    sizes = cfg["sizes"]
    res = Parallel(n_jobs=a.n_jobs)(delayed(e2.score_subset)(n, i, cfg, X, y)
                                    for n in sizes for i in range(cfg["n_subsets"]))
    sub = pd.DataFrame([r["row"] for r in res])
    all_sub.append(sub.assign(model=key))
    observed = sub.groupby("size")["cv_auc"].mean()
    for cut in (3, 4):
        fit_sizes, held = sizes[:cut], sizes[cut:]
        c2 = copy.deepcopy(cfg)
        xs = [e2.x_of(n, cfg) for n in held]
        c2["fit"]["predict_at"] = xs
        c2["fit"]["target_n"] = max(xs)
        fit = e2.fit_and_bootstrap(sub[sub["size"].isin(fit_sizes)], c2)
        ext = fit["extrapolation"]
        lo_c, hi_c = fit["interval_cols"]
        for n, x in zip(held, xs):
            obs = float(observed.loc[n])
            for model in ext["model"].unique():
                r = ext[(ext["model"] == model) & (ext["x"] == x)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append({"variant": key, "fit_on_sizes": ",".join(map(str, fit_sizes)),
                             "max_fit_x": e2.x_of(fit_sizes[-1], cfg), "target_size": n, "target_x": x,
                             "extrapolation_ratio": x / e2.x_of(fit_sizes[-1], cfg), "curve": model,
                             "point": r["point"], "lo": r[lo_c], "hi": r[hi_c], "observed_mean": obs,
                             "error": r["point"] - obs if np.isfinite(r["point"]) else np.nan,
                             "covered": bool(r[lo_c] <= obs <= r[hi_c]),
                             "observed_std_over_subsets": float(sub[sub["size"] == n]["cv_auc"].std(ddof=1))})
bt = pd.DataFrame(rows)
subs = pd.concat(all_sub, ignore_index=True)

with Run("r308_e2_backtest", {"name": "r308_e2_backtest", "variants": {k: str(v.relative_to(ROOT)) for k, v in VARIANTS.items()},
                              "cuts": [3, 4]}, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    run.save_table("backtest", bt)
    run.save_table("subset_scores_recomputed", subs)
    tdir = paths.tables_dir(a.data)
    tdir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(tdir / "r308_e2_backtest.csv", index=False)
    show = bt[bt["curve"].isin(["power_law", "log_linear", "envelope"])]
    with open(tdir / "r308_e2_backtest.md", "w", encoding="utf-8") as f:
        f.write(f"Run {run.run_id}. Fit on the smaller sizes only, predict the held-out larger sizes "
                "(x = effective training rows; observed = mean CV AUC of the 10 subsets).\n\n")
        f.write(show[["variant", "fit_on_sizes", "target_size", "target_x", "extrapolation_ratio", "curve", "point",
                      "lo", "hi", "observed_mean", "error", "covered"]].to_markdown(index=False, floatfmt=".4f") + "\n")
    run.finish(extra_summary={"backtest": bt.to_dict(orient="records"),
                              "observed_means": subs.groupby(["model", "size"])["cv_auc"].mean().reset_index()
                              .to_dict(orient="records")})

print(show[["variant", "fit_on_sizes", "target_x", "curve", "point", "lo", "hi", "observed_mean", "error", "covered"]]
      .to_string(index=False))
