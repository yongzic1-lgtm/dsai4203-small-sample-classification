"""R941 (verifier "ext") - the TRUE learning curve on parametric replicas of the real data.

E2 extrapolates the L1 baseline to 500 training rows (power law 0.906, 90% [0.881, 0.916], envelope
[0.857, 0.974]) and QDA-on-block to 0.9969 [0.9952, 0.9986], from subsets of ONE 250-row sample.
Here the parametric replica of research/lib_gen_replica.py (per-class Gaussians fitted label-free to
the real 20,000 x 150 matrix, same column roles and scales) is used as a stand-in population:
for each draw, a fresh 500-row training set and 20,000 fresh test rows are sampled; models are
trained on stratified nested subsets of t = 40 ... 500 rows and scored on the test rows.

Outputs: the expected test AUC (and its spread across training samples = parent-sample noise) per
training size for
    l1_all     E2 (a): scaler + L1 logistic C=0.1 (liblinear), all 150 features
    qda_hv     E2 (b): QDA reg 0.1 on the label-free high-variance block (train+test standardisation)
    corr30_l2 / corr50_l2  E1's pair (in-fold |corr| top-30 / top-50 + L2 C=0.05), population gap at t=200, 250
and a calibration check against E2's observed CV means at x = 40..200.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vext --high-mem -- research/r941_e2_replica_truth.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.models import build_pipeline, predict_score  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n-draws", type=int, default=100)
p.add_argument("--n-test", type=int, default=20000)
p.add_argument("--n-jobs", type=int, default=-1)
a = p.parse_args()

SIZES = [40, 80, 120, 160, 200, 250, 300, 400, 500]
L1 = {"scaler": "standard", "model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 1.0, "solver": "liblinear"}},
      "random_state": 0}


def corr_l2(k):
    return {"scaler": "standard", "selector": {"name": "corr_topk", "k": k},
            "model": {"name": "logreg", "params": {"C": 0.05, "l1_ratio": 0.0, "solver": "liblinear"}}, "random_state": 0}


HYP = ("If the parametric replica is a faithful population model, its expected test AUC at t = 40..200 training rows "
       "matches E2's observed CV means (L1: 0.659, 0.735, 0.810, 0.830, 0.847; QDA: 0.973 ... 0.996) within their SE. "
       "Then the replica's expected AUC at t = 500 is the target of E2's extrapolation. Expectation: L1 truth at 500 lies "
       "inside E2's envelope [0.857, 0.974] but possibly below the power-law point 0.906 (a fixed-C linear model has a "
       "ceiling well below 1, while the fitted asymptote is a = 0.996); the spread of the AUC across fresh 200-row "
       "samples is larger than E2's within-sample interval, i.e. E2's intervals are too narrow as population statements.")


def strat_nested(y, rng):
    """A random order of the rows such that every prefix is (nearly) stratified."""
    idx1, idx0 = rng.permutation(np.flatnonzero(y == 1)), rng.permutation(np.flatnonzero(y == 0))
    p1 = len(idx1) / len(y)
    order, i1, i0 = [], 0, 0
    for t in range(1, len(y) + 1):
        if i1 < len(idx1) and (i1 + 1) <= round(p1 * t + 1e-9) or i0 >= len(idx0):
            order.append(idx1[i1]); i1 += 1
        else:
            order.append(idx0[i0]); i0 += 1
    return np.array(order)


def one(ps, d: int) -> list[dict]:
    rep = G.sample_parametric(ps, seed=50000 + d, n_total=500 + a.n_test, n_train=500, n_public=100)
    X_all = rep.X_all
    rng = np.random.default_rng([941, d])
    order = strat_nested(rep.y_train, rng)
    qda = G.make_supervised("qda_hv", X_all)
    out = []
    for t in SIZES:
        idx = order[:t]
        Xt, yt = rep.X_train[idx], rep.y_train[idx]
        models = {"l1_all": build_pipeline(L1), "qda_hv": qda}
        if t in (200, 250):
            models.update({"corr30_l2": build_pipeline(corr_l2(30)), "corr50_l2": build_pipeline(corr_l2(50))})
        for name, est in models.items():
            from sklearn.base import clone
            m = clone(est).fit(Xt, yt)
            s = predict_score(m, rep.X_test)
            out.append({"draw": d, "t": t, "model": name, "n_pos": int(yt.sum()),
                        "test_auc": G.auc_fast(rep.y_test, s),
                        "constant": bool(np.ptp(s) < 1e-12)})
    return out


with Run("r941_e2_replica_truth", {"sizes": SIZES, "n_draws": a.n_draws, "n_test": a.n_test, "l1": L1,
                                   "replica": "lib_gen_replica.sample_parametric(fit_parametric(X_all, y)), seed 50000+d",
                                   "model_code": "research/r941_e2_replica_truth.py"},
         kind="analysis", source="raw", hypothesis=HYP) as run:
    X_all = unlabeled_features("raw")
    y = data.load("raw").y
    ps = G.fit_parametric(X_all, y)
    rows = Parallel(n_jobs=a.n_jobs)(delayed(one)(ps, d) for d in range(a.n_draws))
    df = pd.DataFrame([r for rr in rows for r in rr])
    run.save_table("replica_test_auc", df)
    g = df.groupby(["model", "t"])["test_auc"]
    tab = g.agg(mean="mean", std="std", q05=lambda s: s.quantile(0.05), q95=lambda s: s.quantile(0.95),
                n="size").reset_index()
    tab["sem"] = tab["std"] / np.sqrt(tab["n"])
    tab["frac_constant"] = df.groupby(["model", "t"])["constant"].mean().to_numpy()
    obs = {"l1_all": {40: 0.6587, 80: 0.7346, 120: 0.8104, 160: 0.8301, 200: 0.8474},
           "qda_hv": {40: 0.9729, 80: 0.9903, 120: 0.9923, 160: 0.9950, 200: 0.9960}}
    tab["e2_observed_cv_mean"] = [obs.get(m, {}).get(t, np.nan) for m, t in zip(tab["model"], tab["t"])]
    run.save_table("truth_by_size", tab)
    paths.tables_dir("raw").mkdir(parents=True, exist_ok=True)
    tab.to_csv(paths.tables_dir("raw") / "r941_e2_replica_truth.csv", index=False)
    # E1 pair: population gap corr30 - corr50 per training sample
    w = df[df.model.isin(["corr30_l2", "corr50_l2"])].pivot_table(index=["draw", "t"], columns="model", values="test_auc")
    gap = (w["corr30_l2"] - w["corr50_l2"]).groupby(level="t").agg(["mean", "std", "count"])
    gap["sem"] = gap["std"] / np.sqrt(gap["count"])
    run.save_table("e1_pair_population_gap", gap.reset_index())
    e2 = {"l1_all": {"point": 0.9057, "lo": 0.8806, "hi": 0.9158, "env_lo": 0.8567, "env_hi": 0.9738},
          "qda_hv": {"point": 0.9969, "lo": 0.9952, "hi": 0.9986, "env_lo": 0.9952, "env_hi": 1.0081}}
    check = {}
    for m, v in e2.items():
        r = tab[(tab.model == m) & (tab.t == 500)].iloc[0]
        check[m] = {"truth_500_mean": float(r["mean"]), "truth_500_sem": float(r["sem"]), "truth_500_std": float(r["std"]),
                    "e2_point": v["point"], "e2_point_minus_truth": v["point"] - float(r["mean"]),
                    "truth_in_e2_90": bool(v["lo"] <= r["mean"] <= v["hi"]),
                    "truth_in_e2_envelope": bool(v["env_lo"] <= r["mean"] <= v["env_hi"])}
    run.finish(extra_summary={"check_500": check, "e1_pair_gap": gap.reset_index().to_dict(orient="records")})
    pd.set_option("display.width", 220)
    print(tab.round(4).to_string())
    print(gap.round(5))
    print(check)
    print("run:", run.run_id)
