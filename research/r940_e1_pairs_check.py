"""R940 (verifier "ext") - does E1's MDD conclusion hold beyond the one chosen pair?

E1 (run 20261022-064054_e1_validation_protocol) measured the split-seed noise of A - B for ONE
linear pair (in-fold |corr| top-30 vs top-50 + L2 C=0.05) whose CV gap under the official seed was
-0.0101 but only -0.0029 in the 5x50 reference. This check:

1. reproduces E1's corr30/corr50 numbers for 5-fold, 5x10 and 5x20 exactly (same split seeds:
   execution s uses seed 1 + 1000 s + r; the first 10 / first 1 repeats of each 5x20 execution are
   exactly E1's 5x10 / 5-fold executions) and the official-seed gap (seed 2026);
2. scores all 8 linear candidates of R300 on the same 50 x 5x20 splits, so that for all 28 pairs we
   get the "true" CV gap on these 250 rows (mean over 1000 repeats) and the split-seed std of the
   5x20 / 5x10 / 5x1 paired difference -> MDD with 2 sd (E1's definition, 50% power) and 2.8 sd
   (alpha 0.05 two-sided, 80% power), chi-square 95% CI, p_correct(0.01) = Phi(0.01 / sd).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vext --high-mem -- research/r940_e1_pairs_check.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import lib_ext12
from ssc import data, paths
from ssc.tracking import Run
from ssc.validation import Scheme, cross_validate, make_splits

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--n-exec", type=int, default=50)
p.add_argument("--n-rep", type=int, default=20)
a = p.parse_args()


def L2(C):
    return {"name": "logreg", "params": {"C": C, "l1_ratio": 0.0, "solver": "liblinear"}}


def L1(C):
    return {"name": "logreg", "params": {"C": C, "l1_ratio": 1.0, "solver": "liblinear"}}


MODELS = {  # identical to research/r300_e1_pair_screen.py (linear regime)
    "corr10_l2": {"scaler": "standard", "selector": {"name": "corr_topk", "k": 10}, "model": L2(0.05), "random_state": 0},
    "corr20_l2": {"scaler": "standard", "selector": {"name": "corr_topk", "k": 20}, "model": L2(0.05), "random_state": 0},
    "corr30_l2": {"scaler": "standard", "selector": {"name": "corr_topk", "k": 30}, "model": L2(0.05), "random_state": 0},
    "corr50_l2": {"scaler": "standard", "selector": {"name": "corr_topk", "k": 50}, "model": L2(0.05), "random_state": 0},
    "l2_all": {"scaler": "standard", "model": L2(0.05), "random_state": 0},
    "l1_c005": {"scaler": "standard", "model": L1(0.05), "random_state": 0},
    "l1_c01": {"scaler": "standard", "model": L1(0.1), "random_state": 0},
    "l1_c02": {"scaler": "standard", "model": L1(0.2), "random_state": 0},
}
HYP = ("E1's MDD for 5x20 (0.0061) was measured on a pair whose true CV gap is ~0.004 and whose models are highly "
       "correlated (corr of repeat means 0.76). Expectation: (i) E1's corr30/corr50 numbers reproduce exactly and the "
       "pooled 1000-repeat gap is about -0.0044, not -0.010 or -0.003; (ii) less similar linear pairs have a larger "
       "split-seed std of A - B, so for some pairs the 5x20 MDD (2 sd) approaches 0.01, and with the 80%-power "
       "definition (2.8 sd) 5x20 no longer resolves 0.01 with its whole CI.")

ds = data.load(a.data)
X, y = ds.X.to_numpy(float), ds.y


def one(name: str, s: int | None) -> pd.DataFrame:
    seed = 2026 if s is None else 1 + 1000 * s
    scheme = Scheme(kind="kfold", n_splits=5, n_repeats=a.n_rep, seed=seed)
    cv = cross_validate(lib_ext12.build_pipeline(MODELS[name]), X, y, make_splits(y, scheme), scheme,
                        test_pred="none", n_jobs=1)
    rep = cv.fold_scores.groupby("repeat")["auc"].mean().reset_index()
    first = cv.fold_scores[cv.fold_scores["repeat"] == 0]["auc"].mean()
    rep["model"], rep["execution"], rep["seed"], rep["first_rep_check"] = name, -1 if s is None else s, seed, first
    return rep


def sd_ci(sd, n, level=0.95):
    df = n - 1
    return (sd * np.sqrt(df / stats.chi2.ppf(1 - (1 - level) / 2, df)), sd * np.sqrt(df / stats.chi2.ppf((1 - level) / 2, df)))


with Run("r940_e1_pairs_check", {"models": MODELS, "n_exec": a.n_exec, "n_rep": a.n_rep, "seeds": "1 + 1000 s + r; official 2026",
                                 "model_code": "research/r940_e1_pairs_check.py"},
         kind="analysis", source=a.data, hypothesis=HYP) as run:
    tasks = [(m, s) for m in MODELS for s in list(range(a.n_exec)) + [None]]
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(m, s) for m, s in tasks)
    rep = pd.concat(res, ignore_index=True)  # one row per (model, execution, repeat): mean fold AUC of that repeat
    run.save_table("repeat_means", rep)
    off = rep[rep["execution"] == -1].pivot(index="repeat", columns="model", values="auc")
    ex = rep[rep["execution"] >= 0]
    W = {R: ex[ex["repeat"] < R].groupby(["execution", "model"])["auc"].mean().unstack() for R in (1, 10, a.n_rep)}
    allrep = ex.groupby(["model"])["auc"].mean()
    rows = []
    for A, B in itertools.combinations(MODELS, 2):
        d_all = (ex[ex.model == A].set_index(["execution", "repeat"])["auc"]
                 - ex[ex.model == B].set_index(["execution", "repeat"])["auc"])
        true_gap = float(d_all.mean())
        true_gap_se = float(d_all.groupby(level=0).mean().std(ddof=1) / np.sqrt(a.n_exec))  # executions are iid
        row = {"A": A, "B": B, "true_gap_1000rep": true_gap, "true_gap_se": true_gap_se,
               "official_seed_gap_5x20": float((off[A] - off[B]).mean())}
        for R, w in W.items():
            dd = (w[A] - w[B]).to_numpy()
            sd = float(dd.std(ddof=1))
            lo, hi = sd_ci(sd, len(dd))
            tag = f"5x{R}"
            row.update({f"{tag}_mean": float(dd.mean()), f"{tag}_sd": sd, f"{tag}_mdd2": 2 * sd, f"{tag}_mdd2_hi": 2 * hi,
                        f"{tag}_mdd28": 2.8 * sd, f"{tag}_mdd28_hi": 2.8 * hi,
                        f"{tag}_p_correct_0.01": float(stats.norm.cdf(0.01 / sd)),
                        f"{tag}_p_correct_0.01_at_sd_hi": float(stats.norm.cdf(0.01 / hi)),
                        f"{tag}_sign_agree_true": float((np.sign(dd) == np.sign(true_gap)).mean()),
                        f"{tag}_corr_AB": float(np.corrcoef(w[A], w[B])[0, 1]) if np.std(w[A]) > 0 else np.nan})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["abs_true_gap"] = tab["true_gap_1000rep"].abs()
    tab = tab.sort_values("abs_true_gap").reset_index(drop=True)
    run.save_table("pairs", tab)
    paths.tables_dir(a.data).mkdir(parents=True, exist_ok=True)
    tab.to_csv(paths.tables_dir(a.data) / "r940_e1_pairs.csv", index=False)
    T = f"5x{a.n_rep}"
    e1 = tab[(tab.A == "corr30_l2") & (tab.B == "corr50_l2")].iloc[0].to_dict()
    near = tab[(tab.abs_true_gap >= 0.005) & (tab.abs_true_gap <= 0.02)]
    summ = {
        "e1_pair": e1,
        "model_means_1000rep": allrep.to_dict(),
        "n_pairs": int(len(tab)),
        "pairs_5x20_mdd2_hi_below_0.01": int((tab[f"{T}_mdd2_hi"] < 0.01).sum()),
        "pairs_5x20_mdd28_hi_below_0.01": int((tab[f"{T}_mdd28_hi"] < 0.01).sum()),
        "pairs_5x10_mdd2_hi_below_0.01": int((tab["5x10_mdd2_hi"] < 0.01).sum()),
        "near_0.01_pairs": near[["A", "B", "true_gap_1000rep", f"{T}_sd", f"{T}_mdd2", f"{T}_mdd2_hi", f"{T}_mdd28_hi",
                                 "5x10_sd", "5x10_p_correct_0.01", "5x1_sd", f"{T}_corr_AB"]].to_dict(orient="records"),
        "max_5x20_sd": float(tab[f"{T}_sd"].max()), "min_5x20_sd": float(tab[f"{T}_sd"].min()),
    }
    run.finish(extra_summary=summ)
    pd.set_option("display.width", 250)
    print(tab[["A", "B", "true_gap_1000rep", "true_gap_se", "official_seed_gap_5x20", "5x1_sd", "5x10_sd", f"{T}_sd",
               f"{T}_mdd2", f"{T}_mdd2_hi", f"{T}_mdd28_hi", "5x10_mdd2_hi", "5x10_p_correct_0.01", f"{T}_corr_AB",
               f"{T}_sign_agree_true"]].round(5).to_string())
    print({k: v for k, v in summ.items() if k not in ("near_0.01_pairs", "e1_pair")})
    print("E1 pair:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e1.items()})
    print("run:", run.run_id)
