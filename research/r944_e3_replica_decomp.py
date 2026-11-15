"""R944 (verifier "ext") - where does E3's leaky-minus-honest gap come from, on average over samples?

E3 (run 20261022-063908_e3_leakage) explains the gap-vs-k_feat curve as leaky OPTIMISM: below k_feat = 7 a
"winner's curse among real features", above 7 noise features picked for their luck. On the real data the
only external check is ONE 250-row sample scored against GMM proxy labels (r321), and there leaky CV is
close to the proxy test AUC for k_feat <= 10 while honest CV is 0.02-0.05 below it, i.e. the small-k gap
looks like honest PESSIMISM. One sample cannot separate the two.

Here the parametric replica of research/lib_gen_replica.py (fitted to the real 20,000 x 150 matrix, as in
R941/R942) is the population. For each of n_draws fresh 250-row training samples (+ 20,000 test rows with
true labels) and every (k_feat, k_fold) cell, E3's two procedures are run unchanged (scaler + |corr|
top-k_feat + L2 logistic C = 0.1 liblinear, seed 0; stratified k-fold, split seeds 2026 + r) and decomposed:

    leaky_cv, honest_cv        E3's two CV estimates (mean fold AUC over n_repeats split seeds)
    test_final                 test AUC of the final model (selection + fit on all 250 rows) = what the LB sees
    test_leaky_fold            mean test AUC of the leaky CV's own fold models (columns from 250 rows, fit on n_tr)
    test_honest_fold           mean test AUC of the honest CV's own fold models (selection + fit on n_tr rows)

    leak optimism   = leaky_cv  - test_leaky_fold   (validation rows took part in the selection)
    honest bias     = honest_cv - test_honest_fold  (should be ~0: honest CV is unbiased for its own procedure)
    vs LB target    = leaky_cv - test_final, honest_cv - test_final

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vext --high-mem -- research/r944_e3_replica_decomp.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.features import abs_correlation, top_k_indices  # noqa: E402
from ssc.models import build_pipeline, predict_score  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n-draws", type=int, default=100)
p.add_argument("--n-test", type=int, default=20000)
p.add_argument("--n-repeats", type=int, default=3)
p.add_argument("--n-jobs", type=int, default=-1)
a = p.parse_args()

K_FEAT = [2, 5, 7, 10, 13, 20, 30, 50, 100, 150]
K_FOLD = [2, 5, 10]
BASE = {"scaler": "standard", "model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 0.0, "solver": "liblinear"}},
        "random_state": 0}
# E3 real-data 5-fold results (reports/tables/e3_grid.csv) for the calibration columns
E3_GAP5 = {2: 0.0560, 5: 0.0227, 7: 0.0008, 10: 0.0255, 13: 0.0445, 20: 0.0733, 30: 0.0900, 50: 0.0904, 100: 0.0444, 150: 0.0}
R321_PROXY = {2: 0.7756, 5: 0.8214, 7: 0.8517, 10: 0.8638, 13: 0.8413, 20: 0.8195, 30: 0.7831, 50: 0.8250, 100: 0.8052, 150: 0.8015}

HYP = ("E3 attributes its whole gap to leaky optimism (winner's curse among real features for k_feat < 7, lucky noise "
       "features above). Expectation on parametric replicas (100 fresh 250-row samples, E3 protocol unchanged): "
       "(i) honest CV is ~unbiased for its own fold models at every k_feat (|honest_cv - test_honest_fold| < 0.005); "
       "(ii) leak optimism (leaky_cv - test_leaky_fold) is positive for every k_feat < 150 but small (< 0.015) at "
       "k_feat = 2-7 and large (> 0.05) at k_feat 20-50, so the small-k gap is mostly the fact that the honest "
       "filter sees 200 instead of 250 rows (honest pessimism relative to the final model), not optimism; "
       "(iii) against the final 250-row model (the LB target), leaky CV is within +-0.015 at k_feat <= 7 and "
       "optimistic by > 0.03 at k_feat >= 20; (iv) the near-zero gap at k_feat = 7 is specific to the real sample "
       "(mean replica gap at 7 clearly > 0.001), and the replica gap curve has E3's shape (peak at 20-50, 0 at 150).")


def spec_with(selector: dict) -> dict:
    s = dict(BASE)
    s["selector"] = selector
    return s


def one(ps, d: int) -> list[dict]:
    rep = G.sample_parametric(ps, seed=94400 + d, n_total=250 + a.n_test, n_train=250, n_public=100)
    X, y, Xt, yt = rep.X_train, rep.y_train, rep.X_test, rep.y_test
    corr = abs_correlation(X, y, "pearson")
    srt = np.sort(corr)[::-1]
    out = []
    for kf in K_FEAT:
        cols = [int(c) for c in top_k_indices(corr, kf)]
        leaky_spec = spec_with({"name": "fixed_columns", "columns": cols})
        honest_spec = spec_with({"name": "corr_topk", "k": kf, "method": "pearson"})
        final = build_pipeline(honest_spec).fit(X, y)
        test_final = G.auc_fast(yt, predict_score(final, Xt))
        for K in K_FOLD:
            lv, hv, lt, ht, ov = [], [], [], [], []
            for r in range(a.n_repeats):
                skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=2026 + r)
                for tr, va in skf.split(X, y):
                    ml = build_pipeline(leaky_spec).fit(X[tr], y[tr])
                    mh = build_pipeline(honest_spec).fit(X[tr], y[tr])
                    lv.append(G.auc_fast(y[va], predict_score(ml, X[va])))
                    hv.append(G.auc_fast(y[va], predict_score(mh, X[va])))
                    lt.append(G.auc_fast(yt, predict_score(ml, Xt)))
                    ht.append(G.auc_fast(yt, predict_score(mh, Xt)))
                    sel = set(int(c) for c in top_k_indices(abs_correlation(X[tr], y[tr], "pearson"), kf))
                    ov.append(len(sel & set(cols)) / kf)
            out.append({"draw": d, "k_feat": kf, "k_fold": K, "leaky_cv": float(np.mean(lv)), "honest_cv": float(np.mean(hv)),
                        "test_final": test_final, "test_leaky_fold": float(np.mean(lt)),
                        "test_honest_fold": float(np.mean(ht)), "fold_overlap": float(np.mean(ov)),
                        "n_abs_corr_ge_0.3": int((corr >= 0.3).sum()), "corr_rank7": float(srt[6]),
                        "corr_rank8": float(srt[7])})
    return out


with Run("r944_e3_replica_decomp", {"k_feat": K_FEAT, "k_fold": K_FOLD, "n_draws": a.n_draws, "n_test": a.n_test,
                                    "n_repeats": a.n_repeats, "base": BASE, "split_seeds": "2026 + r (as E3)",
                                    "replica": "lib_gen_replica.sample_parametric(fit_parametric(X_all, y)), seed 94400+d",
                                    "model_code": "research/r944_e3_replica_decomp.py"},
         kind="analysis", source="raw", hypothesis=HYP) as run:
    X_all = unlabeled_features("raw")
    y_real = data.load("raw").y
    ps = G.fit_parametric(X_all, y_real)
    rows = Parallel(n_jobs=a.n_jobs)(delayed(one)(ps, d) for d in range(a.n_draws))
    df = pd.DataFrame([r for rr in rows for r in rr])
    df["gap"] = df["leaky_cv"] - df["honest_cv"]
    df["leak_optimism"] = df["leaky_cv"] - df["test_leaky_fold"]
    df["honest_bias"] = df["honest_cv"] - df["test_honest_fold"]
    df["leaky_minus_final"] = df["leaky_cv"] - df["test_final"]
    df["honest_minus_final"] = df["honest_cv"] - df["test_final"]
    run.save_table("per_draw", df)
    cols = ["gap", "leak_optimism", "honest_bias", "leaky_minus_final", "honest_minus_final",
            "leaky_cv", "honest_cv", "test_final", "test_leaky_fold", "test_honest_fold", "fold_overlap"]
    g = df.groupby(["k_fold", "k_feat"])
    tab = g[cols].mean().add_suffix("_mean")
    sem = (g[["gap", "leak_optimism", "honest_bias", "leaky_minus_final", "honest_minus_final"]].std()
           / np.sqrt(a.n_draws)).add_suffix("_sem")
    tab = tab.join(sem).join(g["gap"].std().rename("gap_sd_across_samples")).reset_index()
    tab["e3_real_gap_5fold"] = [E3_GAP5[k] if f == 5 else np.nan for f, k in zip(tab.k_fold, tab.k_feat)]
    tab["r321_real_proxy"] = [R321_PROXY[k] if f == 5 else np.nan for f, k in zip(tab.k_fold, tab.k_feat)]
    run.save_table("decomposition", tab)
    paths.tables_dir("raw").mkdir(parents=True, exist_ok=True)
    tab.to_csv(paths.tables_dir("raw") / "r944_e3_replica_decomp.csv", index=False)
    t5 = tab[tab.k_fold == 5].set_index("k_feat")
    d5 = df[df.k_fold == 5]
    extra = {
        "gap_5fold_mean": t5["gap_mean"].round(4).to_dict(),
        "leak_optimism_5fold_mean": t5["leak_optimism_mean"].round(4).to_dict(),
        "honest_bias_5fold_mean": t5["honest_bias_mean"].round(4).to_dict(),
        "leaky_minus_final_5fold_mean": t5["leaky_minus_final_mean"].round(4).to_dict(),
        "honest_minus_final_5fold_mean": t5["honest_minus_final_mean"].round(4).to_dict(),
        "gap_by_kfold_at_kfeat20": tab[tab.k_feat == 20].set_index("k_fold")["gap_mean"].round(4).to_dict(),
        "leak_opt_by_kfold_at_kfeat20": tab[tab.k_feat == 20].set_index("k_fold")["leak_optimism_mean"].round(4).to_dict(),
        "frac_draws_gap_k7_below_0.005": float((d5[d5.k_feat == 7]["gap"] < 0.005).mean()),
        "argmin_gap_kfeat_le_13_per_draw": d5[d5.k_feat <= 13].loc[d5[d5.k_feat <= 13].groupby("draw")["gap"].idxmin(), "k_feat"]
        .value_counts().sort_index().to_dict(),
        "n_abs_corr_ge_0.3_dist": d5[d5.k_feat == 2]["n_abs_corr_ge_0.3"].value_counts().sort_index().to_dict(),
    }
    run.finish(extra_summary=extra)
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string())
    print(extra)
    print("run:", run.run_id)
