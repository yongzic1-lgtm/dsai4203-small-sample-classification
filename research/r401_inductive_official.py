"""R401 - official-CV experiment runs of two NEW strictly inductive candidates (research round 2, "r2_inductive"),
so that either can be submitted if the replica comparison (research/r402 + r404) selects it for D010 switch 3.

Both pipelines see only the 200 training rows of each outer fold (block rule, scaling, tuning, prior); no test
feature is used anywhere (uses_unlabeled_test_features = false). Test prediction = mean of the 100 fold models.

  bqda_eb_trainonly_widekappa   r255_bqda_eb_trainonly with the EB kappa0 grid widened beyond R255's edge
                                (R255: kappa0 = 1.0 = the largest grid value; grid here 0.001 ... 300)
  ens_r201_bqda_trainonly       ProbAverage(r201_qda_block pipeline, r255_bqda_eb_trainonly pipeline), equal weights

Also writes one analysis run (r401_inductive_official_checks): EB choices per fold of both BQDA variants, and
the ensemble's test_pred against 0.5 * (r201 + r255_trainonly) recomputed with research/r400_inductive_lib
(sha16 of both members against the archived test_pred.npy).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s r2ind2 --high-mem -- research/r401_inductive_official.py --n-jobs 8
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L4 = importlib.import_module("r400_inductive_lib")
C = L4.C
import lib_sup  # noqa: E402
from lib_fm import BayesQDA, TrainHighVar  # noqa: E402
from lib_sup import HighVarColumns, Tuned  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.validation import cross_validate  # noqa: E402

p = lib_sup.common_args("R401 strictly inductive candidates")
a = p.parse_args()
src = a.data
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)
scheme, ds, splits = lib_sup.setup(src)
if a.max_folds:
    if src == "raw" and not os.environ.get("SSC_ARCHIVE_ROOT"):
        raise SystemExit("--max-folds is for smoke tests into a scratch archive only")
    splits = splits[: a.max_folds]
REG = C.REG


def r201_pipe():
    return Tuned(Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", QDA())]),
                 {"model__reg_param": REG})


def bqda_to(kappa_grid=None):
    kw = {} if kappa_grid is None else {"kappa_grid": tuple(kappa_grid)}
    return Pipeline([("block", TrainHighVar(1.5)), ("model", BayesQDA(**kw))])


CANDS = {
    "bqda_eb_trainonly_widekappa": (
        lambda: bqda_to(L4.WIDE_KAPPA),
        "Bayesian QDA (EB NIW prior, Student-t predictive), block + scaling from the fold's training rows only, "
        f"EB kappa0 grid {list(L4.WIDE_KAPPA)}",
        "Written before running. R255's EB chose kappa0 = 1.0, the largest grid value, in 100/100 folds (for the hv "
        "version R256 moved it to 3 in 95/100 when widened). Prediction: with the wide grid EB picks kappa0 = 3-10 in "
        "most folds, and CV AUC stays within 2 split-SE of r255_bqda_eb_trainonly (0.99903), i.e. 0.9988-0.9993. "
        "Refuted if EB stays at 1.0 or the AUC moves by more than 2 split-SE.", 9 * 11 * 3),
    "ens_r201_bqda_trainonly": (
        lambda: L4.ProbAverage([r201_pipe(), bqda_to()]),
        "Equal-weight average of P(class 1) of r201_qda_block (in-fold block, nested reg by inner AUC) and "
        "r255_bqda_eb_trainonly (in-fold block, EB Bayesian QDA); both fitted on the same training fold",
        "Written before running. Averaging r201 (CV 0.99990) and r255_bqda_eb_trainonly (0.99903) gives a CV AUC "
        "between the two and within 2 split-SE of r201, about 0.9996-0.9999. Its test_pred equals 0.5 * (r201 + "
        "r255_trainonly) of the archived runs up to floating-point rounding (< 1e-15), because the members are "
        "fitted exactly as in those runs.", len(REG) + 132),
}

rows, tp = [], {}
for name, (make, desc, hyp, ncfg) in CANDS.items():
    if a.only and name not in a.only:
        continue
    cfg = {"name": f"r401_{name}", "description": desc, "model_code": "research/r401_inductive_official.py + "
           "research/r400_inductive_lib.py + research/lib_fm.py + research/lib_sup.py", "group": "supervised-generative",
           "features": "block(in-fold highvar)", "n_configs": ncfg, "cv": scheme.to_dict(), "test_pred": "fold_mean",
           "uses_unlabeled_test_features": False, "wide_kappa_grid": list(L4.WIDE_KAPPA) if "widekappa" in name else None,
           "members": ["20261022-071234_r201_qda_block", "20261022-070202_r255_bqda_eb_trainonly"] if name.startswith("ens") else None}
    t0 = time.time()
    with Run(f"r401_{name}", cfg, kind="experiment", source=src, author=a.author, hypothesis=hyp) as run:
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
        s = run.save_cv(cv)
        extra = {"official_cv": not a.max_folds, "uses_unlabeled_test_features": False, "n_configs": ncfg,
                 "wall_seconds": round(time.time() - t0, 1), **lib_sup.prob_metrics(cv.oof, ds.y),
                 "test_pred_sha16": L4.sha16(cv.test_pred), "n_folds_below_1": int((cv.fold_scores.auc < 1).sum())}
        s = run.finish(extra_summary=extra)
        tp[name] = cv.test_pred
        rows.append({"candidate": f"r401_{name}", "run_id": run.run_id, "auc_mean": s["auc_mean"],
                     "auc_std_repeat": s["auc_std_repeat"], "oof_logloss": s.get("oof_logloss"),
                     "n_folds_below_1": extra["n_folds_below_1"], "uses_unlabeled_test_features": False,
                     "description": desc})
        print(f"{name}: auc={s['auc_mean']:.6f} +/- {s['auc_std_repeat']:.6f} logloss={s.get('oof_logloss')} "
              f"folds<1={extra['n_folds_below_1']} run={run.run_id}", flush=True)

res = pd.DataFrame(rows)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
res.to_csv(tables / "r401_candidates.csv", index=False)
print(res.to_string(index=False), flush=True)

# ---- checks: EB choices per fold; ensemble = average of the archived members
cfg = {"name": "r401_inductive_official_checks", "inputs": "real train rows (labels), official splits",
       "code": "research/r400_inductive_lib.py"}
hyp = ("Written before running. (1) Default-grid EB of r255_bqda_eb_trainonly picks kappa0 = 1.0 (grid edge) in "
       "100/100 folds as archived; the wide grid moves it to 3-10. (2) The re-implementation in r400_inductive_lib "
       "reproduces the archived test_pred of r201_qda_block and r255_bqda_eb_trainonly bit-identically, and the "
       "official ensemble run's test_pred equals their average to < 1e-15.")
with Run("r401_inductive_official_checks", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    X, Xt, y = ds.X.to_numpy(float), ds.X_test.to_numpy(float), np.asarray(ds.y).astype(int)
    b_def, ch_def = L4.bqda_trainonly_scores(X, Xt, y, splits)
    b_wk, ch_wk = L4.bqda_trainonly_scores(X, Xt, y, splits, kappa_grid=L4.WIDE_KAPPA)
    nq, chosen = C.nested_qda_scores(X, Xt, y, splits)
    ch = pd.DataFrame([{"split": i, "repeat": s.repeat, "fold": s.fold, "kappa0_default": c0[0], "nu0_extra_default": c0[1],
                        "prior_default": c0[2], "kappa0_wide": c1[0], "nu0_extra_wide": c1[1], "prior_wide": c1[2],
                        "block_size": c0[3]} for i, (s, c0, c1) in enumerate(zip(splits, ch_def, ch_wk))])
    run.save_table("eb_choice_per_fold", ch)
    freq = (ch.groupby(["kappa0_default", "kappa0_wide"]).size().rename("n_folds").reset_index())
    run.save_table("eb_kappa_frequency", freq)
    freq.to_csv(tables / "r401_eb_kappa_frequency.csv", index=False)
    out = {"eb_default": dict(Counter(f"{c[0]:g}|{c[1]:g}|{c[2]}" for c in ch_def)),
           "eb_wide": dict(Counter(f"{c[0]:g}|{c[1]:g}|{c[2]}" for c in ch_wk)),
           "block_size": dict(Counter(int(c[3]) for c in ch_def)),
           "sha16_r201_recomputed": L4.sha16(nq["r201_qda_block"]),
           "r201_bit_identical": L4.sha16(nq["r201_qda_block"]) == L4.ARCHIVED_SHA16["r201_qda_block"],
           "sha16_bqda_trainonly_recomputed": L4.sha16(b_def),
           "bqda_trainonly_bit_identical": L4.sha16(b_def) == L4.ARCHIVED_SHA16["r255_bqda_eb_trainonly"],
           "sha16_bqda_wide_recomputed": L4.sha16(b_wk)}
    if "bqda_eb_trainonly_widekappa" in tp:
        out["wide_official_equals_recomputed_max_abs"] = float(np.max(np.abs(tp["bqda_eb_trainonly_widekappa"] - b_wk)))
    if "ens_r201_bqda_trainonly" in tp:
        ref = 0.5 * (nq["r201_qda_block"] + b_def)
        out["ens_official_minus_member_average_max_abs"] = float(np.max(np.abs(tp["ens_r201_bqda_trainonly"] - ref)))
        out["ens_r8_values_differing"] = int(sum("%.8f" % u != "%.8f" % v for u, v in zip(tp["ens_r201_bqda_trainonly"], ref)))
    print(out, flush=True)
    run.finish(extra_summary=out)
