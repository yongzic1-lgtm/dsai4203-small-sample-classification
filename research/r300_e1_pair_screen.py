"""R300 - screen neighbouring candidate pairs for extension E1 under the OFFICIAL CV.

E1 asks which validation scheme can be trusted to choose between two models that differ
by about 0.01 AUC. The transductive GMM is saturated at CV AUC 1.0 on the real data, so
E1 needs a realistic pair from the supervised track whose official-CV gap is about 0.01.
Every candidate below is one experiment run (official 5-fold x 20, seed 2026, identical
splits, test predictions stored); then every pre-declared neighbouring pair is compared
with ssc.validation.paired_compare.

Pre-registered selection rule (written before the results):
  * main E1 pair   = the neighbouring pair of the LINEAR regime (corr-filter / L1 / L2
                     logistic) whose |mean paired difference| is closest to 0.01, among
                     pairs with 0.005 <= |diff| <= 0.02;
  * second E1 pair = the QDA reg_param pair closest to 0.01 (ceiling regime), if any pair
                     has |diff| in [0.003, 0.02]; it shows how the MDD depends on the pair.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext12 --high-mem -- research/r300_e1_pair_screen.py --n-jobs -1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

import lib_ext12
from ssc import data, paths
from ssc.tracking import Run
from ssc.validation import Scheme, cross_validate, make_splits, paired_compare

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--repeats", type=int, default=None, help="smoke test only (breaks official CV)")
a = p.parse_args()

official = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))["official_cv"]
scheme = Scheme.from_dict(official)
if a.repeats:
    scheme.n_repeats = a.repeats
is_official = scheme.to_dict() == Scheme.from_dict(official).to_dict()
ds = data.load(a.data)
splits = make_splits(ds.y, scheme)


def L2(C):
    return {"name": "logreg", "params": {"C": C, "l1_ratio": 0.0, "solver": "liblinear"}}


def L1(C):
    return {"name": "logreg", "params": {"C": C, "l1_ratio": 1.0, "solver": "liblinear"}}


# name -> (pipeline spec, regime, description)
CANDIDATES = {
    "corr10_l2": ({"scaler": "standard", "selector": {"name": "corr_topk", "k": 10}, "model": L2(0.05)},
                  "linear", "in-fold |corr| top-10 + L2 logistic C=0.05"),
    "corr20_l2": ({"scaler": "standard", "selector": {"name": "corr_topk", "k": 20}, "model": L2(0.05)},
                  "linear", "in-fold |corr| top-20 + L2 logistic C=0.05"),
    "corr30_l2": ({"scaler": "standard", "selector": {"name": "corr_topk", "k": 30}, "model": L2(0.05)},
                  "linear", "in-fold |corr| top-30 + L2 logistic C=0.05"),
    "corr50_l2": ({"scaler": "standard", "selector": {"name": "corr_topk", "k": 50}, "model": L2(0.05)},
                  "linear", "in-fold |corr| top-50 + L2 logistic C=0.05"),
    "l2_all": ({"scaler": "standard", "model": L2(0.05)}, "linear", "L2 logistic C=0.05, all 150 features"),
    "l1_c005": ({"scaler": "standard", "model": L1(0.05)}, "linear", "L1 logistic C=0.05, all 150 features"),
    "l1_c01": ({"scaler": "standard", "model": L1(0.1)}, "linear",
               "L1 logistic C=0.1, all 150 features (= R003 sup_l1_all, reproducibility check)"),
    "l1_c02": ({"scaler": "standard", "model": L1(0.2)}, "linear", "L1 logistic C=0.2, all 150 features"),
    "block_l2_c01": ({"scaler": "standard", "selector": {"name": "fixed_columns", "columns": lib_ext12.BLOCK},
                      "model": L2(0.1)}, "linear", "L2 logistic C=0.1 on the 13-column block"),
    "block_l2_c1": ({"scaler": "standard", "selector": {"name": "fixed_columns", "columns": lib_ext12.BLOCK},
                     "model": L2(1.0)}, "linear", "L2 logistic C=1 on the 13-column block"),
    "qda_r01": ({"scaler": "none", "model": {"name": "subspace_qda", "params": {"reg_param": 0.1, "source": a.data}}},
                "qda", "QDA reg 0.1 on the block (= R003 sup_qda_hv, reproducibility check)"),
    "qda_r03": ({"scaler": "none", "model": {"name": "subspace_qda", "params": {"reg_param": 0.3, "source": a.data}}},
                "qda", "QDA reg 0.3 on the block"),
    "qda_r05": ({"scaler": "none", "model": {"name": "subspace_qda", "params": {"reg_param": 0.5, "source": a.data}}},
                "qda", "QDA reg 0.5 on the block"),
    "qda_r08": ({"scaler": "none", "model": {"name": "subspace_qda", "params": {"reg_param": 0.8, "source": a.data}}},
                "qda", "QDA reg 0.8 on the block"),
}
# neighbouring pairs (A, B): one hyper-parameter step apart
PAIRS = [("corr10_l2", "corr20_l2"), ("corr20_l2", "corr30_l2"), ("corr30_l2", "corr50_l2"),
         ("corr50_l2", "l2_all"), ("l1_c005", "l1_c01"), ("l1_c01", "l1_c02"), ("l2_all", "l1_c01"),
         ("block_l2_c01", "block_l2_c1"),
         ("qda_r01", "qda_r03"), ("qda_r03", "qda_r05"), ("qda_r05", "qda_r08")]

HYP = ("Neighbouring supervised candidates differ by about 0.005-0.02 official-CV AUC: corr top-k steps and "
       "L1 C steps in the 0.80-0.90 range, QDA reg_param steps by < 0.01 near the 0.996 ceiling. "
       "The pair closest to 0.01 becomes the A/B pair of extension E1 (pre-registered rule in the script).")

names = a.only or list(CANDIDATES)
rows, folds = [], {}
for name in names:
    spec, regime, desc = CANDIDATES[name]
    cfg = {"name": f"r300_{name}", "description": desc, "model_code": "research/r300_e1_pair_screen.py",
           "pipeline": spec, "cv": scheme.to_dict(), "test_pred": "fold_mean",
           "uses_unlabeled_test_features": regime == "qda"}
    with Run(f"r300_{name}", cfg, kind="experiment", source=a.data, author=a.author, hypothesis=HYP) as run:
        cv = cross_validate(lib_ext12.build_pipeline(spec), ds.X, ds.y, splits, scheme, X_test=ds.X_test,
                            n_jobs=a.n_jobs)
        s = run.save_cv(cv)
        n_const = int(cv.fold_scores["constant_pred"].sum())
        run.finish(extra_summary={"official_cv": is_official, "regime": regime, "description": desc,
                                  "n_constant_folds": n_const,
                                  "uses_unlabeled_test_features": regime == "qda"})
        folds[name] = cv.fold_scores
        rows.append({"candidate": name, "regime": regime, "run_id": run.run_id, "auc_mean": s["auc_mean"],
                     "auc_std_repeat": s["auc_std_repeat"], "auc_std_fold": s["auc_std_fold"],
                     "oof_auc_mean": s.get("oof_auc_mean"), "n_constant_folds": n_const, "description": desc})
        print(f"{name:13s} auc={s['auc_mean']:.4f} +/- {s['auc_std_repeat']:.4f} (repeat sd)  run={run.run_id}",
              flush=True)

res = pd.DataFrame(rows)
pc = []
for A, B in PAIRS:
    if A in folds and B in folds:
        c = paired_compare(folds[A], folds[B])
        ra = folds[A].groupby("repeat")["auc"].mean()
        rb = folds[B].groupby("repeat")["auc"].mean()
        regime = CANDIDATES[A][1]
        pc.append({"A": A, "B": B, "regime": regime, **c,
                   "abs_diff": abs(c["mean_diff"]), "dist_to_0.01": abs(abs(c["mean_diff"]) - 0.01),
                   "corr_repeat_means": float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else np.nan,
                   "std_repeat_A": float(ra.std(ddof=1)), "std_repeat_B": float(rb.std(ddof=1)),
                   "run_A": res.set_index("candidate").loc[A, "run_id"],
                   "run_B": res.set_index("candidate").loc[B, "run_id"]})
pcd = pd.DataFrame(pc)


def pick(regime, lo, hi):
    g = pcd[(pcd["regime"] == regime) & pcd["abs_diff"].between(lo, hi)] if len(pcd) else pcd
    return None if g.empty else g.sort_values("dist_to_0.01").iloc[0]


main = pick("linear", 0.005, 0.02)
second = pick("qda", 0.003, 0.02)
pcd["selected"] = ""
for tag, r in (("E1 main pair", main), ("E1 second pair", second)):
    if r is not None:
        pcd.loc[(pcd["A"] == r["A"]) & (pcd["B"] == r["B"]), "selected"] = tag

tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
res.to_csv(tables / "r300_candidates.csv", index=False)
pcd.to_csv(tables / "r300_pair_screen.csv", index=False)
with Run("r300_e1_pair_screen", {"name": "r300_e1_pair_screen", "candidates": list(names), "pairs": PAIRS,
                                 "cv": scheme.to_dict()},
         kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    run.save_table("candidates", res)
    run.save_table("pair_screen", pcd)
    run.finish(extra_summary={"official_cv": is_official, "main_pair": None if main is None else [main["A"], main["B"]],
                              "second_pair": None if second is None else [second["A"], second["B"]],
                              "pairs": pcd.to_dict(orient="records")})
print("\n" + res[["candidate", "auc_mean", "auc_std_repeat", "n_constant_folds", "run_id"]].to_string(index=False))
if len(pcd):
    print("\n" + pcd[["A", "B", "mean_diff", "se_split", "std_diff_repeat", "corr_repeat_means", "win_rate_fold",
                      "corrected_p", "selected"]].to_string(index=False))
print(f"\nmain pair: {None if main is None else (main['A'], main['B'])}; "
      f"second pair: {None if second is None else (second['A'], second['B'])}")
