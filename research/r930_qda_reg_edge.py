"""R930 - verifier "cand": is r210_qda_block_nll's reg choice (0.001 = bottom of its grid in 100/100 folds)
biased by the grid edge, and would a smaller reg change the D009 conclusion?

Official-CV experiment runs (project.yaml official_cv, identical splits, test_pred = fold mean), strictly
supervised pipeline of r201/r210 (lib_sup: HighVarColumns(1.5) in-fold -> StandardScaler -> QDA):
  qda_block_nll_ext     nested reg by inner neg_log_loss over REG + [3e-4, 1e-4, 3e-5, 1e-5]
  qda_block_auc_ext     the same grid, inner roc_auc (tie-break check)
  qda_block_fix_<r>     fixed reg_param r in {0.003, 0.001, 3e-4, 1e-4} (no tuning: the CV-AUC / OOF log-loss
                        profile around the edge; post-hoc by construction, not honest estimates of a procedure)
  qda_block_fix_<r>_tol fixed reg r in {3e-5, 1e-5} with QDA tol 1e-8: sklearn's default rank check (tol 1e-4)
                        rejects reg < 1e-4 because the 13-column block has an exact linear dependency
Then one analysis run (r930_reg_edge_summary): paired comparison (ssc.validation.paired_compare) of every run with
TransductiveGMM (S003 model, recomputed on the same splits), plus the sign test of fold AUCs and the 95% interval of
the paired mean with the Nadeau-Bengio corrected SE (what CV can and cannot rule out).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vcand --high-mem -- research/r930_qda_reg_edge.py --n-jobs 8
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_sup  # noqa: E402
from lib_sup import Cand, HighVarColumns, Tuned, common_args, run_candidates, setup  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import TransductiveGMM  # noqa: E402
from ssc.validation import cross_validate, paired_compare  # noqa: E402

C = importlib.import_module("r931_cand_lib")
REG = C.REG
REG_EXT = REG + [3e-4, 1e-4, 3e-5, 1e-5]

H_EXT = ("Written before running. sklearn QDA's rank check (tol 1e-4) makes reg < 1e-4 fail on the rank-12 block, so "
         "the inner log-loss can only move to 3e-4 or 1e-4; prediction: it does (r931 smoke test on 10 folds chose "
         "1e-4/3e-4), outer AUC stays within 1e-4 of r210 (0.99998) and outer OOF log-loss changes by < 0.003. The "
         "AUC-tuned version keeps 0.001-0.01 (ties go to the first grid entry). Either way the CV comparison with the "
         "GMM stays non-significant because both are saturated; D009's 'indistinguishable in CV' is not driven by "
         "the grid edge.")
H_FIX = ("Profile of fixed reg values around the r210 grid edge (post-hoc, not an honest estimate of a procedure): "
         "prediction: CV AUC 0.9999-1.0000 for every reg in [1e-5, 3e-3]; OOF log-loss minimal near 1e-4-3e-4 and "
         "rising again at 1e-5 (the 3-10 extra noise columns of the in-fold block make tiny reg overfit).")


def block(est):
    return Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", est)])


def candidates():
    out = [
        Cand("qda_block_nll_ext", lambda d: Tuned(block(QDA()), {"model__reg_param": REG_EXT}, scoring="neg_log_loss", log_dir=d),
             "supervised-generative", "block(in-fold highvar)", False, len(REG_EXT),
             "QDA on in-fold block, nested reg (grid extended below the r210 edge) by inner log-loss", H_EXT,
             tuning=f"reg_param in {REG_EXT}, inner neg_log_loss"),
        Cand("qda_block_auc_ext", lambda d: Tuned(block(QDA()), {"model__reg_param": REG_EXT}, scoring="roc_auc", log_dir=d),
             "supervised-generative", "block(in-fold highvar)", False, len(REG_EXT),
             "QDA on in-fold block, nested reg (extended grid) by inner AUC", H_EXT,
             tuning=f"reg_param in {REG_EXT}, inner roc_auc"),
    ]
    for r in (0.003, 0.001, 3e-4, 1e-4):
        out.append(Cand(f"qda_block_fix_{r:g}", (lambda rr: (lambda d: Tuned(block(QDA(reg_param=rr)), None, log_dir=d)))(r),
                        "supervised-generative", "block(in-fold highvar)", False, 1,
                        f"QDA reg {r:g} (fixed, post-hoc profile) on the in-fold block", H_FIX, tuning="none (fixed)"))
    for r in (3e-5, 1e-5):
        out.append(Cand(f"qda_block_fix_{r:g}_tol", (lambda rr: (lambda d: Tuned(block(QDA(reg_param=rr, tol=1e-8)), None, log_dir=d)))(r),
                        "supervised-generative", "block(in-fold highvar)", False, 1,
                        f"QDA reg {r:g}, rank-check tol 1e-8 (fixed, post-hoc profile) on the in-fold block", H_FIX,
                        tuning="none (fixed)"))
    return out


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    res = run_candidates(candidates(), "r930", a.data, a.n_jobs, a.author, "research/r930_qda_reg_edge.py", a.only, a.max_folds)

    # ------------------------------------------------------------------ paired comparison summary
    scheme, ds, splits = setup(a.data)
    y = np.asarray(ds.y).astype(int)
    hyp = ("GMM fold AUC is 1.0 in 100/100 folds; every supervised QDA variant is <= GMM fold by fold, so the paired mean "
           "is <= 0 with a one-sided sign; the Nadeau-Bengio corrected test is non-significant for all (|t| small "
           "because the corrected SE is inflated by n_val/n_train = 0.25); the 95% interval of the paired mean "
           "excludes differences larger than ~5e-4, i.e. CV can only bound the gap, not show equality.")
    cfg = {"name": "r930_reg_edge_summary", "compared_runs": list(res.get("run_id", [])), "reference": "TransductiveGMM highvar K=2 (S003 model)"}
    with Run("r930_reg_edge_summary", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=hyp) as run:
        gmm = cross_validate(TransductiveGMM("highvar", 2, source=a.data), ds.X, ds.y, splits, scheme, X_test=None, n_jobs=1)
        rows = []
        exp_dir = paths.experiments_dir(a.data)
        for _, r in res.iterrows():
            if r.get("run_id") in (None, "FAILED") or not isinstance(r.get("run_id"), str):
                continue
            fs = pd.read_csv(exp_dir / r["run_id"] / "fold_scores.csv")
            pc = paired_compare(fs, gmm.fold_scores)
            d = (fs.sort_values(["repeat", "fold"])["auc"].to_numpy() - gmm.fold_scores.sort_values(["repeat", "fold"])["auc"].to_numpy())
            n_neg, n_pos = int((d < 0).sum()), int((d > 0).sum())
            sign_p = float(stats.binomtest(n_neg, n_neg + n_pos, 0.5).pvalue) if n_neg + n_pos else 1.0
            se_c = np.sqrt((1 / len(d) + 50 / 200) * d.var(ddof=1)) if d.var(ddof=1) > 0 else 0.0
            tq = stats.t.ppf(0.975, len(d) - 1)
            rows.append({"candidate": r["candidate"], "run_id": r["run_id"], "auc_mean": fs["auc"].mean(),
                         "n_folds_below_1": int((fs["auc"] < 1).sum()), "oof_logloss": r.get("oof_logloss"),
                         "chosen": r.get("chosen"), "diff_vs_gmm": pc["mean_diff"], "se_split": pc["se_split"],
                         "corrected_p": pc["corrected_p"], "folds_worse_than_gmm": n_neg, "folds_better": n_pos,
                         "sign_test_p": sign_p, "ci95_corrected_lo": pc["mean_diff"] - tq * se_c,
                         "ci95_corrected_hi": pc["mean_diff"] + tq * se_c})
        out = pd.DataFrame(rows)
        run.save_table("paired_vs_gmm", out)
        tables = paths.tables_dir(a.data)
        out.to_csv(tables / "r930_paired_vs_gmm.csv", index=False)
        (tables / "r930_paired_vs_gmm.md").write_text(out.to_markdown(index=False, floatfmt=".6g"), encoding="utf-8")
        print(out.to_string(index=False), flush=True)
        run.finish(extra_summary={"gmm_auc_mean": float(gmm.fold_scores["auc"].mean()),
                                  "gmm_folds_below_1": int((gmm.fold_scores["auc"] < 1).sum()),
                                  "n_compared": int(len(out))})
