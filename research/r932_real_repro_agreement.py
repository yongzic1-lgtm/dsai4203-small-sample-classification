"""R932 - real data: re-run the 8 final-candidate procedures (research/r931_cand_lib.py) on the competition data
and measure how differently their test rankings can score on the private LB (verifier "cand").

Part A (reproduction): every procedure of r931_cand_lib is run on the real train+test matrix with the official
splits (project.yaml official_cv); the test predictions are saved as test_pred_<proc>.npy in this run folder so
they can be compared bit-for-bit with the archived test_pred.npy of each candidate's run (done on the laptop
after the merge; pure file comparison). The nested-QDA reg choices per fold are saved for comparison with the
archived fold_fit_info.csv of r201 / r210. Also produces x_qda_block_nll_ext (r210 with the reg grid extended
below its edge).

Part B (agreement / hedge potential on the REAL test rows):
  * Spearman / Kendall of every pair on the S003 boundary rows (S003 score in (0.1, 0.9)),
  * number of test rows whose rank differs, number of discordant pairs (Kendall distance) over all 19,750 rows,
  * plug-in distribution of the private-AUC difference vs S003: labels drawn as Bernoulli(p_ref) with
    p_ref = (1 - f) pi + f / 2 (f = flip_y), pi = a reference posterior, private part = 19,355 random test rows;
    reported for several references (converged GMM MLE posterior = close to the generator truth, and each
    candidate's own score as if it were the truth), so the circularity is visible.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vcand --high-mem -- research/r932_real_repro_agreement.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("r931_cand_lib")
import lib_gen_replica as G  # noqa: E402
import lib_mix as L  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
from ssc.validation import Scheme, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--n-label-draws", type=int, default=2000)
p.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.0044, 0.01])
a = p.parse_args()

HYP = ("(A) r931_cand_lib reproduces every archived candidate's test_pred.npy (max |diff| < 1e-9 for the GMM "
       "family, < 1e-6 for the others) and the archived nested reg choices of r201/r210 in 100/100 folds. "
       "(B) The 4 GMM-family candidates (S003, r105, r155, r102) order the test rows almost identically (Spearman "
       "> 0.99 on the S003 boundary rows); BQDA and self-training differ more (0.7-0.95) and strictly supervised "
       "QDA most (0.4-0.6, R249 reported 0.45-0.49). Under a plug-in truth the private-AUC differences vs S003 "
       "are of order 1e-5 to 1e-4 for the supervised ones and < 1e-6 within the GMM family, i.e. far smaller "
       "than the label-noise spread of the private AUC itself.")

scheme = Scheme.from_dict(yaml.safe_load(open(paths.PROJECT_YAML, encoding="utf-8"))["official_cv"])
ds = data.load(a.data)
y = np.asarray(ds.y).astype(int)
Xtr = ds.X.to_numpy(float)
Xte = ds.X_test.to_numpy(float)
X_all = unlabeled_features(a.data)
n_tr = len(y)
assert np.allclose(X_all[:n_tr], Xtr) and np.allclose(X_all[n_tr:], Xte)
tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)

cfg = {"name": "r932_real_repro_agreement", "procedures": C.PROCS, "source_runs": C.RUN_IDS,
       "cv": scheme.to_dict(), "test_pred": "fold_mean", "reg_grid_ext": C.REG_EXT,
       "n_label_draws": a.n_label_draws, "flips": a.flips, "code": "research/r931_cand_lib.py"}


def one_proc(name):
    t = time.perf_counter()
    if name == "nested_sklearn_metrics":   # same nested QDA with sklearn's own roc_auc_score / log_loss
        splits = make_splits(y, scheme)
        sc, ch = C.nested_qda_scores(Xtr, Xte, y, splits, exact_metrics=True)
        return name, {f"{k}__sklearn_metrics": v for k, v in sc.items()}, {"chosen_reg": ch, "total_seconds": time.perf_counter() - t}
    lf = C.LabelFree(X_all, n_tr, need=[name])
    sc, info = C.all_scores(lf, Xtr, Xte, y, scheme, procs=[name] if not name.endswith(("qda_block", "_nll", "_ext"))
                            else ["r201_qda_block", "r210_qda_block_nll", "x_qda_block_nll_ext"])
    info["label_free_seconds"] = lf.seconds
    info["total_seconds"] = time.perf_counter() - t
    if hasattr(lf, "inf_cols"):
        info["informative_cols"] = [int(c) for c in lf.inf_cols]
    info["highvar_cols"] = [int(c) for c in lf.hv]
    return name, sc, info


with Run("r932_real_repro_agreement", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    jobs = [n for n in C.PROCS if n not in ("r201_qda_block", "x_qda_block_nll_ext")]   # r210 job computes all three
    jobs.append("nested_sklearn_metrics")
    res = Parallel(n_jobs=min(a.n_jobs, len(jobs)))(delayed(one_proc)(n) for n in jobs)
    preds, infos, sk = {}, {}, {}
    for name, sc, info in res:
        if name == "nested_sklearn_metrics":
            sk = sc
        else:
            preds.update(sc)
        infos[name] = info
    for k in C.PROCS:
        run.save_array(f"test_pred_{k}", preds[k])
    ch = infos["r210_qda_block_nll"]["chosen_reg"]
    ch_sk = infos["nested_sklearn_metrics"]["chosen_reg"]
    splits = make_splits(y, scheme)
    run.save_table("nested_qda_chosen_reg", pd.DataFrame(
        {"repeat": [s.repeat for s in splits], "fold": [s.fold for s in splits], **{k: v for k, v in ch.items()},
         **{f"{k}__sklearn_metrics": v for k, v in ch_sk.items()}}))
    metric_copy_check = {k: {"same_choice_folds": int(sum(u == v for u, v in zip(ch[k], ch_sk[k]))),
                             "max_abs_test_diff": float(np.abs(preds[k] - sk[f"{k}__sklearn_metrics"]).max())}
                         for k in ch}
    print("fast metric copies vs sklearn metrics:", metric_copy_check, flush=True)
    timing = {k: {kk: vv for kk, vv in v.items() if kk != "chosen_reg"} for k, v in infos.items()}
    print(json.dumps(timing, indent=1, default=str), flush=True)

    # ---------------------------------------------------------------- B: agreement on the real test rows
    s3 = preds["s003_gmm2_hv"]
    bnd = (s3 > 0.1) & (s3 < 0.9)
    names = C.PROCS
    rows = []
    for i, ni in enumerate(names):
        for nj in names[i + 1:]:
            a_, b_ = preds[ni], preds[nj]
            kt_all = kendalltau(a_, b_)
            n = len(a_)
            # discordant pairs from tau-b is awkward with ties; use the label-free 'pairs that change order' among
            # the union of uncertain rows of either score (exact O(m^2))
            u = ((a_ > 0.01) & (a_ < 0.99)) | ((b_ > 0.01) & (b_ < 0.99)) | bnd
            ia, ib = np.triu_indices(int(u.sum()), 1)
            ra, rb = a_[u], b_[u]
            disc = int(((np.sign(ra[ia] - ra[ib]) * np.sign(rb[ia] - rb[ib])) < 0).sum())
            rows.append({"a": ni, "b": nj, "spearman_s003_boundary": float(spearmanr(a_[bnd], b_[bnd]).statistic),
                         "kendall_s003_boundary": float(kendalltau(a_[bnd], b_[bnd]).statistic),
                         "kendall_all": float(kt_all.statistic), "n_uncertain_union": int(u.sum()),
                         "discordant_pairs_uncertain_union": disc,
                         "max_abs_diff": float(np.abs(a_ - b_).max()),
                         "n_rank_changed": int((pd.Series(a_).rank().to_numpy() != pd.Series(b_).rank().to_numpy()).sum())})
    agr = pd.DataFrame(rows)
    run.save_table("pairwise_agreement", agr)
    agr.to_csv(tables / "r932_pairwise_agreement.csv", index=False)
    print(agr.to_string(index=False), flush=True)

    # ---------------------------------------------------------------- B: plug-in private-AUC difference distribution
    mle_post = L.oriented(L.gmm_fit_arr(L.basis_arr(X_all, tuple(int(c) for c in G.highvar_columns(X_all))),
                                        2, "full", 5, 0, 1e-9)["post"], y)[n_tr:]
    refs = {"gmm_mle_posterior": mle_post, "bqda_score": preds["r255_bqda_eb_hv"],
            "selftrain_score": preds["r209_selftrain_qda_lw"], "qda_nll_score": preds["r210_qda_block_nll"]}
    n_te = len(s3)

    def plugin(f, rname, pi, seed):
        rng = np.random.default_rng(seed)
        pr = (1 - f) * np.clip(pi, 0, 1) + f / 2
        aucs = {k: C.SubsetAUC(preds[k]) for k in names}
        A = np.empty((a.n_label_draws, len(names)))
        for t in range(a.n_label_draws):
            yl = (rng.random(n_te) < pr).astype(int)
            m = np.ones(n_te, bool)
            m[rng.choice(n_te, G.N_PUBLIC, replace=False)] = False
            A[t] = [aucs[k](yl, m) for k in names]
        res_ = []
        for j, k in enumerate(names):
            dd = A[:, j] - A[:, 0]
            res_.append({"flip_y": f, "reference": rname, "proc": k, "private_auc_mean": float(A[:, j].mean()),
                         "private_auc_sd": float(A[:, j].std(ddof=1)), "diff_vs_s003_mean": float(dd.mean()),
                         "diff_vs_s003_sd": float(dd.std(ddof=1)), "diff_mc_se": float(dd.std(ddof=1) / np.sqrt(len(dd))),
                         "p_beats_s003": float((dd > 0).mean()), "p_ties_s003": float((dd == 0).mean()),
                         "expected_auc_under_ref": float(L.expected_auc(preds[k], pr)),
                         "expected_gain_max_with_s003": float(np.maximum(dd, 0).mean()),
                         "corr_private_auc_with_s003": float(np.corrcoef(A[:, j], A[:, 0])[0, 1]) if j else 1.0})
        return res_

    assert names[0] == "s003_gmm2_hv"
    combos = [(f, rn, pi, 1000 * i + j) for i, f in enumerate(a.flips) for j, (rn, pi) in enumerate(refs.items())]
    out = sum(Parallel(n_jobs=a.n_jobs)(delayed(plugin)(*c) for c in combos), [])
    plug = pd.DataFrame(out)
    run.save_table("plugin_private_diff", plug)
    plug.to_csv(tables / "r932_plugin_private_diff.csv", index=False)
    print(plug.to_string(index=False), flush=True)
    run.finish(extra_summary={"timing": timing, "n_s003_boundary": int(bnd.sum()), "metric_copy_check": metric_copy_check,
                              "informative_cols_r155": infos["r155_gmm2_inf_tight"].get("informative_cols"),
                              "highvar_cols": infos["s003_gmm2_hv"]["highvar_cols"],
                              "chosen_reg_counts": {k: pd.Series(v).value_counts().to_dict() for k, v in ch.items()}})
