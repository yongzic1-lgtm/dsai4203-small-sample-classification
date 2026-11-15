"""R900 - label-permutation null of the top pipelines under the OFFICIAL CV (verifier `leak`).

Question: can label information from outside the training fold reach a validation prediction in any
of the pipelines behind our best CV numbers? Two nulls, both on the official scheme
(project.yaml official_cv, 5-fold x 20, seed 2026), each with n_perm label permutations
y_p = default_rng(perm_seed + p).permutation(y):

  Null A (classic permutation test): train AND evaluate on y_p, splits = make_splits(y_p, scheme).
          Detects leakage through the label vector handed to the CV (e.g. a selector fitted on all
          rows): an honest pipeline gives mean AUC ~0.5; the positive control `ctrl_e3_leaky`
          (E3's leaky procedure: top-20 |corr| columns chosen on ALL rows) must stay well above 0.5.
  Null B (cross null): train on y_p, evaluate against the TRUE labels, official splits of the true
          labels (split signature 5708ce5c3828). Detects leakage of the TRUE labels through any other
          channel (an estimator reading train.csv's target, a cached object built with labels, ...),
          which Null A cannot see because there the evaluation labels are themselves permuted.
          An honest pipeline is symmetric around 0.5 (its learned direction / component orientation
          is random); the positive control `ctrl_disk_reader` (ignores the labels it is given and
          fits QDA on all 250 rows with the true labels from disk) must stay near 1.

Pipelines (exact re-constructions of the archived runs; `real` rows re-run them on the true labels
and are compared with the archived CV AUC):
  s003_gmm2_hv           ssc.transductive.TransductiveGMM("highvar", 2)          = R003 gmm2_hv / S003
  r155_gmm2_inf_tight    lib_gen_replica.TransductiveGMMTight(columns="informative")
  r201_qda_block         Tuned(HighVarColumns -> StandardScaler -> QDA, reg grid, inner AUC)
  r210_qda_block_nll     same, inner neg_log_loss
  r209_selftrain_qda_lw  Tuned(PoolSelfTraining(QDA Ledoit-Wolf, threshold 0.99)), 20,000 rows
  r255_bqda_eb_hv        SubspaceModel(BayesQDA(), "highvar")  (empirical-Bayes NIW prior per fold)
  e3_honest              StandardScaler -> CorrelationTopK(20) -> L2 logreg C=0.1 (E3 honest / final model)
  ctrl_e3_leaky          positive control, path (a): E3 leaky procedure (FixedColumns from all rows)
  ctrl_disk_reader       positive control, path (b): reads the true labels from disk
  ctrl_random            chance floor: random scores (no model)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vleak --high-mem -- research/r900_perm_null.py --null A
    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vleak --high-mem -- research/r900_perm_null.py --null B --skip-real
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from lib_fm import BayesQDA  # noqa: E402
from lib_sup import HighVarColumns, PoolSelfTraining, Tuned  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.data import FEATURES  # noqa: E402
from ssc.features import CorrelationTopK  # noqa: E402
from ssc.models import build_pipeline  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import SubspaceModel, TransductiveGMM  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

REG = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001]  # = r201 / r210 grid
E3_BASE = {"scaler": "standard", "model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 0.0, "solver": "liblinear"}},
           "random_state": 0}  # = extensions/e3_leakage/config.yaml pipeline
E3_K = 20

# archived CV AUC on the true labels (summary.json auc_mean of the archived runs)
ARCHIVED = {
    "s003_gmm2_hv": ("20261022-050708_r003_gmm2_hv", 1.0),
    "r155_gmm2_inf_tight": ("20261022-080453_r155_gmm2_inf_tight", 1.0),
    "r201_qda_block": ("20261022-071234_r201_qda_block", 0.9999028588159024),
    "r210_qda_block_nll": ("20261022-082047_r210_qda_block_nll", 0.9999837662337663),
    "r209_selftrain_qda_lw": ("20261022-080632_r209_selftrain_qda_lw", 1.0),
    "r255_bqda_eb_hv": ("20261022-070200_r255_bqda_eb_hv", 1.0),
    "e3_honest": ("20261022-063908_e3_leakage (honest, k_feat 20, 5-fold)", 0.7934408526256352),
    "ctrl_e3_leaky": ("20261022-063908_e3_leakage (leaky, k_feat 20, 5-fold)", 0.8667809827048956),
}


class DiskReader(ClassifierMixin, BaseEstimator):
    """POSITIVE CONTROL for leakage path (b): ignores the labels passed to fit and fits QDA(reg 0.01)
    on the label-free block of ALL 250 training rows with the TRUE labels read from train.csv
    (validation rows included). A deliberately broken pipeline."""

    def __init__(self, source: str = "raw"):
        self.source = source

    def fit(self, X, y):
        tr = pd.read_csv(paths.data_dir(self.source) / "train.csv")  # the TRUE labels, read from disk
        tr.columns = [str(c) for c in tr.columns]
        Xt, yt = tr[FEATURES].to_numpy(dtype=float), tr["target"].to_numpy().astype(int)
        self.m_ = SubspaceModel(QDA(reg_param=0.01), "highvar", source=self.source).fit(Xt, yt)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


class RandomScore(ClassifierMixin, BaseEstimator):
    """Chance floor: uniform random scores, deterministic per (training set, scored rows)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.key_ = hashlib.sha1(np.ascontiguousarray(X, dtype=float).tobytes()).hexdigest()[:8]
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        h = hashlib.sha1((self.key_ + str(self.seed)).encode() + np.ascontiguousarray(X, dtype=float).tobytes())
        p = np.random.default_rng(int(h.hexdigest()[:12], 16)).random(len(X))
        return np.c_[1 - p, p]


def leaky_cols(X, y_sel):
    return [int(c) for c in CorrelationTopK(k=E3_K, method="pearson").fit(X, y_sel).get_support(indices=True)]


def pipelines(src, X, n_jobs):
    """name -> (factory(ctx), n_jobs for the outer CV, description). ctx['y_sel'] = the label vector of
    the experiment on all 250 rows (what a leaky selector would see)."""
    block_qda = lambda: Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", QDA())])  # noqa: E731
    return {
        "s003_gmm2_hv": (lambda ctx: TransductiveGMM("highvar", 2, source=src), 1,
                         "TransductiveGMM K=2 full on the train+test high-variance block (S003)"),
        "r155_gmm2_inf_tight": (lambda ctx: G.TransductiveGMMTight(columns="informative", source=src), 1,
                                "TransductiveGMMTight on the 12 label-free identified informative columns, EM tol 1e-8"),
        "r201_qda_block": (lambda ctx: Tuned(block_qda(), {"model__reg_param": REG}), n_jobs,
                           "QDA on in-fold block, nested reg (inner RepeatedStratifiedKFold(5,2), AUC)"),
        "r210_qda_block_nll": (lambda ctx: Tuned(block_qda(), {"model__reg_param": REG}, scoring="neg_log_loss"), n_jobs,
                               "QDA on in-fold block, nested reg by inner log-loss"),
        "r209_selftrain_qda_lw": (lambda ctx: Tuned(PoolSelfTraining(QDA(solver="eigen", shrinkage="auto"), threshold=0.99,
                                                                     source=src), None), n_jobs,
                                  "SelfTraining(QDA Ledoit-Wolf, thr 0.99) on 20,000 rows, fold labels only"),
        "r255_bqda_eb_hv": (lambda ctx: SubspaceModel(BayesQDA(), "highvar", source=src), n_jobs,
                            "Bayesian QDA, NIW prior by empirical Bayes on the training fold"),
        "e3_honest": (lambda ctx: build_pipeline({**E3_BASE, "selector": {"name": "corr_topk", "k": E3_K, "method": "pearson"}}),
                      n_jobs, "E3 honest: scaler + in-fold CorrelationTopK(20) + L2 logreg C=0.1"),
        "ctrl_e3_leaky": (lambda ctx: build_pipeline({**E3_BASE, "selector": {"name": "fixed_columns",
                                                                               "columns": leaky_cols(X, ctx["y_sel"])}}),
                          n_jobs, "POSITIVE CONTROL (a): E3 leaky, top-20 |corr| columns chosen on all 250 rows"),
        "ctrl_disk_reader": (lambda ctx: DiskReader(source=src), n_jobs,
                             "POSITIVE CONTROL (b): QDA fitted on all 250 rows with the true labels read from disk"),
        "ctrl_random": (lambda ctx: RandomScore(), n_jobs, "chance floor: random scores"),
    }


def fold_aucs(oof, splits, y_eval):
    return np.array([roc_auc_score(y_eval[s.val_idx], oof[s.repeat, s.val_idx]) for s in splits])


p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--n-perm", type=int, default=20)
p.add_argument("--perm-seed", type=int, default=9000)
p.add_argument("--null", choices=["A", "B"], required=True)
p.add_argument("--skip-real", action="store_true", help="do not re-run the true-label reference")
p.add_argument("--only", nargs="*", default=None)
p.add_argument("--max-folds", type=int, default=None, help="smoke tests on synthetic data only")
a = p.parse_args()
src = a.data
ON_VM = os.path.isdir("/content") and os.name != "nt"
if a.max_folds and src == "raw" and (ON_VM or not os.environ.get("SSC_ARCHIVE_ROOT")):
    raise SystemExit("--max-folds is for local smoke tests into a scratch archive only (the official CV uses all 100 folds)")
n_jobs = a.n_jobs if a.n_jobs > 0 else (os.cpu_count() or 2)

official = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))["official_cv"]
scheme = Scheme.from_dict(official)
ds = data.load(src)
X, y = ds.X.to_numpy(dtype=float), ds.y.astype(int)
splits_real = make_splits(y, scheme)
if a.max_folds:
    splits_real = splits_real[: a.max_folds]
PIPES = pipelines(src, X, n_jobs)
names = a.only or list(PIPES)
perms = ([] if a.skip_real else [-1]) + list(range(a.n_perm))
print(f"[r900] null={a.null} cpus={os.cpu_count()} n_jobs={n_jobs} n_perm={a.n_perm} perm_seed={a.perm_seed} "
      f"pipelines={names}", flush=True)

hyp = {
    "A": ("Null A (train and evaluate on permuted labels, official scheme): every honest pipeline "
          "(s003_gmm2_hv, r155_gmm2_inf_tight, r201_qda_block, r210_qda_block_nll, r209_selftrain_qda_lw, r255_bqda_eb_hv, "
          "e3_honest) has a mean AUC over permutations within 0.5 +/- 0.03, no permutation above 0.6, and an across-"
          "permutation SD comparable to e3_honest (<= ~0.05); the positive control ctrl_e3_leaky stays above 0.6 in every "
          "permutation (E3's own null: 0.705 at k=20, 5-fold); ctrl_disk_reader is ~0.5 here (Null A cannot see path b). "
          "The true-label rows reproduce the archived CV AUCs within 1e-4."),
    "B": ("Null B (train on permuted labels, evaluate against the TRUE labels, official splits): honest pipelines are "
          "symmetric around 0.5 (count of permutations above 0.5 consistent with Binomial(n, 1/2); the GMM pipelines are "
          "bimodal near 0 or 1 because labels only orient the components); ctrl_disk_reader stays > 0.95 in every "
          "permutation; ctrl_e3_leaky ~0.5 here."),
}[a.null]
cfg = {"name": f"r900_perm_null_{a.null}", "kind": "analysis", "verifier": "leak", "null": a.null,
       "n_perm": a.n_perm, "perm_seed": a.perm_seed, "cv": scheme.to_dict(), "pipelines": {k: PIPES[k][2] for k in names},
       "archived_reference": {k: v[0] for k, v in ARCHIVED.items() if k in names}, "max_folds": a.max_folds,
       "inputs": "train features + labels (permuted), train+test features (label-free parts)"}

rows, fold_rows = [], []
with Run(f"r900_perm_null_{a.null}", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    t_all = time.time()
    for pi in perms:
        real = pi < 0
        y_p = y if real else np.random.default_rng(a.perm_seed + pi).permutation(y)
        if real or a.null == "B":
            splits = splits_real
        else:
            splits = make_splits(y_p, scheme)
            if a.max_folds:
                splits = splits[: a.max_folds]
        y_eval = y if (real or a.null == "B") else y_p
        ctx = {"y_sel": y_p}
        for name in names:
            make, nj, _ = PIPES[name]
            t0 = time.time()
            cv = cross_validate(make(ctx), X, y_p, splits, scheme, test_pred="none", n_jobs=nj)
            au = fold_aucs(cv.oof, splits, y_eval)
            reps = sorted({s.repeat for s in splits})
            pooled = [roc_auc_score(y_eval, cv.oof[r]) for r in reps if np.isfinite(cv.oof[r]).all()]
            rows.append({"null": "real" if real else a.null, "perm": pi, "pipeline": name, "auc_mean": float(au.mean()),
                         "auc_sd_fold": float(au.std(ddof=1)) if len(au) > 1 else np.nan,
                         "oof_auc_mean": float(np.mean(pooled)) if pooled else np.nan,
                         "frac_folds_gt_0.6": float((au > 0.6).mean()), "frac_folds_lt_0.4": float((au < 0.4).mean()),
                         "n_constant_folds": int(cv.fold_scores["constant_pred"].sum()), "split_sig": cv.split_signature,
                         "seconds": round(time.time() - t0, 2)})
            fold_rows.append(pd.DataFrame({"null": "real" if real else a.null, "perm": pi, "pipeline": name,
                                           "repeat": [s.repeat for s in splits], "fold": [s.fold for s in splits],
                                           "auc": au}))
            print(f"[r900] {('real' if real else a.null)} perm={pi:3d} {name:22s} auc={au.mean():.4f} "
                  f"(fold sd {rows[-1]['auc_sd_fold']:.3f}) {rows[-1]['seconds']:.1f}s", flush=True)
    res = pd.DataFrame(rows)
    folds = pd.concat(fold_rows, ignore_index=True)
    run.save_table("perm_results", res)
    run.save_table("perm_fold_aucs", folds)

    # ---- summary per pipeline
    summ = []
    for name in names:
        pr = res[(res["pipeline"] == name) & (res["perm"] >= 0)]
        rr = res[(res["pipeline"] == name) & (res["perm"] < 0)]
        v = pr["auc_mean"].to_numpy()
        real_auc = float(rr["auc_mean"].iloc[0]) if len(rr) else np.nan
        n = len(v)
        k_above = int((v > 0.5).sum())
        row = {"null": a.null, "pipeline": name, "n_perm": n, "perm_auc_mean": float(v.mean()) if n else np.nan,
               "perm_auc_sd": float(v.std(ddof=1)) if n > 1 else np.nan,
               "perm_auc_min": float(v.min()) if n else np.nan, "perm_auc_max": float(v.max()) if n else np.nan,
               "frac_perm_gt_0.6": float((v > 0.6).mean()) if n else np.nan,
               "frac_perm_lt_0.4": float((v < 0.4).mean()) if n else np.nan,
               "n_perm_above_0.5": k_above,
               "sign_test_p": float(stats.binomtest(k_above, n, 0.5).pvalue) if n else np.nan,
               "t_vs_0.5": float((v.mean() - 0.5) / (v.std(ddof=1) / np.sqrt(n))) if n > 1 and v.std() > 0 else np.nan,
               "fold_auc_mean_all_perm": float(folds[(folds["pipeline"] == name) & (folds["perm"] >= 0)]["auc"].mean()) if n else np.nan,
               "frac_folds_gt_0.6_all_perm": float((folds[(folds["pipeline"] == name) & (folds["perm"] >= 0)]["auc"] > 0.6).mean()) if n else np.nan,
               "real_auc": real_auc,
               "perm_p_value_real": float((1 + (v >= real_auc).sum()) / (1 + n)) if n and np.isfinite(real_auc) else np.nan,
               "archived_run": ARCHIVED.get(name, ("", np.nan))[0], "archived_auc": ARCHIVED.get(name, ("", np.nan))[1]}
        row["real_minus_archived"] = row["real_auc"] - row["archived_auc"] if np.isfinite(row["archived_auc"]) else np.nan
        summ.append(row)
    summ = pd.DataFrame(summ)
    run.save_table("perm_summary", summ)
    tables = paths.tables_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    summ.to_csv(tables / f"r900_perm_null_{a.null}_summary.csv", index=False)
    res.to_csv(tables / f"r900_perm_null_{a.null}_per_perm.csv", index=False)

    # ---- figure: per-permutation mean AUC per pipeline
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, name in enumerate(names):
        v = res[(res["pipeline"] == name) & (res["perm"] >= 0)]["auc_mean"].to_numpy()
        jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        col = "#eb6834" if name.startswith("ctrl_e3") or name == "ctrl_disk_reader" else ("#898781" if name == "ctrl_random" else "#2a78d6")
        ax.scatter(np.full(len(v), i) + jit, v, s=16, color=col, alpha=0.8, zorder=3)
        rr = res[(res["pipeline"] == name) & (res["perm"] < 0)]
        if len(rr):
            ax.scatter([i], rr["auc_mean"], marker="*", s=90, color="#0b0b0b", zorder=4)
    ax.axhline(0.5, color="#898781", lw=1, ls="--")
    ax.axhline(0.6, color="#c3c2b7", lw=1, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean validation AUC over 100 folds")
    ax.set_title(f"R900 Null {a.null}: one dot per label permutation ({a.n_perm}); star = true labels", fontsize=10, loc="left")
    ax.grid(True, color="#e1e0d9", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    run.save_figure(f"perm_null_{a.null}", fig)
    figs = paths.figures_dir(src)
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"r900_perm_null_{a.null}.png", dpi=150, bbox_inches="tight")

    run.finish(extra_summary={"null": a.null, "n_perm": a.n_perm, "wall_seconds": round(time.time() - t_all, 1),
                              "summary": summ.to_dict(orient="records")})
    print("\n" + summ[["pipeline", "perm_auc_mean", "perm_auc_sd", "perm_auc_min", "perm_auc_max", "frac_perm_gt_0.6",
                       "frac_perm_lt_0.4", "n_perm_above_0.5", "sign_test_p", "real_auc", "archived_auc",
                       "real_minus_archived"]].to_string(index=False), flush=True)
