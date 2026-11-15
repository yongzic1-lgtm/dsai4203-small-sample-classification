"""R206 - value of unlabeled data: AUC as a function of the number of labelled rows.

Analysis run (kind = analysis). Within the first `--repeats` repeats of the OFFICIAL splits
(5 folds each), the 200-row training fold is subsampled (stratified, `--draws` random draws)
to n_l labelled rows; every model is scored on the full 50-row validation fold. Hyperparameters
are fixed a priori (no tuning: an inner CV is impossible at n_l = 4). Models:

  sup_qda_infold_lw  QDA with per-class Ledoit-Wolf shrinkage on the block re-derived from the n_l labelled
                     rows (strict supervised, no test features)
  sup_qda_infold_r003 QDA(reg_param 0.003) on the in-fold block (0.003 = modal nested choice of R201 qda_block at
                     n = 200, so this curve is informed by R201; svd QDA, NaN below ~14 rows per class)
  sup_qda_tt_lw      same QDA-LW on the train+test block (label-free column choice)
  sup_qda_tt_r003    QDA(reg_param 0.003) on the train+test block
  sup_qda_tt_r01     QDA(reg_param 0.1) on the train+test block (R003 / S002 model; sklearn's svd QDA needs
                     more than 13 rows per class, so it is undefined (NaN) for small n_l)
  sup_lr_tt          L2 logistic C=1 on the train+test block (linear reference)
  ssl_selftrain_lw   SelfTrainingClassifier(QDA-LW, threshold 0.75) on all 20,000 rows
  ssl_selftrain_r01  SelfTrainingClassifier(QDA reg 0.1, threshold 0.75) on all 20,000 rows
  ssl_labelspread    LabelSpreading(knn k=7, alpha 0.2) on all 20,000 rows
  trans_gmm2         TransductiveGMM K=2 full covariance on the block (R003 gmm2_hv)

The supervised curves use the 13-column block chosen with train+test features in three of
the four cases, which already favours them; the strict in-fold variant shows what happens
without it.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r206_label_budget.py
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import lib_sup  # noqa: E402
from lib_sup import HighVarColumns, PoolGraphSSL, PoolSelfTraining, quiet, setup  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.models import predict_score  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import SubspaceModel, TransductiveGMM  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--repeats", type=int, default=4, help="first R official repeats (5 folds each)")
p.add_argument("--draws", type=int, default=3)
p.add_argument("--budgets", type=int, nargs="*", default=[4, 6, 10, 20, 40, 80, 120, 160, 200])
a = p.parse_args()
src = a.data
lib_sup.CURRENT["source"] = src

def qlw():
    """QDA with per-class Ledoit-Wolf shrinkage (solver='eigen'): defined for any n_k >= 2. sklearn 1.9's svd
    QDA (reg_param) needs more rows per class than features (n_k > 13) and is NaN below that."""
    return QDA(solver="eigen", shrinkage="auto")


MODELS = {
    "sup_qda_infold_lw": lambda: Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", qlw())]),
    "sup_qda_tt_lw": lambda: SubspaceModel(qlw(), "highvar", source=src),
    "sup_qda_infold_r003": lambda: Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", QDA(reg_param=0.003))]),
    "sup_qda_tt_r01": lambda: SubspaceModel(QDA(reg_param=0.1), "highvar", source=src),
    "sup_qda_tt_r003": lambda: SubspaceModel(QDA(reg_param=0.003), "highvar", source=src),
    "sup_lr_tt": lambda: SubspaceModel(LogisticRegression(C=1.0, max_iter=5000), "highvar", source=src),
    "ssl_selftrain_lw": lambda: PoolSelfTraining(qlw(), threshold=0.75, source=src),
    "ssl_selftrain_r01": lambda: PoolSelfTraining(QDA(reg_param=0.1), threshold=0.75, source=src),
    "ssl_labelspread": lambda: PoolGraphSSL("spreading", 7, 0.2, source=src),
    "trans_gmm2": lambda: TransductiveGMM("highvar", 2, source=src),
}
LABELS = {"sup_qda_infold_lw": "QDA-LW, block found in-fold (strict supervised)",
          "sup_qda_infold_r003": "QDA reg 0.003, block found in-fold (strict supervised)",
          "sup_qda_tt_lw": "QDA-LW, train+test block", "sup_qda_tt_r01": "QDA reg 0.1, train+test block",
          "sup_qda_tt_r003": "QDA reg 0.003, train+test block",
          "sup_lr_tt": "L2 logistic, train+test block", "ssl_selftrain_lw": "Self-training QDA-LW (20,000 rows)",
          "ssl_selftrain_r01": "Self-training QDA reg 0.1 (20,000 rows)",
          "ssl_labelspread": "Label spreading kNN (20,000 rows)", "trans_gmm2": "Transductive GMM K=2"}


def subsample(train_idx, y, n_l, rng):
    if n_l >= len(train_idx):
        return np.asarray(train_idx)
    yt = y[train_idx]
    pos, neg = train_idx[yt == 1], train_idx[yt == 0]
    n_pos = int(np.clip(round(n_l * yt.mean()), 2, n_l - 2))
    return np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_l - n_pos, replace=False)]


def task(X, y, s, n_l, d):
    rng = np.random.default_rng(100_000 + 1000 * s.repeat + 100 * s.fold + 10 * d + n_l)
    tr = subsample(s.train_idx, y, n_l, rng)
    out = []
    cm = quiet()
    try:
        for name, make in MODELS.items():
            t0 = time.perf_counter()
            try:
                m = clone(make()).fit(X[tr], y[tr])
                sc = predict_score(m, X[s.val_idx])
                auc = roc_auc_score(y[s.val_idx], sc) if np.ptp(sc) > 0 else 0.5
                err = ""
            except Exception as e:  # noqa: BLE001 - record and continue (e.g. degenerate tiny samples)
                auc, err = np.nan, f"{type(e).__name__}: {str(e)[:80]}"
            out.append({"model": name, "n_labels": n_l, "repeat": s.repeat, "fold": s.fold, "draw": d,
                        "n_pos": int(y[tr].sum()), "auc": auc, "seconds": time.perf_counter() - t0, "error": err})
    finally:
        cm.__exit__(None, None, None)
    return out


scheme, ds, splits = setup(src)
X, y = np.asarray(ds.X, dtype=float), ds.y
use = [s for s in splits if s.repeat < a.repeats]
jobs = [(s, n_l, d) for s in use for n_l in a.budgets for d in (range(a.draws) if n_l < 200 else [0])]
cfg = {"name": "r206_label_budget", "models": {k: LABELS[k] for k in MODELS}, "budgets": a.budgets,
       "repeats_used": a.repeats, "draws": a.draws, "cv": scheme.to_dict(), "n_evaluations_per_model": len(jobs),
       "model_code": "research/r206_label_budget.py + research/lib_sup.py", "uses_unlabeled_test_features": True}
H = ("Value of unlabeled data. Prediction: TransductiveGMM stays at AUC ~1.0 down to 4-6 labels (labels only name "
     "the two components); self-training QDA and label spreading lie in between; supervised QDA on the train+test "
     "block needs >= 80 labels to pass 0.98 and falls below 0.9 at <= 10 labels; the strict in-fold QDA is worse "
     "still at small n_l because the variance rule on a few rows cannot find the block; L2 logistic stays <= 0.9. "
     "(A smoke check showed SelfTraining with a weak base model can fall BELOW its base: self-training is not "
     "guaranteed to help.)")
with Run("r206_label_budget", cfg, kind="analysis", source=src, author=a.author, hypothesis=H) as run:
    print(f"[r206] {len(jobs)} tasks x {len(MODELS)} models, cpus={os.cpu_count()}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs)(delayed(task)(X, y, s, n_l, d) for s, n_l, d in jobs)
    df = pd.DataFrame([r for chunk in res for r in chunk])
    run.save_table("label_budget_evals", df)
    g = df.groupby(["model", "n_labels"])["auc"]
    summ = pd.DataFrame({"auc_mean": g.mean(), "auc_sd": g.std(ddof=1), "n": g.count(),
                         "auc_q10": g.quantile(0.10), "frac_auc_1": df.assign(one=df["auc"] >= 1 - 1e-12).groupby(["model", "n_labels"])["one"].mean(),
                         "n_failed": df.assign(f=df["auc"].isna()).groupby(["model", "n_labels"])["f"].sum()}).reset_index()
    summ["auc_se"] = summ["auc_sd"] / np.sqrt(summ["n"])
    summ["label"] = summ["model"].map(LABELS)
    run.save_table("label_budget_summary", summ)
    tables, figs = paths.tables_dir(src), paths.figures_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    summ.to_csv(tables / "r206_label_budget.csv", index=False)
    wide = summ.pivot(index="n_labels", columns="model", values="auc_mean")[list(MODELS)]
    (tables / "r206_label_budget.md").write_text(
        "Mean validation-fold AUC by number of labelled rows (official splits, first "
        f"{a.repeats} repeats x 5 folds x {a.draws} draws; run {run.run_id})\n\n" + wide.round(4).to_markdown() + "\n",
        encoding="utf-8")

    # figure: one line per model, categorical slots in fixed order (dataviz reference palette, light mode)
    panels = [("Supervised only (250-row labels, no unlabeled rows in the model)",
               ["sup_qda_infold_r003", "sup_qda_infold_lw", "sup_qda_tt_r003", "sup_qda_tt_r01", "sup_qda_tt_lw", "sup_lr_tt"]),
              ("Using the 20,000 unlabeled rows", ["trans_gmm2", "ssl_selftrain_r01", "ssl_selftrain_lw", "ssl_labelspread"])]
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]  # dataviz reference palette, fixed order
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, ms) in zip(axes, panels):
        for i, m in enumerate(ms):
            d = summ[(summ["model"] == m) & (summ["n_failed"] == 0)].sort_values("n_labels")  # only fully defined points
            c = "#0b0b0b" if m == "trans_gmm2" else slots[i % len(slots)]
            ax.plot(d["n_labels"], d["auc_mean"], "-", color=c, lw=2, marker="o", ms=4, label=LABELS[m])
            ax.fill_between(d["n_labels"], d["auc_mean"] - 2 * d["auc_se"], d["auc_mean"] + 2 * d["auc_se"], color=c, alpha=0.12, lw=0)
        if ms[0] == "trans_gmm2":  # strict supervised reference in the right panel
            d = summ[(summ["model"] == "sup_qda_infold_r003") & (summ["n_failed"] == 0)].sort_values("n_labels")
            ax.plot(d["n_labels"], d["auc_mean"], "--", color="#8a8984", lw=1.5, label="reference: " + LABELS["sup_qda_infold_r003"])
        ax.set_xscale("log")
        ax.set_xticks(a.budgets)
        ax.set_xticklabels([str(b) for b in a.budgets])
        ax.set_ylim(0.5, 1.005)
        ax.set_xlabel("labelled training rows (log scale)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("validation-fold ROC AUC (mean, band = 2 SE)")
    fig.suptitle("Value of unlabeled data: AUC vs number of labelled rows (official splits)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(figs / "r206_label_budget.png", dpi=160)
    run.save_figure("label_budget", fig)
    print(wide.round(4).to_string(), flush=True)
    run.finish(extra_summary={"wall_seconds": round(time.time() - t0, 1), "n_evals": int(len(df)),
                              "n_failed_evals": int(df["auc"].isna().sum()),
                              "auc_at_min_budget": summ[summ["n_labels"] == min(a.budgets)].set_index("model")["auc_mean"].round(4).to_dict(),
                              "auc_at_200": summ[summ["n_labels"] == 200].set_index("model")["auc_mean"].round(4).to_dict()})
