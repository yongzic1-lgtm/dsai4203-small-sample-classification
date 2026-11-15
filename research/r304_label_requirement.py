"""R304 - how many LABELS does each model need? (supporting analysis for extension E2)

E2's learning curve starts at 50 rows, where the transductive GMM is already saturated.
This analysis pushes the label count down to 2 to show where each model breaks:
for each n_labels in SIZES, NSUB stratified random label sets (at least one row per class)
are drawn from the 250 training rows; the model is fitted on those labelled rows only and
scored (ROC AUC) on the other 250 - n rows ("out-of-subset", no CV inside tiny subsets).

Models (fixed hyper-parameters, model seed 0):
  gmm2_hv   TransductiveGMM K=2 full on the 13-column block (mixture fitted once on the 20,000
            train+test FEATURE rows without labels; labels only map components -> classes)
  qda_hv    supervised QDA (reg_param 0.1) on the same block (SubspaceModel)
  l1_all    supervised L1 logistic regression C=0.1, all 150 features (E2 baseline (a))

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext12 --high-mem -- research/r304_label_requirement.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import lib_ext12  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.models import predict_score  # noqa: E402
from ssc.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--nsub", type=int, default=50)
a = p.parse_args()

SIZES = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
MODELS = {
    "gmm2_hv": {"scaler": "none", "model": {"name": "transductive_gmm",
                                            "params": {"columns": "highvar", "n_components": 2, "source": a.data}}},
    "qda_hv": {"scaler": "none", "model": {"name": "subspace_qda", "params": {"reg_param": 0.1, "source": a.data}}},
    "l1_all": {"scaler": "standard", "model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 1.0,
                                                                             "solver": "liblinear"}}},
}
HYP = ("The transductive GMM reaches AUC ~1 on the unseen rows with only a handful of labels (one per class "
       "maps the two components correctly whenever the labelled negative is not in the positive component); "
       "supervised QDA on the block needs on the order of 100-200 labels to approach 0.99, and the L1 "
       "baseline stays below 0.85 even at 200 labels.")

ds = data.load(a.data)
X, y = ds.X.to_numpy(float), ds.y
N = len(y)


def draw(n: int, i: int) -> np.ndarray:
    """Stratified label set of size n with >= 1 row per class (rng keyed by (2026, n, i))."""
    rng = np.random.default_rng([2026, n, i])
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k_pos = int(np.clip(round(n * len(pos) / N), 1, n - 1))
    return np.sort(np.concatenate([rng.choice(pos, k_pos, replace=False), rng.choice(neg, n - k_pos, replace=False)]))


def one(name: str, n: int, i: int) -> dict:
    idx = draw(n, i)
    rest = np.setdiff1d(np.arange(N), idx)
    row = {"model": name, "n_labels": n, "subset": i, "n_pos": int(y[idx].sum()), "n_eval": int(len(rest))}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = lib_ext12.build_pipeline(MODELS[name]).fit(X[idx], y[idx])
            s = predict_score(m, X[rest])
        row["auc"] = float(roc_auc_score(y[rest], s))
        row["constant"] = bool(np.ptp(s) < 1e-12)
        row["error"] = ""
    except Exception as e:  # noqa: BLE001 - e.g. QDA with a single row in a class
        row.update({"auc": np.nan, "constant": False, "error": type(e).__name__})
    return row


with Run("r304_label_requirement", {"name": "r304_label_requirement", "sizes": SIZES, "n_subsets": a.nsub,
                                    "models": MODELS, "evaluation": "out-of-subset AUC on the 250 - n other rows"},
         kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(m, n, i) for m in MODELS for n in SIZES for i in range(a.nsub))
    df = pd.DataFrame(res)
    g = df.groupby(["model", "n_labels"])
    tab = g["auc"].agg(["mean", "std", "min", "median", "max", "count"]).reset_index()
    tab = tab.merge(g["auc"].apply(lambda s: float((s >= 0.99).mean())).rename("frac_auc_ge_0.99").reset_index(),
                    on=["model", "n_labels"])
    tab = tab.merge(g["auc"].apply(lambda s: float((s >= 0.999).mean())).rename("frac_auc_ge_0.999").reset_index(),
                    on=["model", "n_labels"])
    tab = tab.merge(g["constant"].mean().rename("frac_constant").reset_index(), on=["model", "n_labels"])
    tab = tab.merge(g["error"].apply(lambda s: int((s != "").sum())).rename("n_errors").reset_index(),
                    on=["model", "n_labels"])
    run.save_table("label_requirement_all", df)
    run.save_table("label_requirement", tab)

    # smallest label count at which the MEDIAN out-of-subset AUC reaches 0.99 / the mean reaches 0.95
    first = []
    for name, t in tab.groupby("model"):
        t = t.sort_values("n_labels")
        hit99 = t.loc[t["median"] >= 0.99, "n_labels"]
        hit95 = t.loc[t["mean"] >= 0.95, "n_labels"]
        first.append({"model": name, "min_labels_median_ge_0.99": int(hit99.iloc[0]) if len(hit99) else None,
                      "min_labels_mean_ge_0.95": int(hit95.iloc[0]) if len(hit95) else None})
    first = pd.DataFrame(first)

    tdir, fdir = paths.tables_dir(a.data), paths.figures_dir(a.data)
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(tdir / "r304_label_requirement.csv", index=False)
    with open(tdir / "r304_label_requirement.md", "w", encoding="utf-8") as f:
        f.write(tab[["model", "n_labels", "mean", "std", "min", "median", "frac_auc_ge_0.99", "frac_constant",
                     "n_errors"]].to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write(first.to_markdown(index=False) + "\n")

    colors = {"gmm2_hv": "#2a78d6", "qda_hv": "#eb6834", "l1_all": "#1baf7a"}
    labels = {"gmm2_hv": "transductive GMM K=2 (block)", "qda_hv": "supervised QDA reg 0.1 (block)",
              "l1_all": "supervised L1 logistic C=0.1 (all 150)"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rng = np.random.default_rng(0)
    for name in MODELS:
        d = df[df["model"] == name].dropna(subset=["auc"])
        t = tab[tab["model"] == name].sort_values("n_labels")
        ax.scatter(d["n_labels"] * np.exp(rng.uniform(-0.04, 0.04, len(d))), d["auc"], s=6, alpha=0.25,
                   color=colors[name], edgecolor="none")
        ax.plot(t["n_labels"], t["median"], marker="o", ms=4, color=colors[name], linewidth=1.8,
                label=f"{labels[name]} (median of {a.nsub})")
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(s) for s in SIZES], fontsize=7.5)
    ax.axhline(0.99, color="#898781", linewidth=0.8, linestyle="--")
    ax.set_xlabel("labelled training rows (stratified, >= 1 per class)", fontsize=9)
    ax.set_ylabel(f"AUC on the other 250 - n rows", fontsize=9)
    ax.set_title("How many labels does each model need? (dots: individual label sets)", fontsize=10, loc="left")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fdir / "r304_label_requirement.png", dpi=200, bbox_inches="tight")
    run.save_figure("label_requirement", fig)
    run.finish(extra_summary={"per_size": tab.to_dict(orient="records"), "first_hits": first.to_dict(orient="records")})

print(tab[["model", "n_labels", "mean", "std", "min", "median", "frac_auc_ge_0.99", "frac_constant", "n_errors"]]
      .to_string(index=False))
print("\n" + first.to_string(index=False))
