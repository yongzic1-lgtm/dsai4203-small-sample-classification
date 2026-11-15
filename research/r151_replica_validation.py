"""R151 - do the replica generators reproduce the real data?

Compares the real 20,000 x 150 matrix (+ 250 train labels) with replicas from research/lib_gen_replica.py:
  (a) parametric   sample_parametric(fit_parametric(real)), new rows from the fitted structure
  (b) make_classification  sample_make_classification(class_sep, hamming), new random A_k, B, scales
on label-free summary statistics (std profile, correlation eigen-spectrum, number/rank of high-variance and
structured columns, GMM purity of the train labels, BIC K=3 vs K=2, boundary-row fraction, within-class
Mardia statistics) and on the R003-style model ranking (repeated stratified 5-fold CV AUC on the 250 train
rows: L1 logreg all features, shrinkage LDA / QDA / L2 logreg on the high-variance block, transductive GMM
K=2 on the block). The same reduced CV (5 folds x n_repeats, seed 2026) is applied to the real data, so the
numbers are like-for-like; the official R003 numbers (5x20) are listed for reference.
Also checks that lib_gen_replica.make_classification_with_truth reproduces sklearn's output bit for bit.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen --high-mem -- research/r151_replica_validation.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--n-param", type=int, default=30)
p.add_argument("--n-mc", type=int, default=30)
p.add_argument("--class-sep", type=float, default=0.71)
p.add_argument("--hamming", type=int, default=4, help="-1 = free")
p.add_argument("--n-repeats", type=int, default=4)
a = p.parse_args()

MODELS = ("l1_all", "lda_hv", "qda_hv", "logreg_hv", "gmm2_hv")
R003 = {"l1_all": (0.8469, "20261022-050647_r003_sup_l1_all"), "lda_hv": (0.8779, "20261022-050653_r003_sup_lda_hv"),
        "qda_hv": (0.9958, "20261022-050659_r003_sup_qda_hv"), "logreg_hv": (0.8977, "20261022-050704_r003_sup_logreg_hv"),
        "gmm2_hv": (1.0000, "20261022-050708_r003_gmm2_hv")}
ds = data.load(a.data)
X_real = unlabeled_features(a.data)
y_real = ds.y
n_tr = len(y_real)


def evaluate(rep, informative_cols, label):
    st = G.summary_stats(rep.X_all, rep.y_train, informative_cols=informative_cols)
    cache = {}
    rk = G.cv_ranking(rep, MODELS, n_repeats=a.n_repeats, gmm_cache=cache)
    for r in rk:
        r.pop("_test")
    st.update({"bayes_auc_private": G.bayes_auc(rep) if np.isfinite(rep.bayes_test).all() else np.nan})
    return label, st, rk


def real_job():
    pseudo = G.Replica(X_real[:n_tr], y_real, X_real[n_tr:], np.zeros(len(X_real) - n_tr, int), y_real, np.zeros(len(X_real) - n_tr, int),
                       np.full(n_tr, np.nan), np.full(len(X_real) - n_tr, np.nan), np.arange(G.N_PUBLIC), np.array([]), {})
    return evaluate(pseudo, list(G.REAL_INFORMATIVE), ("real", 0))


def param_job(ps, seed):
    return evaluate(G.sample_parametric(ps, seed), ps.cols_I, ("parametric", seed))


def mc_job(seed):
    rep = G.sample_make_classification(seed, class_sep=a.class_sep, hamming=None if a.hamming < 0 else a.hamming)
    return evaluate(rep, np.where(rep.roles == "informative")[0], ("make_classification", seed))


tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r151_replica_validation", "n_param": a.n_param, "n_mc": a.n_mc, "class_sep": a.class_sep,
       "hamming": a.hamming, "cv_scheme": f"stratified 5-fold x {a.n_repeats}, seed 2026 (reduced official scheme)",
       "models": list(MODELS)}
HYP = ("Parametric replicas reproduce every real summary statistic within their 5-95% band and the R003 model "
       "ranking (GMM > QDA >> logreg_hv > LDA > L1) with CV AUCs within ~0.02; make_classification replicas "
       "(class_sep 0.71, hamming 4) reproduce the structure counts exactly but show a wider spread of "
       "difficulty because A_k, B and the scales are redrawn.")
with Run("r151_replica_validation", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    # bit-exactness of the captured make_classification internals
    from sklearn.datasets import make_classification
    exact = []
    for s in (0, 1, 2026):
        Xa, ya, tr = G.make_classification_with_truth(2000, 150, 12, 8, 0, 1, None, 0.01, 0.71, True, random_state=s)
        Xb, yb = make_classification(n_samples=2000, n_features=150, n_informative=12, n_redundant=8, n_repeated=0,
                                     n_classes=2, n_clusters_per_class=1, flip_y=0.01, class_sep=0.71, shift=0.0,
                                     scale=1.0, shuffle=False, random_state=s)
        u = Xa[:, :12]
        rebuilt = np.vstack([tr["Z"][tr["cluster_id"] == k] @ tr["A"][k] + tr["centroids"][k] for k in range(2)])
        exact.append(bool(np.array_equal(Xa, Xb) and np.array_equal(ya, yb) and np.allclose(rebuilt, u)
                          and np.allclose(Xa[:, 12:20], u @ tr["B"])))
    print("sklearn exactness:", exact, flush=True)

    ps = G.fit_parametric(X_real, y_real)
    jobs = [delayed(real_job)()] + [delayed(param_job)(ps, s) for s in range(a.n_param)] + \
           [delayed(mc_job)(s) for s in range(a.n_mc)]
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(jobs)
    long, rank_rows = [], []
    for (src, seed), st, rk in res:
        for k, v in st.items():
            long.append({"source": src, "seed": seed, "stat": k, "value": v})
        for r in rk:
            rank_rows.append({"source": src, "seed": seed, **r})
    long = pd.DataFrame(long)
    ranks = pd.DataFrame(rank_rows)
    run.save_table("stats_long", long)
    run.save_table("ranking_long", ranks)
    long.to_csv(tables / "r151_stats_long.csv", index=False)
    ranks.to_csv(tables / "r151_ranking_long.csv", index=False)

    comp = []
    for stat, g in long.groupby("stat", sort=False):
        real = g[g.source == "real"].value.iloc[0]
        row = {"stat": stat, "real": real}
        for src, tag in (("parametric", "param"), ("make_classification", "mc")):
            v = g[g.source == src].value.astype(float).dropna()
            if len(v):
                row.update({f"{tag}_mean": v.mean(), f"{tag}_sd": v.std(ddof=1), f"{tag}_q05": v.quantile(.05),
                            f"{tag}_q95": v.quantile(.95), f"{tag}_real_pct": float((v < real).mean() + 0.5 * (v == real).mean())})
        comp.append(row)
    comp = pd.DataFrame(comp)
    run.save_table("stats_compare", comp)
    comp.to_csv(tables / "r151_stats_compare.csv", index=False)
    (tables / "r151_stats_compare.md").write_text(comp.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    rk_rows = []
    real_order = ranks[ranks.source == "real"].sort_values("cv_auc_mean", ascending=False).model.tolist()
    for m in MODELS:
        row = {"model": m, "real_cv_5x%d" % a.n_repeats: float(ranks[(ranks.source == "real") & (ranks.model == m)].cv_auc_mean.iloc[0]),
               "real_cv_official_R003": R003[m][0], "R003_run": R003[m][1]}
        for src, tag in (("parametric", "param"), ("make_classification", "mc")):
            g = ranks[(ranks.source == src) & (ranks.model == m)]
            row.update({f"{tag}_cv_mean": g.cv_auc_mean.mean(), f"{tag}_cv_sd": g.cv_auc_mean.std(ddof=1),
                        f"{tag}_test_private_mean": g.test_auc_private.mean(), f"{tag}_test_private_sd": g.test_auc_private.std(ddof=1)})
        rk_rows.append(row)
    rk = pd.DataFrame(rk_rows)
    order_match = {}
    for src in ("parametric", "make_classification"):
        same = []
        for s, g in ranks[ranks.source == src].groupby("seed"):
            same.append(g.sort_values("cv_auc_mean", ascending=False).model.tolist() == real_order)
        order_match[src] = float(np.mean(same))
    run.save_table("ranking_compare", rk)
    rk.to_csv(tables / "r151_ranking_compare.csv", index=False)
    (tables / "r151_ranking_compare.md").write_text(rk.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    # figure: std profile + eigen-spectrum + ranking
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    reps_std = {"parametric": [], "make_classification": []}
    for s in range(min(a.n_param, 10)):
        reps_std["parametric"].append(np.sort(G.sample_parametric(ps, s).X_all.std(0))[::-1])
    for s in range(min(a.n_mc, 10)):
        reps_std["make_classification"].append(np.sort(G.sample_make_classification(s, class_sep=a.class_sep, hamming=None if a.hamming < 0 else a.hamming).X_all.std(0))[::-1])
    for src, c in (("parametric", "tab:blue"), ("make_classification", "tab:green")):
        for i, v in enumerate(reps_std[src]):
            ax[0].plot(v, c=c, alpha=0.3, lw=0.8, label=src if i == 0 else None)
    ax[0].plot(np.sort(X_real.std(0))[::-1], c="k", lw=1.5, label="real")
    ax[0].set_yscale("log"); ax[0].set_title("sorted column std"); ax[0].legend()
    for src, c in (("parametric", "tab:blue"), ("make_classification", "tab:green"), ("real", "k")):
        g = long[(long.source == src) & long.stat.str.startswith("corr_eig_")]
        piv = g.pivot_table(index="seed", columns="stat", values="value")
        cols = sorted(piv.columns, key=lambda t: int(t.split("_")[-1]))
        for i, (_, row) in enumerate(piv[cols].iterrows()):
            ax[1].plot(range(1, len(cols) + 1), row.values, c=c, alpha=0.3 if src != "real" else 1, lw=0.8 if src != "real" else 1.5,
                       label=src if i == 0 else None)
    ax[1].set_title("top correlation-matrix eigenvalues"); ax[1].legend()
    pos = np.arange(len(MODELS))
    for j, (src, c) in enumerate((("parametric", "tab:blue"), ("make_classification", "tab:green"))):
        data_ = [ranks[(ranks.source == src) & (ranks.model == m)].cv_auc_mean.values for m in MODELS]
        ax[2].boxplot(data_, positions=pos + (j - 0.5) * 0.3, widths=0.25, patch_artist=True,
                      boxprops=dict(facecolor=c, alpha=0.4), medianprops=dict(color=c))
    ax[2].scatter(pos, [ranks[(ranks.source == "real") & (ranks.model == m)].cv_auc_mean.iloc[0] for m in MODELS], c="k", zorder=5, label="real (5x%d)" % a.n_repeats)
    ax[2].scatter(pos, [R003[m][0] for m in MODELS], marker="x", c="r", zorder=5, label="real R003 (5x20)")
    ax[2].set_xticks(pos); ax[2].set_xticklabels(MODELS, rotation=20); ax[2].set_title("CV AUC: real vs replicas (blue=param, green=mc)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(figs / "r151_validation.png", dpi=150)
    run.save_figure("validation", fig)

    out = {"sklearn_bit_exact": exact, "ranking_order_real": real_order, "ranking_order_match_rate": order_match,
           "n_param": a.n_param, "n_mc": a.n_mc,
           "stats_real_outside_param_5_95": comp[(comp.param_real_pct < 0.05) | (comp.param_real_pct > 0.95)].stat.tolist(),
           "stats_real_outside_mc_5_95": comp[(comp.mc_real_pct < 0.05) | (comp.mc_real_pct > 0.95)].stat.tolist()}
    run.finish(extra_summary=out)
    for k, v in out.items():
        print(f"{k}: {v}")
    print(rk.to_string(index=False))
    print(comp.to_string(index=False))
