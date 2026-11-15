"""R152 - does the label-free structure identification work, and what class_sep produced the real data?

Simulation on make_classification replicas with KNOWN truth (research/lib_gen_replica.py, variant b):
for a grid of class_sep values, generate replicas (n_informative=12, n_redundant=8, n_clusters_per_class=1,
hypercube=True, flip_y=0, affine map a~U(0.5,3), b~U(-5,5)), run the same label-free identification as R150
(identify_structure) and score it against the truth:
  * structured set / rank / informative set / redundant set / zero-difference set recovered exactly?
  * class_sep estimate (mean |delta u|/2 over active coordinates) vs the true class_sep, hamming estimate;
  * boundary-row fraction of the label-free GMM posterior (0.1 < p < 0.9) and true Bayes AUC.
Main grid conditions on hamming = 4 (as estimated for the real data); a second grid leaves hamming free.
The real data's estimates (R150: class_sep_hat, frac boundary) are then inverted through the simulated
relation to give a calibrated class_sep with an interval.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen --high-mem -- research/r152_calibration.py --n-jobs 8
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
from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--grid", type=float, nargs="*", default=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0])
p.add_argument("--seeds-h4", type=int, default=16)
p.add_argument("--seeds-free", type=int, default=6)
a = p.parse_args()


def one(cs, seed, hamming):
    rep = G.sample_make_classification(seed, class_sep=cs, flip_y=0.0, hamming=hamming)
    X = rep.X_all
    st = G.identify_structure(X)
    roles = rep.roles
    true_S = sorted(np.where(roles != "noise")[0].tolist())
    true_I = sorted(np.where(roles == "informative")[0].tolist())
    true_R = sorted(np.where(roles == "redundant")[0].tolist())
    cp = rep.params["colperm"]
    # true inactive informative columns: original informative index whose centroids agree
    cen = rep.params["centroids"]
    agree = np.abs(cen[0] - cen[1]) < 1e-9
    true_zero = sorted(int(c) for c in true_I if agree[cp[c]])
    a_true = rep.params["a"]
    a_hat_I = dict(zip(st["informative"], st["a_informative"]))
    rel = [a_hat_I[c] / a_true[c] - 1 for c in st["informative"] if c in true_I]
    post = st["posterior"]
    ranked = [t["added_columns"] for t in st["basis_table"]]
    true_key = " ".join(str(c) for c in true_I)
    return {"class_sep": cs, "seed": seed, "hamming_cond": -1 if hamming is None else hamming,
            "hamming_true": rep.params["hamming"], "hamming_hat": st["hamming"],
            "S_ok": st["structured"] == true_S, "rank_hat": st["rank"],
            "informative_ok": st["informative"] == true_I, "redundant_ok": st["redundant"] == true_R,
            "zero_ok": st["zero_diff"] == true_zero, "true_basis_rank_in_table": (ranked.index(true_key) + 1) if true_key in ranked else -1,
            "basis_margin": st["basis_margin_logdet"], "class_sep_hat": st["class_sep_hat"],
            "a_rel_err_mean": float(np.mean(rel)) if rel else np.nan, "a_rel_err_sd": float(np.std(rel)) if rel else np.nan,
            "u_cov_offdiag_sd": st["u_cov_offdiag_sd"], "B_min": float(st["B_unscaled"].min()), "B_max": float(st["B_unscaled"].max()),
            "frac_post_0.1_0.9": float(((post > 0.1) & (post < 0.9)).mean()),
            "mean_min_post": float(np.minimum(post, 1 - post).mean()),
            "bayes_auc_private": G.bayes_auc(rep), "bayes_auc_all": G.bayes_auc(rep, "all"),
            "true_frac_post_0.1_0.9": float((np.abs(rep.bayes_test) < np.log(9)).mean())}   # 0.1 < posterior < 0.9


tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r152_calibration", "grid": a.grid, "seeds_h4": a.seeds_h4, "seeds_free": a.seeds_free,
       "generator": "lib_gen_replica.sample_make_classification(flip_y=0)", "identification": "lib_gen_replica.identify_structure"}
HYP = ("The label-free identification recovers the informative/redundant/noise roles exactly in >= 90% of "
       "make_classification replicas with class_sep >= 0.6; the class_sep estimator is nearly unbiased with "
       "sd ~0.05 at hamming 4, so the real data's class_sep is about 0.7 (between 0.6 and 0.8).")
with Run("r152_calibration", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    jobs = [(cs, s, 4) for cs in a.grid for s in range(a.seeds_h4)] + \
           [(cs, 1000 + s, None) for cs in a.grid for s in range(a.seeds_free)]
    print(f"{len(jobs)} replicas on {os.cpu_count()} cpus", flush=True)
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    run.save_table("replicas", df)
    df.to_csv(tables / "r152_replicas.csv", index=False)

    agg = df.groupby(["hamming_cond", "class_sep"]).agg(
        n=("seed", "size"), S_ok=("S_ok", "mean"), rank_ok=("rank_hat", lambda v: float((v == 12).mean())),
        informative_ok=("informative_ok", "mean"), redundant_ok=("redundant_ok", "mean"), zero_ok=("zero_ok", "mean"),
        hamming_ok=("hamming_hat", "size"), cs_hat_mean=("class_sep_hat", "mean"), cs_hat_sd=("class_sep_hat", "std"),
        frac_boundary_mean=("frac_post_0.1_0.9", "mean"), frac_boundary_sd=("frac_post_0.1_0.9", "std"),
        bayes_auc_private_mean=("bayes_auc_private", "mean"), bayes_auc_private_min=("bayes_auc_private", "min"),
        a_rel_err_sd=("a_rel_err_sd", "mean"), offdiag_sd=("u_cov_offdiag_sd", "mean")).reset_index()
    agg["hamming_ok"] = df.groupby(["hamming_cond", "class_sep"]).apply(
        lambda g: float((g.hamming_hat == g.hamming_true).mean()), include_groups=False).to_numpy()
    run.save_table("summary", agg)
    agg.to_csv(tables / "r152_summary.csv", index=False)
    (tables / "r152_summary.md").write_text(agg.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    # ---- invert for the real data
    X_real = unlabeled_features(a.data)
    st = G.identify_structure(X_real)
    post = st["posterior"]
    real_cs_hat = st["class_sep_hat"]
    real_frac = float(((post > 0.1) & (post < 0.9)).mean())
    h4 = df[df.hamming_cond == 4]
    # approximate likelihood of the real cs_hat under each grid value: normal with the simulated mean/sd
    lik = []
    for cs, g in h4.groupby("class_sep"):
        m, s = g.class_sep_hat.mean(), g.class_sep_hat.std(ddof=1)
        mf, sf = np.log(g["frac_post_0.1_0.9"]).mean(), np.log(g["frac_post_0.1_0.9"]).std(ddof=1)
        lik.append({"class_sep": cs, "cs_hat_mean": m, "cs_hat_sd": s,
                    "loglik_cs_hat": float(-0.5 * ((real_cs_hat - m) / s) ** 2 - np.log(s)),
                    "loglik_frac_boundary": float(-0.5 * ((np.log(real_frac) - mf) / sf) ** 2 - np.log(sf)),
                    "frac_boundary_geo_mean": float(np.exp(mf))})
    lik = pd.DataFrame(lik)
    lik["post_cs_hat"] = np.exp(lik.loglik_cs_hat - lik.loglik_cs_hat.max())
    lik["post_cs_hat"] /= lik.post_cs_hat.sum()
    run.save_table("real_inversion", lik)
    lik.to_csv(tables / "r152_real_inversion.csv", index=False)
    # linear calibration cs_hat = alpha + beta * cs on the h=4 grid -> invert
    beta, alpha = np.polyfit(h4.class_sep, h4.class_sep_hat, 1)
    resid_sd = float(np.std(h4.class_sep_hat - (alpha + beta * h4.class_sep), ddof=2))
    cs_cal = (real_cs_hat - alpha) / beta
    cs_lo, cs_hi = (real_cs_hat - 1.96 * resid_sd - alpha) / beta, (real_cs_hat + 1.96 * resid_sd - alpha) / beta

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(h4.class_sep + np.random.default_rng(0).normal(0, 0.005, len(h4)), h4.class_sep_hat, s=8)
    xs = np.linspace(min(a.grid), max(a.grid), 10)
    ax[0].plot(xs, alpha + beta * xs, c="k", lw=0.8)
    ax[0].axhline(real_cs_hat, c="r", ls="--", label=f"real estimate {real_cs_hat:.3f}")
    ax[0].set_xlabel("true class_sep"); ax[0].set_ylabel("estimated class_sep"); ax[0].legend()
    ax[0].set_title("class_sep estimator (hamming = 4 replicas)")
    ax[1].semilogy(h4.class_sep, h4["frac_post_0.1_0.9"], "o", ms=3)
    ax[1].axhline(real_frac, c="r", ls="--", label=f"real {real_frac:.4f}")
    ax[1].set_xlabel("true class_sep"); ax[1].set_ylabel("fraction of rows with 0.1<p<0.9"); ax[1].legend()
    acc = agg[agg.hamming_cond == -1]
    ax[2].plot(acc.class_sep, acc.informative_ok, marker="o", label="informative set exact")
    ax[2].plot(acc.class_sep, acc.zero_ok, marker="s", label="zero-difference set exact")
    ax[2].plot(acc.class_sep, acc.S_ok, marker="^", label="structured set exact")
    ax[2].set_ylim(-0.05, 1.05); ax[2].legend(); ax[2].set_xlabel("true class_sep")
    ax[2].set_title("label-free role recovery (hamming free)")
    fig.tight_layout()
    fig.savefig(figs / "r152_calibration.png", dpi=150)
    run.save_figure("calibration", fig)

    allr = df
    out = {"n_replicas": int(len(df)), "real_class_sep_hat_raw": real_cs_hat, "real_frac_boundary": real_frac,
           "calib_alpha": float(alpha), "calib_beta": float(beta), "calib_resid_sd": resid_sd,
           "real_class_sep_calibrated": float(cs_cal), "real_class_sep_95": [float(cs_lo), float(cs_hi)],
           "grid_posterior_cs_hat": dict(zip(lik.class_sep.astype(str), lik.post_cs_hat.round(4))),
           "role_recovery_informative_all": float(allr.informative_ok.mean()),
           "role_recovery_redundant_all": float(allr.redundant_ok.mean()),
           "role_recovery_zero_set_all": float(allr.zero_ok.mean()),
           "structured_set_ok_all": float(allr.S_ok.mean()), "rank_ok_all": float((allr.rank_hat == 12).mean()),
           "hamming_ok_all": float((allr.hamming_hat == allr.hamming_true).mean())}
    run.finish(extra_summary=out)
    for k, v in out.items():
        print(f"{k}: {v}")
    print(agg.to_string(index=False))
