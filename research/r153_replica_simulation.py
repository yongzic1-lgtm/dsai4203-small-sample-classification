"""R153 - replica simulation: Bayes AUC, private-test AUC of our procedures, public-LB noise, outside-block columns.

Each replica (research/lib_gen_replica.py) is a full competition (250 labelled + 19,750 test rows) with
known labels and the TRUE Bayes log-odds. Label-free mixture fits are shared across label-noise levels:
for every replica the clean labels (generating cluster) are corrupted with the make_classification flip
process (row flipped with prob flip_y, new label uniform) for each value in --flips, using the same random
numbers (nested flips), so differences between flip levels are not confounded with new data.

Procedures (as in src/ssc + research/r003; transductive ones use train+test FEATURES, labels only map
components to classes):
  gmm2_hv          TransductiveGMM K=2 full on the high-variance block (S003 = this model)
  gmm2_hv_tight    same, EM run to convergence (tol 1e-8 instead of sklearn's default 1e-3)
  gmm2_hv_pca      K=2 in the PCA(rank) basis of the block
  gmm4_hv          K=4 on the block
  gmm2_struct_pca  K=2 on ALL linearly-structured columns (label-free rule), PCA(rank) basis   [question iv]
  gmm2_inf_oracle  K=2 on the true informative columns (oracle column choice)               [question iv]
  gmm2_all         K=2 on all 150 columns (n_init 2)
  qda_hv           supervised QDA(reg 0.1) on the block (fold-mean of CV models, as S002)
  l1_all           supervised L1 logistic regression C=0.1 on all columns (fold-mean, as S001)
  bayes            the true posterior (known generating parameters)
For each: CV AUC on the 250 train rows (stratified 5-fold x --cv-repeats, seed 2026), test AUC on the
Kaggle-like private part (19,355 rows), public part (395 rows) and all test rows, test log-loss/Brier.
Public-LB noise: --lb-draws random public subsets per replica -> sd of the public AUC, P(public = 1),
how often two procedures swap order (public vs private) or tie on the public LB. The S003 public score
(0.99923) is used for an ABC-style likelihood of flip_y.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen --high-mem -- research/r153_replica_simulation.py --variant param --n-seeds 100 --n-jobs 8
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--variant", default="param", choices=["param", "mc"])
p.add_argument("--seed-start", type=int, default=0)
p.add_argument("--n-seeds", type=int, default=100)
p.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.0025, 0.005, 0.01])
p.add_argument("--class-sep", type=float, default=0.71)
p.add_argument("--hamming", type=int, default=4, help="-1 = free (mc variant only)")
p.add_argument("--n-jobs", type=int, default=-1)
p.add_argument("--lb-draws", type=int, default=200)
p.add_argument("--cv-repeats", type=int, default=4)
p.add_argument("--skip-all", action="store_true", help="skip gmm2_all (the slow 150-dim mixture)")
p.add_argument("--tag", default=None)
a = p.parse_args()
TAG = a.tag or (a.variant if a.variant == "param" else f"mc_cs{a.class_sep:g}_h{a.hamming if a.hamming >= 0 else 'free'}")
S003_PUBLIC = 0.99923
ABC_WINDOW = 0.00015

GMMS = ["gmm2_hv", "gmm2_hv_tight", "gmm2_hv_pca", "gmm4_hv", "gmm2_struct_pca"] + ([] if a.skip_all else ["gmm2_all"])
SUP = ["qda_hv", "l1_all"]
LB_MODELS = ["bayes", "gmm2_hv", "gmm2_hv_tight", "gmm2_hv_pca", "gmm4_hv", "gmm2_struct_pca", "gmm2_inf_oracle", "qda_hv"] + \
            ([] if a.skip_all else ["gmm2_all"])
PAIRS = [("gmm2_hv", "gmm2_hv_pca"), ("gmm2_hv", "gmm2_hv_tight"), ("gmm2_hv", "gmm4_hv"), ("gmm2_hv", "gmm2_struct_pca"),
         ("gmm2_hv", "gmm2_inf_oracle"), ("gmm2_hv", "qda_hv"), ("bayes", "gmm2_hv")] + \
        ([] if a.skip_all else [("gmm2_hv", "gmm2_all")])


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_rep(seed, ps):
    if a.variant == "param":
        return G.sample_parametric(ps, seed, flip_y=0.0)
    return G.sample_make_classification(seed, class_sep=a.class_sep, flip_y=0.0, hamming=None if a.hamming < 0 else a.hamming)


def worker(seed, ps):
    t0 = time.time()
    rep = make_rep(seed, ps)
    X = rep.X_all
    n_tr = len(rep.y_train)
    clus = np.r_[rep.cluster_train, rep.cluster_test]
    bayes = np.r_[rep.bayes_train, rep.bayes_test]
    inf_cols = np.where(rep.roles == "informative")[0]
    fits = {name: G.UnsupervisedGMM(X, **G.GMM_SPECS[name]).fit() for name in GMMS}
    fits["gmm2_inf_oracle"] = G.UnsupervisedGMM(X, inf_cols, 2, tol=1e-8, max_iter=5000).fit()
    hv = G.highvar_columns(X)
    S = G.structured_columns(X)[0]
    Zb = (X[:, hv] - X[:, hv].mean(0)) / X[:, hv].std(0)
    evb = np.linalg.eigvalsh(Zb.T @ Zb / len(Zb))
    rank_hv = int((evb > 1e-9 * evb.max()).sum())
    n_noise_hv = int(np.isin(hv, np.where(rep.roles == "noise")[0]).sum())
    struct = {"seed": seed, "n_highvar": len(hv), "rank_highvar": rank_hv, "n_noise_in_block": n_noise_hv,
              "n_structured": len(S), "block_spans_informative": bool(rank_hv - n_noise_hv == len(inf_cols)),
              "hamming": rep.params.get("hamming", 4), "gmm2_hv_n_iter": fits["gmm2_hv"].n_iter_,
              "gmm2_hv_tight_n_iter": fits["gmm2_hv_tight"].n_iter_,
              **{f"bic_{k}": g.bic for k, g in fits.items()}}
    rng_f = np.random.default_rng([int(seed), 4242])
    U, newlab = rng_f.random(len(X)), rng_f.integers(0, 2, len(X))
    pub = rep.public_idx
    priv = rep.private_mask
    rows_m, rows_lb, rows_pair = [], [], []
    for f in a.flips:
        y = clus.copy()
        m = U < f
        y[m] = newlab[m]
        ytr, yte = y[:n_tr], y[n_tr:]
        scheme = Scheme(kind="kfold", n_splits=5, n_repeats=a.cv_repeats, seed=2026)
        splits = make_splits(ytr, scheme)
        test_scores, probs, cvs = {"bayes": bayes[n_tr:]}, {"bayes": 1 / (1 + np.exp(-np.clip(bayes[n_tr:], -700, 700)))}, {}
        cvs["bayes"] = [G.auc_fast(ytr[s.val_idx], bayes[:n_tr][s.val_idx]) for s in splits]
        for name, g in fits.items():
            sc = g.scores(np.arange(n_tr), ytr)
            test_scores[name] = probs[name] = sc[n_tr:]
            cvs[name] = []
            for s in splits:
                scf = g.scores(s.train_idx, ytr[s.train_idx])
                cvs[name].append(G.auc_fast(ytr[s.val_idx], scf[s.val_idx]))
        for name in SUP:
            cv = cross_validate(G.make_supervised(name, X), rep.X_train, ytr, splits, scheme, X_test=rep.X_test, n_jobs=1)
            test_scores[name] = probs[name] = cv.test_pred
            cvs[name] = cv.fold_scores["auc"].tolist()
        for name, s in test_scores.items():
            c = np.asarray(cvs[name])
            rows_m.append({"seed": seed, "flip_y": f, "model": name, "n_flipped_train": int((ytr != clus[:n_tr]).sum()),
                           "n_flipped_test": int((yte != clus[n_tr:]).sum()), "n_flipped_public": int((yte != clus[n_tr:])[pub].sum()),
                           "cv_auc_mean": float(c.mean()), "cv_frac_auc1": float((c == 1).mean()),
                           "test_private": G.auc_fast(yte[priv], s[priv]), "test_public": G.auc_fast(yte[pub], s[pub]),
                           "test_all": G.auc_fast(yte, s), "test_private_clean": G.auc_fast(clus[n_tr:][priv], s[priv]),
                           "logloss_test": logloss(yte, probs[name]), "brier_test": float(np.mean((probs[name] - yte) ** 2)),
                           "n_test_boundary": int(((probs[name] > 0.1) & (probs[name] < 0.9)).sum())})
        lb = pd.DataFrame(G.lb_resample(yte, {k: test_scores[k] for k in LB_MODELS}, n_draws=a.lb_draws, seed=int(seed) * 31 + 7))
        for name, g in lb.groupby("model"):
            rows_lb.append({"seed": seed, "flip_y": f, "model": name, "public_mean": g.public.mean(), "public_sd": g.public.std(ddof=1),
                            "public_q05": g.public.quantile(.05), "public_min": g.public.min(),
                            "p_public_eq_1": float((g.public == 1).mean()), "private_mean": g.private.mean(),
                            "p_public_near_s003": float((np.abs(g.public - S003_PUBLIC) <= ABC_WINDOW).mean()),
                            "p_public_le_s003": float((g.public <= S003_PUBLIC + 1e-12).mean()),
                            "n_near_s003": int((np.abs(g.public - S003_PUBLIC) <= ABC_WINDOW).sum()),
                            "private_mean_near_s003": float(g.private[np.abs(g.public - S003_PUBLIC) <= ABC_WINDOW].mean())
                            if (np.abs(g.public - S003_PUBLIC) <= ABC_WINDOW).any() else np.nan})
        piv_pub = lb.pivot(index="draw", columns="model", values="public")
        piv_prv = lb.pivot(index="draw", columns="model", values="private")
        for A, B in PAIRS:
            dp = piv_pub[A] - piv_pub[B]
            dv = piv_prv[A] - piv_prv[B]
            rows_pair.append({"seed": seed, "flip_y": f, "A": A, "B": B, "private_diff_mean": float(dv.mean()),
                              "p_public_tie": float((dp == 0).mean()),
                              "p_swap": float(((np.sign(dp) != np.sign(dv)) & (dp != 0) & (dv != 0)).mean()),
                              "p_public_says_A_better": float((dp > 0).mean()), "p_private_A_better": float((dv > 0).mean())})
    struct["seconds"] = time.time() - t0
    return rows_m, rows_lb, rows_pair, struct


tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
cfg = {"name": f"r153_{TAG}", "variant": a.variant, "seeds": [a.seed_start, a.seed_start + a.n_seeds], "flips": a.flips,
       "class_sep": a.class_sep if a.variant == "mc" else None, "hamming": a.hamming if a.variant == "mc" else None,
       "lb_draws": a.lb_draws, "cv_scheme": f"stratified 5-fold x {a.cv_repeats}, seed 2026", "gmms": GMMS + ["gmm2_inf_oracle"],
       "supervised": SUP, "pairs": [list(t) for t in PAIRS], "library": "research/lib_gen_replica.py"}
HYP = ("(i) Bayes AUC on the private test is ~0.9999-1.0 for flip_y=0 and drops by about flip_y/2 per unit label noise "
       "(~0.995 at flip_y=0.01); (ii) gmm2_hv is within ~1e-4 of Bayes, gmm2_all and gmm4_hv lose more, qda_hv ~0.99, "
       "l1_all ~0.85; (iii) at AUC ~0.999 the public-LB sd is ~5e-4-1e-3 and transductive variants swap or tie on the "
       "public LB in most draws; (iv) adding the informative columns outside the block does not change the private AUC "
       "when the block already spans the informative space (as in the real data).")
with Run(f"r153_{TAG}", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    ps = None
    if a.variant == "param":
        ps = G.fit_parametric(unlabeled_features(a.data), data.load(a.data).y)
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    print(f"{TAG}: {len(seeds)} replicas, flips {a.flips}, {os.cpu_count()} cpus", flush=True)
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(worker)(s, ps) for s in seeds)
    M = pd.DataFrame([r for x in res for r in x[0]])
    LB = pd.DataFrame([r for x in res for r in x[1]])
    PR = pd.DataFrame([r for x in res for r in x[2]])
    ST = pd.DataFrame([x[3] for x in res])
    for name, df in (("models", M), ("lb", LB), ("pairs", PR), ("struct", ST)):
        run.save_table(name, df)
        df.to_csv(tables / f"r153_{TAG}_{name}.csv", index=False)

    # ---- (i)+(ii) private AUC per model, paired regret vs Bayes
    Mb = M[M.model == "bayes"][["seed", "flip_y", "test_private"]].rename(columns={"test_private": "bayes_private"})
    MM = M.merge(Mb, on=["seed", "flip_y"])
    MM["regret_private"] = MM.bayes_private - MM.test_private
    g2 = M[M.model == "gmm2_hv"][["seed", "flip_y", "test_private"]].rename(columns={"test_private": "gmm2_private"})
    MM = MM.merge(g2, on=["seed", "flip_y"])
    MM["minus_gmm2_hv"] = MM.test_private - MM.gmm2_private
    summ = MM.groupby(["flip_y", "model"]).agg(
        n=("seed", "size"), private_mean=("test_private", "mean"), private_sd=("test_private", "std"),
        private_q05=("test_private", lambda v: v.quantile(.05)), private_median=("test_private", "median"),
        private_q95=("test_private", lambda v: v.quantile(.95)), regret_mean=("regret_private", "mean"),
        regret_sd=("regret_private", "std"), diff_vs_gmm2_hv_mean=("minus_gmm2_hv", "mean"),
        diff_vs_gmm2_hv_se=("minus_gmm2_hv", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        p_beats_gmm2_hv=("minus_gmm2_hv", lambda v: float((v > 0).mean() + 0.5 * (v == 0).mean())),
        clean_private_mean=("test_private_clean", "mean"), public_kaggle_mean=("test_public", "mean"),
        public_kaggle_sd=("test_public", "std"), cv_auc_mean=("cv_auc_mean", "mean"), cv_frac_auc1=("cv_frac_auc1", "mean"),
        logloss_test=("logloss_test", "mean"), brier_test=("brier_test", "mean"),
        n_flipped_test=("n_flipped_test", "mean")).reset_index()
    run.save_table("summary_models", summ)
    summ.to_csv(tables / f"r153_{TAG}_summary_models.csv", index=False)
    (tables / f"r153_{TAG}_summary_models.md").write_text(summ.to_markdown(index=False, floatfmt=".6f"), encoding="utf-8")

    # ---- (iii) LB noise and swaps
    lbs = LB.groupby(["flip_y", "model"]).agg(public_sd_within=("public_sd", "mean"), public_mean=("public_mean", "mean"),
                                              p_public_eq_1=("p_public_eq_1", "mean"), p_near_s003=("p_public_near_s003", "mean"),
                                              p_le_s003=("p_public_le_s003", "mean"), private_mean=("private_mean", "mean")).reset_index()
    run.save_table("summary_lb", lbs)
    lbs.to_csv(tables / f"r153_{TAG}_summary_lb.csv", index=False)
    (tables / f"r153_{TAG}_summary_lb.md").write_text(lbs.to_markdown(index=False, floatfmt=".6f"), encoding="utf-8")
    prs = PR.groupby(["flip_y", "A", "B"]).agg(private_diff_mean=("private_diff_mean", "mean"),
                                                p_private_A_better=("p_private_A_better", "mean"),
                                                p_public_tie=("p_public_tie", "mean"), p_swap=("p_swap", "mean"),
                                                p_public_says_A_better=("p_public_says_A_better", "mean")).reset_index()
    run.save_table("summary_pairs", prs)
    prs.to_csv(tables / f"r153_{TAG}_summary_pairs.csv", index=False)
    (tables / f"r153_{TAG}_summary_pairs.md").write_text(prs.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

    # ---- ABC likelihood of flip_y from the S003 public score (gmm2_hv, random public subsets)
    abc = lbs[lbs.model == "gmm2_hv"][["flip_y", "p_near_s003", "p_le_s003", "public_mean"]].copy()
    abc["posterior_uniform_prior"] = abc.p_near_s003 / abc.p_near_s003.sum() if abc.p_near_s003.sum() > 0 else np.nan
    run.save_table("flip_abc", abc)
    abc.to_csv(tables / f"r153_{TAG}_flip_abc.csv", index=False)
    # posterior predictive of the private AUC of S003-like submissions given public ~ 0.99923 (uniform prior on flips)
    def predictive(nb):
        if not len(nb):
            return {}
        w_ = nb.n_near_s003.to_numpy(float)
        v_ = nb.private_mean_near_s003.to_numpy(float)
        o_ = np.argsort(v_)
        cw = np.cumsum(w_[o_]) / w_.sum()
        q = lambda t: float(v_[o_][min(np.searchsorted(cw, t), len(v_) - 1)])  # noqa: E731
        return {"n_replica_cells": int(len(nb)), "n_draws_near": int(w_.sum()), "private_mean": float(np.average(v_, weights=w_)),
                "private_q05": q(0.05), "private_q50": q(0.5), "private_q95": q(0.95)}
    nb_all = LB[(LB.model == "gmm2_hv") & (LB.n_near_s003 > 0)]
    pred_rows = [{"flip_y": "all (uniform prior)", **predictive(nb_all)}]
    pred_rows += [{"flip_y": f, **predictive(nb_all[nb_all.flip_y == f])} for f in a.flips]
    pred = pred_rows[0]
    pd.DataFrame(pred_rows).to_csv(tables / f"r153_{TAG}_s003_private_predictive.csv", index=False)
    run.save_table("s003_private_predictive", pd.DataFrame(pred_rows))

    # ---- (iv) columns outside the block
    w = M.pivot_table(index=["seed", "flip_y"], columns="model", values="test_private").reset_index().merge(ST[["seed", "block_spans_informative", "rank_highvar", "n_noise_in_block"]], on="seed")
    iv = []
    for (f, spans), g in w.groupby(["flip_y", "block_spans_informative"]):
        for other in ("gmm2_struct_pca", "gmm2_inf_oracle", "gmm2_hv_tight", "gmm2_hv_pca"):
            d = g[other] - g["gmm2_hv"]
            iv.append({"flip_y": f, "block_spans_informative": spans, "n_replicas": len(g), "variant": other,
                       "mean_diff_vs_gmm2_hv": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                       "p_better": float((d > 0).mean()), "p_worse": float((d < 0).mean()), "max_abs_diff": float(d.abs().max())})
    iv = pd.DataFrame(iv)
    run.save_table("outside_block", iv)
    iv.to_csv(tables / f"r153_{TAG}_outside_block.csv", index=False)
    (tables / f"r153_{TAG}_outside_block.md").write_text(iv.to_markdown(index=False, floatfmt=".6f"), encoding="utf-8")

    f0 = summ[summ.flip_y == a.flips[0]].set_index("model")
    out = {"tag": TAG, "n_replicas": len(seeds), "seconds_per_replica_mean": float(ST.seconds.mean()),
           "block_spans_informative_rate": float(ST.block_spans_informative.mean()),
           "bayes_private_by_flip": summ[summ.model == "bayes"].set_index("flip_y").private_mean.to_dict(),
           "gmm2_hv_private_by_flip": summ[summ.model == "gmm2_hv"].set_index("flip_y").private_mean.to_dict(),
           "gmm2_hv_regret_flip0": float(f0.loc["gmm2_hv", "regret_mean"]),
           "private_mean_flip0": f0.private_mean.to_dict(),
           "public_sd_within_gmm2_hv_by_flip": lbs[lbs.model == "gmm2_hv"].set_index("flip_y").public_sd_within.to_dict(),
           "flip_abc_posterior": abc.set_index("flip_y").posterior_uniform_prior.to_dict(),
           "s003_private_predictive": pred}
    run.finish(extra_summary=out)
    for k, v in out.items():
        print(f"{k}: {v}")
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(lbs.to_string(index=False))
    print(prs.to_string(index=False))
    print(abc.to_string(index=False))
    print(iv.to_string(index=False))
