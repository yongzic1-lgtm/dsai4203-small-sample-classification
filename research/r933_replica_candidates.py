"""R933 - replica simulation of the PRIVATE-LB AUC of the 8 final candidates (+ r210 with an extended reg grid),
jointly, on the same replica competitions (verifier "cand").

Each replica (research/lib_gen_replica.py) is a full competition: 250 labelled train rows, 19,750 test rows,
known test labels, true Bayes log-odds. Variant "param" = parametric bootstrap of the real data (fitted per-class
Gaussians in the 12 informative coordinates, real redundancy map, real noise columns); variant "mc" = sklearn
make_classification(class_sep 0.71, hamming 4) with new random parameters per seed.

Label noise (sklearn flip_y semantics: a row is re-labelled uniformly at random with probability flip_y, so about
flip_y/2 labels change) is applied to train AND test labels with nested random numbers, at the levels of the R156
grid (0, 0.0025, 0.005, 0.01; R156 posterior 0.077 / 0.407 / 0.361 / 0.155). Every procedure is re-run end to
end for every distinct train-label vector (official 5x20 splits of those labels, fold-mean test score), exactly
as in research/r931_cand_lib.py (which research/r932 checks against the archived test predictions).

Per replica and noise level: private AUC (19,355 rows) and public AUC (395 rows) for the replica's own public
subset (draw 0) and for --n-draws - 1 further random public subsets (private = complement), AUC against the clean
labels, the expected private AUC under the noisy Bayes posterior (variance-reduced), and Kendall tau with the Bayes
posterior on the Bayes boundary rows. Output: arrays in the run folder (draws x levels x procs), a long table for
draw 0, and the nested-QDA reg choices. research/r934 combines several r933 runs.

Options added after the first job (seeds 0-7; defaults keep the original behaviour):
  --train-noise 0      flip TEST labels only. The real training labels contain no confident contradiction (the GMM
                       has CV AUC 1.0 in 100/100 folds; R156's flip_y posterior already uses that evidence), so the
                       decision-relevant replicas are those whose train labels are clean; this samples exactly that
                       conditional at every noise level (and needs one label vector per replica, ~2x faster).
  --round-decimals 8   also score every procedure on its test scores formatted like the submission CSV
                       (ssc.submission writes float_format="%.8f"), arrays pub_auc_r8 / priv_auc_r8. Rounding to
                       8 decimals merges thousands of confident rows into ties (S003: 7,503 distinct test scores ->
                       2,897), which changes tie-dependent quantities (P(beat), the hedge value) though not the
                       expected AUC.
  --cv-stats           also store the official-CV validation-fold AUCs (100 folds) of every procedure on the replica's
                       CLEAN training labels (array cv_auc: seeds x procs x folds), so the replicas can be conditioned
                       on what the real CV showed (r210: 1 fold below 1.0, r201: 4, every other candidate 0), and run
                       the same CV once on the REAL training data (array real_cv_auc), to be checked locally against
                       the archived fold_scores.csv of every candidate run.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vcand --high-mem -- research/r933_replica_candidates.py --seed-start 0 --n-seeds 8 --n-jobs 8
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
from scipy.stats import kendalltau  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("r931_cand_lib")
import lib_gen_replica as G  # noqa: E402
import lib_mix as L  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
from ssc.validation import Scheme  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--variant", default="param", choices=["param", "mc"])
p.add_argument("--seed-start", type=int, default=0)
p.add_argument("--n-seeds", type=int, default=8)
p.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.0025, 0.005, 0.01])
p.add_argument("--n-draws", type=int, default=16, help="public/private splits per replica (draw 0 = replica's own)")
p.add_argument("--class-sep", type=float, default=0.71)
p.add_argument("--hamming", type=int, default=4)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--train-noise", type=int, default=1, choices=[0, 1], help="0 = flip test labels only")
p.add_argument("--round-decimals", type=int, default=None, help="also evaluate scores formatted with %%.<d>f")
p.add_argument("--cv-stats", action="store_true", help="store official-CV fold AUCs (replicas + real data)")
a = p.parse_args()

PROCS = C.PROCS
EVAL = PROCS + ["bayes"]
scheme = Scheme.from_dict(yaml.safe_load(open(paths.PROJECT_YAML, encoding="utf-8"))["official_cv"])
N_FOLDS = int(scheme.n_splits * scheme.n_repeats)
seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
tag = f"{a.variant}_s{a.seed_start}-{a.seed_start + a.n_seeds - 1}" + ("" if a.train_noise else "_tn0")     + (f"_r{a.round_decimals}" if a.round_decimals is not None else "") + ("_cv" if a.cv_stats else "")
HYP = ("Predictions (written before running): on parametric replicas the GMM family (S003, r105, r155, r102) is "
       "within 1e-6 of the Bayes private AUC and within 1e-7 of each other (R103/R153); BQDA (r255) and self-training "
       "(r209) lose 1e-5 to 1e-4; strictly supervised nested QDA (r201, r210) loses 1e-4 to 1e-3 (R153: QDA reg 0.1 "
       "loses 7e-3, tuning removes most of it). Label noise lowers every procedure's private AUC by ~flip_y/2 and "
       "dominates its spread (sd ~4e-4 at flip 0.005), so private AUCs are correlated > 0.95 across procedures and "
       "the probability that any candidate beats S003 on the private LB is < 0.5 for all, near 0.5 only inside the "
       "GMM family where most replicas tie. The extended reg grid (x_qda_block_nll_ext) changes r210 by < 1e-4.")
if not a.train_noise:
    HYP = ("Test-label noise only (train labels clean, as on the real data). Predictions (written before running): the "
           "supervised nested QDAs lose ~2e-4 (r210) to ~3e-4 (r201) private AUC to S003 at every noise level (the "
           "clean-label gap of the s0-7 run), not the 4e-4 to 1.3e-3 seen with train-label noise; BQDA ~1.4e-4 and "
           "self-training ~7e-5 below; the GMM family within 1e-6 of S003 in expected AUC. With the CSV's 8-decimal "
           "rounding the GMM family ties S003 exactly in most replicas, so the r102 'lottery' hedge (r932: E[max(0, "
           "diff)] = 3.9e-5 on raw scores) shrinks to < 1e-5; P(beat S003) < 0.5 for every candidate.")


def fmt(s):
    """Scores as the submission CSV stores them (float_format '%.<d>f', parsed back to float)."""
    if a.round_decimals is None:
        return None
    f = f"%.{a.round_decimals}f"
    return np.array([float(f % v) for v in np.asarray(s, dtype=float)])

ps = None
if a.variant == "param":
    X_real = unlabeled_features(a.data)
    y_real = data.load(a.data).y
    ps = G.fit_parametric(X_real, y_real)


def make_rep(seed):
    if a.variant == "param":
        return G.sample_parametric(ps, seed, flip_y=0.0)
    return G.sample_make_classification(seed, class_sep=a.class_sep, flip_y=0.0, hamming=a.hamming)


def worker(seed):
    t0 = time.perf_counter()
    rep = make_rep(seed)
    X = rep.X_all
    n_tr = len(rep.y_train)
    lf = C.LabelFree(X, n_tr)
    clus = np.r_[rep.cluster_train, rep.cluster_test].astype(int)
    rng = np.random.default_rng([int(seed), 93301])
    U, newlab = rng.random(len(X)), rng.integers(0, 2, len(X))
    n_te = len(rep.y_test)
    drng = np.random.default_rng([int(seed), 93302])
    pub_masks = [np.isin(np.arange(n_te), rep.public_idx)]
    for _ in range(a.n_draws - 1):
        m = np.zeros(n_te, bool)
        m[drng.choice(n_te, G.N_PUBLIC, replace=False)] = True
        pub_masks.append(m)
    pi = 1.0 / (1.0 + np.exp(-np.clip(rep.bayes_test, -700, 700)))
    bnd = (pi > 0.1) & (pi < 0.9)
    cache, infos = {}, []
    PUB = np.full((len(a.flips), a.n_draws, len(EVAL)), np.nan)
    PRIV = np.full_like(PUB, np.nan)
    PUB8 = np.full_like(PUB, np.nan)
    PRIV8 = np.full_like(PUB, np.nan)
    EXTRA = np.full((len(a.flips), len(EVAL), 4), np.nan)  # clean-label private AUC, expected noisy AUC, kendall bayes bnd, kendall s003 bnd
    CV = np.full((len(EVAL), N_FOLDS), np.nan)
    n_flip_train, n_flip_test = [], []
    for li, f in enumerate(a.flips):
        yl = clus.copy()
        m = U < f
        if not a.train_noise:
            m[:n_tr] = False
        yl[m] = newlab[m]
        ytr, yte = yl[:n_tr], yl[n_tr:]
        n_flip_train.append(int((ytr != clus[:n_tr]).sum()))
        n_flip_test.append(int((yte != clus[n_tr:]).sum()))
        key = ytr.tobytes()
        if key not in cache:
            if len(np.unique(ytr)) < 2:
                raise RuntimeError("one-class train labels")
            cvd = {} if (a.cv_stats and not cache) else None     # CV of the first (= clean at flip 0) label vector
            sc, info = C.all_scores(lf, rep.X_train, rep.X_test, ytr, scheme, cv_out=cvd)
            if cvd is not None:
                CV[:] = np.array([cvd.get(k, [np.nan] * CV.shape[1]) for k in PROCS] + [[np.nan] * CV.shape[1]])
            sc["bayes"] = rep.bayes_test
            cache[key] = {k: (sc[k], C.SubsetAUC(sc[k]),
                              C.SubsetAUC(fmt(sc[k])) if a.round_decimals is not None else None) for k in EVAL}
            info["flip_level"] = f
            infos.append(info)
        sc = cache[key]
        pr = (1 - f) * pi + f / 2
        for d, pm in enumerate(pub_masks):
            for j, k in enumerate(EVAL):
                PUB[li, d, j] = sc[k][1](yte, pm)
                PRIV[li, d, j] = sc[k][1](yte, ~pm)
                if sc[k][2] is not None:
                    PUB8[li, d, j] = sc[k][2](yte, pm)
                    PRIV8[li, d, j] = sc[k][2](yte, ~pm)
        priv0 = ~pub_masks[0]
        for j, k in enumerate(EVAL):
            s = sc[k][0]
            EXTRA[li, j, 0] = sc[k][1](clus[n_tr:], priv0)
            EXTRA[li, j, 1] = L.expected_auc(s[priv0], pr[priv0])
            EXTRA[li, j, 2] = kendalltau(s[bnd], pi[bnd]).statistic if bnd.sum() > 2 else np.nan
            EXTRA[li, j, 3] = kendalltau(s[bnd], sc["s003_gmm2_hv"][0][bnd]).statistic if bnd.sum() > 2 else np.nan
    meta = {"seed": int(seed), "n_label_vectors": len(cache), "n_flip_train": n_flip_train, "n_flip_test": n_flip_test,
            "n_bayes_boundary": int(bnd.sum()), "highvar": [int(c) for c in lf.hv],
            "n_highvar": int(len(lf.hv)), "informative_found": [int(c) for c in lf.inf_cols],
            "informative_true": [int(c) for c in np.where(rep.roles == "informative")[0]],
            "label_free_seconds": lf.seconds, "seconds": time.perf_counter() - t0,
            "per_label_vector": [{k: v for k, v in i.items() if k != "chosen_reg"} for i in infos],
            "chosen_reg": [i["chosen_reg"] for i in infos]}
    meta["n_distinct_scores"] = {k: int(len(np.unique(cache[next(iter(cache))][k][0]))) for k in EVAL}
    if a.round_decimals is not None:
        meta["n_distinct_scores_rounded"] = {k: int(len(np.unique(fmt(cache[next(iter(cache))][k][0]))))
                                             for k in EVAL}
    return PUB, PRIV, EXTRA, meta, PUB8, PRIV8, CV


def worker_real():
    """Official-CV fold AUCs of every procedure on the REAL training data (check against the archived runs)."""
    ds = data.load(a.data)
    X_all = unlabeled_features(a.data)
    n_tr = len(ds.y)
    lf = C.LabelFree(X_all, n_tr)
    cvd = {}
    C.all_scores(lf, ds.X.to_numpy(float), ds.X_test.to_numpy(float), np.asarray(ds.y).astype(int), scheme,
                 cv_out=cvd)
    return np.array([cvd.get(k, [np.nan] * N_FOLDS) for k in PROCS] + [[np.nan] * N_FOLDS])


cfg = {"name": f"r933_replicas_{tag}", "variant": a.variant, "seeds": [seeds[0], seeds[-1]], "flips": a.flips,
       "n_draws": a.n_draws, "procedures": EVAL, "source_runs": C.RUN_IDS, "cv": scheme.to_dict(),
       "test_pred": "fold_mean", "code": "research/r931_cand_lib.py + research/lib_gen_replica.py",
       "class_sep": a.class_sep if a.variant == "mc" else None, "hamming": a.hamming if a.variant == "mc" else None,
       "train_noise": bool(a.train_noise), "round_decimals": a.round_decimals, "cv_stats": bool(a.cv_stats)}
with Run(f"r933_replicas_{tag}", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t0 = time.time()
    tasks = [delayed(worker)(s) for s in seeds] + ([delayed(worker_real)()] if a.cv_stats else [])
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(tasks)
    if a.cv_stats:
        real_cv = res.pop()
        run.save_array("real_cv_auc", real_cv)                    # (procs incl. bayes row of NaN, folds)
        run.save_array("cv_auc", np.stack([r[6] for r in res]))   # (n_seeds, procs, folds)
        print("real-data CV: " + ", ".join(f"{k} {np.nanmean(real_cv[j]):.6f} ({int((real_cv[j] < 1).sum())} folds < 1)"
                                           for j, k in enumerate(PROCS)), flush=True)
    PUB = np.stack([r[0] for r in res])
    PRIV = np.stack([r[1] for r in res])
    EXTRA = np.stack([r[2] for r in res])
    metas = [r[3] for r in res]
    run.save_array("pub_auc", PUB)          # (n_seeds, n_levels, n_draws, n_procs)
    run.save_array("priv_auc", PRIV)
    run.save_array("extra", EXTRA)          # (n_seeds, n_levels, n_procs, 4)
    run.save_array("seeds", np.array(seeds))
    if a.round_decimals is not None:
        run.save_array(f"pub_auc_r{a.round_decimals}", np.stack([r[4] for r in res]))
        run.save_array(f"priv_auc_r{a.round_decimals}", np.stack([r[5] for r in res]))
    (run.dir / "replica_meta.json").write_text(json.dumps(metas, default=str), encoding="utf-8")
    rows = []
    for i, s in enumerate(seeds):
        for li, f in enumerate(a.flips):
            for j, k in enumerate(EVAL):
                rows.append({"seed": s, "flip_y": f, "proc": k, "private_auc": PRIV[i, li, 0, j],
                             "public_auc": PUB[i, li, 0, j], "private_auc_clean_labels": EXTRA[i, li, j, 0],
                             "expected_private_auc_noisy_bayes": EXTRA[i, li, j, 1],
                             "kendall_bayes_boundary": EXTRA[i, li, j, 2], "kendall_s003_boundary": EXTRA[i, li, j, 3]})
    long = pd.DataFrame(rows)
    run.save_table("replica_draw0", long)
    # quick per-run summary (the pooled analysis is r934)
    summ = []
    for li, f in enumerate(a.flips):
        base = PRIV[:, li, 0, 0]
        for j, k in enumerate(EVAL):
            d = PRIV[:, li, 0, j] - base
            summ.append({"flip_y": f, "proc": k, "private_mean": float(PRIV[:, li, 0, j].mean()),
                         "private_sd": float(PRIV[:, li, 0, j].std(ddof=1)) if len(seeds) > 1 else np.nan,
                         "diff_vs_s003": float(d.mean()), "p_beats": float((d > 0).mean()), "p_ties": float((d == 0).mean()),
                         "exp_auc_diff_vs_s003": float((EXTRA[:, li, j, 1] - EXTRA[:, li, 0, 1]).mean())})
    summ = pd.DataFrame(summ)
    run.save_table("quick_summary", summ)
    print(summ.to_string(index=False), flush=True)
    secs = [m["seconds"] for m in metas]
    run.finish(extra_summary={"n_seeds": len(seeds), "variant": a.variant, "wall_seconds": time.time() - t0,
                              "replica_seconds_mean": float(np.mean(secs)), "replica_seconds_max": float(np.max(secs)),
                              "n_label_vectors_mean": float(np.mean([m["n_label_vectors"] for m in metas])),
                              "procs": EVAL, "flips": a.flips})
