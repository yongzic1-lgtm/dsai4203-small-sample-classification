"""R402 - replica simulation of the strictly inductive final candidates (D010 switch 3) and of the boundary-row
ranking of GMM / BQDA / supervised QDA / self-training (D009 evidence 2). Research round 2, label "r2_inductive".

Design = the r933 replica design used by r936 for the final selection ("tn0"):
  * parametric replicas of the real data (research/lib_gen_replica.py, variant "param"), seeds 136-357 = the same
    222 replicas as the tn0 group of r936 (r933 runs 20261022-165947_..._s136-246_tn0_r8_cv and
    20261022-165009_..._s247-357_tn0_r8_cv); same random streams (rng [seed, 93301] for the label flips,
    [seed, 93302] for the public subsets), so the procedures shared with r933 must give identical numbers
    (checked in research/r404);
  * clean TRAINING labels, test labels flipped with sklearn flip_y semantics (nested flips across levels);
  * every procedure re-run end to end on the replica (official 5x20 splits of the replica's labels, fold-mean
    test score), scored on full-precision scores AND on scores formatted like the submission CSV ("%.8f");
  * 16 public/private splits per replica (draw 0 = the replica's own public subset).
New here: (1) the strictly inductive procedures of research/r400_inductive_lib.py; (2) noise levels extended to
flip_y 0.0075, 0.015, 0.02, 0.03 so the flat-prior posterior of R922 (likelihood support up to ~0.015) can be
weighted without truncation at 0.01; (3) boundary-row statistics per procedure: rows whose TRUE Bayes posterior is in
(0.1, 0.9) (all 19,750 test rows), AUC against the clean labels, expected AUC under the Bayes posterior, Kendall tau
with the posterior; and the r254 definition (rows where S003's own score is in (0.1, 0.9)).

Arrays in the run folder (seeds x levels x draws x procs): priv_auc, pub_auc (raw), priv_auc_r8, pub_auc_r8 (CSV);
exp_auc (seeds x levels x procs x 2: expected private AUC on draw 0's private rows under the noisy Bayes posterior,
raw / CSV scores); bnd_stats (seeds x procs x 6); real_test_pred (procs x 19750, --real-check only).
research/r403 packs them for the pooled analysis research/r404.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s r2ind --high-mem -- research/r402_inductive_replicas.py --seed-start 136 --n-seeds 111 --real-check
    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s r2ind2 --high-mem -- research/r402_inductive_replicas.py --seed-start 247 --n-seeds 111
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L4 = importlib.import_module("r400_inductive_lib")
C = L4.C
import lib_gen_replica as G  # noqa: E402
import lib_mix as LM  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
from ssc.validation import Scheme, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--seed-start", type=int, default=136)
p.add_argument("--n-seeds", type=int, default=111)
p.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03])
p.add_argument("--n-draws", type=int, default=16)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--real-check", action="store_true", help="also run every procedure once on the REAL data")
p.add_argument("--max-folds", type=int, default=None, help="smoke tests into a scratch archive only")
a = p.parse_args()
if a.max_folds and not os.environ.get("SSC_ARCHIVE_ROOT"):
    raise SystemExit("--max-folds is for smoke tests into a scratch archive only")

EVAL = L4.EVAL
PROCS = [k for k in EVAL if k != "bayes"]
JB = EVAL.index("bayes")
J0 = EVAL.index("s003_gmm2_hv")
scheme = Scheme.from_dict(yaml.safe_load(open(paths.PROJECT_YAML, encoding="utf-8"))["official_cv"])
seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
tag = f"s{seeds[0]}-{seeds[-1]}" + ("_smoke" if a.max_folds else "")

HYP = ("Written before running. Same replicas and random streams as the tn0 group of r936, so S003, r255_hv, r209, "
       "r201, r210 and the ext-grid QDA reproduce r933's CSV-score private AUCs exactly at flip 0-0.01. Strictly "
       "inductive candidates, private AUC difference vs r201 (CSV scores, R156 weights): ext-grid QDA +1.5e-4 to "
       "+2.2e-4 and r210 +6e-5 to +1.2e-4 (r936: +1.94e-4, +8.8e-5); r255_bqda_eb_trainonly (real CV 0.9990 vs r201 "
       "0.9999) -2e-4 to -1e-3; the wide-kappa EB variant within 1e-4 of it and choosing kappa0 > 1 in most folds; "
       "QDA-LW on the in-fold block (real CV 0.986) -3e-3 to -1.5e-2; the r201+BQDA-trainonly probability average "
       "within +-1.5e-4 of r201. Differences are flat across flip_y 0-0.03 (sd of private AUC grows ~linearly in "
       "flip_y). Boundary rows (Bayes posterior in (0.1, 0.9), clean labels): the Bayes posterior reaches ~0.83; "
       "S003 is within 0.005 of it; r209 and r255_hv lose 0.01-0.05, the nested supervised QDAs 0.03-0.10, and "
       "the BQDA-trainonly / QDA-LW more. The r254 gap 0.825 vs 0.702 (6 replicas, GMM-boundary rows) shrinks on the "
       "neutral Bayes-boundary definition but keeps its sign in > 90% of replicas.")


def fmt8(s):
    """Scores as the submission CSV stores them (ssc.submission float_format '%.8f', parsed back)."""
    return np.array([float("%.8f" % v) for v in np.asarray(s, dtype=float)])


ps = G.fit_parametric(unlabeled_features(a.data), data.load(a.data).y)


def scores_for(lf, Xtr, Xte, ytr):
    if not a.max_folds:
        return L4.all_scores(lf, Xtr, Xte, ytr, scheme)
    import ssc.validation as V
    orig = V.make_splits
    try:   # smoke tests only: truncate the official splits inside both libraries
        C.make_splits = lambda y, sch: orig(y, sch)[: a.max_folds]
        V.make_splits = C.make_splits
        return L4.all_scores(lf, Xtr, Xte, ytr, scheme)
    finally:
        C.make_splits = orig
        V.make_splits = orig


def boundary_stats(s, pi, yc, bnd, s003_b):
    """(AUC vs clean labels on Bayes-boundary rows, expected AUC under the Bayes posterior on them, Kendall tau with
    the posterior on them, AUC vs clean labels on S003-boundary rows (r254 definition), n rows of each)."""
    out = np.full(6, np.nan)
    if bnd.sum() > 2 and len(np.unique(yc[bnd])) == 2:
        out[0] = C.roc_auc_exact(yc[bnd], s[bnd])
        out[1] = LM.expected_auc(s[bnd], pi[bnd])
        out[2] = kendalltau(s[bnd], pi[bnd]).statistic
    if s003_b.sum() > 2 and len(np.unique(yc[s003_b])) == 2:
        out[3] = C.roc_auc_exact(yc[s003_b], s[s003_b])
    out[4], out[5] = bnd.sum(), s003_b.sum()
    return out


def worker(seed):
    t0 = time.perf_counter()
    rep = G.sample_parametric(ps, seed, flip_y=0.0)
    X = rep.X_all
    n_tr = len(rep.y_train)
    lf = C.LabelFree(X, n_tr, need=["s003_gmm2_hv"])
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
    ytr = clus[:n_tr]                                    # tn0: training labels clean
    if len(np.unique(ytr)) < 2:
        raise RuntimeError("one-class train labels")
    sc, info = scores_for(lf, rep.X_train, rep.X_test, ytr)
    sc["bayes"] = rep.bayes_test
    raw = {k: C.SubsetAUC(sc[k]) for k in EVAL}
    r8 = {k: C.SubsetAUC(fmt8(sc[k])) for k in EVAL}
    s8 = {k: fmt8(sc[k]) for k in EVAL}
    nL, nD, nP = len(a.flips), a.n_draws, len(EVAL)
    PUB, PRIV = np.full((nL, nD, nP), np.nan), np.full((nL, nD, nP), np.nan)
    PUB8, PRIV8 = np.full_like(PUB, np.nan), np.full_like(PUB, np.nan)
    EXP = np.full((nL, nP, 2), np.nan)
    n_flip_test = []
    priv0 = ~pub_masks[0]
    for li, f in enumerate(a.flips):
        yl = clus.copy()
        m = U < f
        m[:n_tr] = False
        yl[m] = newlab[m]
        yte = yl[n_tr:]
        n_flip_test.append(int((yte != clus[n_tr:]).sum()))
        pr = (1 - f) * pi + f / 2
        for d, pm in enumerate(pub_masks):
            for j, k in enumerate(EVAL):
                PUB[li, d, j] = raw[k](yte, pm)
                PRIV[li, d, j] = raw[k](yte, ~pm)
                PUB8[li, d, j] = r8[k](yte, pm)
                PRIV8[li, d, j] = r8[k](yte, ~pm)
        for j, k in enumerate(EVAL):
            EXP[li, j, 0] = LM.expected_auc(sc[k][priv0], pr[priv0])
            EXP[li, j, 1] = LM.expected_auc(s8[k][priv0], pr[priv0])
    yc = clus[n_tr:]
    s003_b = (sc["s003_gmm2_hv"] > 0.1) & (sc["s003_gmm2_hv"] < 0.9)
    BND = np.stack([boundary_stats(np.asarray(sc[k], float), pi, yc, bnd, s003_b) for k in EVAL])
    wide = (pi > 0.02) & (pi < 0.98)                     # rows kept for later boundary re-analysis
    keep = {"idx": np.where(wide)[0].astype(np.int32), "pi": pi[wide], "y": yc[wide].astype(np.int8),
            "scores": np.stack([np.asarray(sc[k], float)[wide] for k in EVAL])}
    eb = {k: Counter(f"{c[0]:g}|{c[1]:g}|{c[2]}" for c in info[k]) for k in ("eb_trainonly", "eb_trainonly_wk")}
    ncols = Counter(c[3] for c in info["eb_trainonly"])
    meta = {"seed": int(seed), "n_flip_test": n_flip_test, "n_bayes_boundary": int(bnd.sum()),
            "n_s003_boundary": int(s003_b.sum()), "seconds": time.perf_counter() - t0,
            "seconds_by_part": info.get("seconds", {}), "highvar": [int(c) for c in lf.hv],
            "chosen_reg": {k: dict(Counter(f"{r:g}" for r in v)) for k, v in info["chosen_reg"].items()},
            "eb_choice": {k: dict(v) for k, v in eb.items()}, "trainonly_block_size": dict(ncols),
            "selftrain_n_pseudo_mean": info.get("selftrain_n_pseudo_mean"),
            "n_distinct_scores": {k: int(len(np.unique(sc[k]))) for k in EVAL},
            "n_distinct_scores_r8": {k: int(len(np.unique(s8[k]))) for k in EVAL}}
    return PUB, PRIV, PUB8, PRIV8, EXP, BND, keep, meta


def worker_real():
    """Every procedure once on the REAL data: test predictions to compare with the archived test_pred.npy."""
    ds = data.load(a.data)
    X_all = unlabeled_features(a.data)
    lf = C.LabelFree(X_all, len(ds.y), need=["s003_gmm2_hv"])
    sc, info = scores_for(lf, ds.X.to_numpy(float), ds.X_test.to_numpy(float), np.asarray(ds.y).astype(int))
    eb = {k: dict(Counter(f"{c[0]:g}|{c[1]:g}|{c[2]}" for c in info[k])) for k in ("eb_trainonly", "eb_trainonly_wk")}
    return np.stack([sc[k] for k in PROCS]), eb


cfg = {"name": f"r402_inductive_replicas_{tag}", "variant": "param", "seeds": [seeds[0], seeds[-1]],
       "flips": a.flips, "n_draws": a.n_draws, "train_noise": False, "round_decimals": 8, "procedures": EVAL,
       "inductive": L4.INDUCTIVE, "transductive_references": L4.TRANSDUCTIVE, "source_runs": L4.RUN_IDS,
       "cv": scheme.to_dict(), "test_pred": "fold_mean", "wide_kappa_grid": list(L4.WIDE_KAPPA),
       "boundary": "Bayes posterior in (0.1, 0.9), all 19,750 test rows, clean labels",
       "code": "research/r400_inductive_lib.py + research/r931_cand_lib.py + research/lib_gen_replica.py",
       "real_check": bool(a.real_check), "max_folds": a.max_folds}
with Run(f"r402_inductive_replicas_{tag}", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t0 = time.time()
    tasks = [delayed(worker)(s) for s in seeds] + ([delayed(worker_real)()] if a.real_check else [])
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(tasks)
    extra = {}
    if a.real_check:
        real, real_eb = res.pop()
        run.save_array("real_test_pred", real)
        chk = []
        for j, k in enumerate(PROCS):
            h = L4.sha16(real[j])
            chk.append({"proc": k, "archived_run": L4.RUN_IDS.get(k), "sha16_recomputed": h,
                        "sha16_archived": L4.ARCHIVED_SHA16.get(k),
                        "bit_identical": (h == L4.ARCHIVED_SHA16[k]) if k in L4.ARCHIVED_SHA16 else None})
        chk = pd.DataFrame(chk)
        run.save_table("real_check", chk)
        print(chk.to_string(index=False), flush=True)
        extra["real_check"] = chk.to_dict("records")
        extra["real_eb_choice"] = real_eb
    names = ["pub_auc", "priv_auc", "pub_auc_r8", "priv_auc_r8", "exp_auc", "bnd_stats"]
    for i, nm in enumerate(names):
        run.save_array(nm, np.stack([r[i] for r in res]))
    run.save_array("seeds", np.array(seeds))
    keeps = [r[6] for r in res]
    off = np.cumsum([0] + [len(k["idx"]) for k in keeps])
    run.save_array("bndrows_offsets", off)
    run.save_array("bndrows_idx", np.concatenate([k["idx"] for k in keeps]))
    run.save_array("bndrows_pi", np.concatenate([k["pi"] for k in keeps]))
    run.save_array("bndrows_y", np.concatenate([k["y"] for k in keeps]))
    run.save_array("bndrows_scores", np.concatenate([k["scores"] for k in keeps], axis=1))
    metas = [r[7] for r in res]
    (run.dir / "replica_meta.json").write_text(json.dumps(metas, default=str), encoding="utf-8")
    # quick per-run summary (pooled analysis: research/r404)
    PRIV8 = np.stack([r[3] for r in res])
    BND = np.stack([r[5] for r in res])
    jr = EVAL.index("r201_qda_block")
    rows = []
    for li, f in enumerate(a.flips):
        for j, k in enumerate(EVAL):
            d = PRIV8[:, li, :, j] - PRIV8[:, li, :, jr]
            rows.append({"flip_y": f, "proc": k, "private_r8_mean": float(PRIV8[:, li, :, j].mean()),
                         "diff_vs_r201": float(d.mean()), "diff_vs_r201_mc_se": float(d.mean(1).std(ddof=1) / np.sqrt(len(d)))
                         if len(d) > 1 else np.nan,
                         "p_beats_r201": float((d > 0).mean()), "p_ties_r201": float((d == 0).mean()),
                         "diff_vs_s003": float((PRIV8[:, li, :, j] - PRIV8[:, li, :, J0]).mean())})
    q = pd.DataFrame(rows)
    run.save_table("quick_summary", q)
    b = pd.DataFrame({"proc": EVAL, "bnd_auc_clean": np.nanmean(BND[:, :, 0], 0), "bnd_expected_auc": np.nanmean(BND[:, :, 1], 0),
                      "bnd_kendall": np.nanmean(BND[:, :, 2], 0), "s003bnd_auc_clean": np.nanmean(BND[:, :, 3], 0),
                      "n_bnd": np.nanmean(BND[:, :, 4], 0), "n_s003bnd": np.nanmean(BND[:, :, 5], 0)})
    run.save_table("quick_boundary", b)
    with pd.option_context("display.width", 220, "display.float_format", "{:.6g}".format):
        print(q[q.flip_y.isin([0.0, 0.005, 0.01])].to_string(index=False), flush=True)
        print(b.to_string(index=False), flush=True)
    secs = [m["seconds"] for m in metas]
    eb_tot = {k: sum((Counter(m["eb_choice"][k]) for m in metas), Counter()) for k in ("eb_trainonly", "eb_trainonly_wk")}
    run.finish(extra_summary={"n_seeds": len(seeds), "wall_seconds": time.time() - t0,
                              "replica_seconds_mean": float(np.mean(secs)), "replica_seconds_max": float(np.max(secs)),
                              "procs": EVAL, "flips": a.flips,
                              "eb_choice_total": {k: dict(v.most_common(8)) for k, v in eb_tot.items()}, **extra})
