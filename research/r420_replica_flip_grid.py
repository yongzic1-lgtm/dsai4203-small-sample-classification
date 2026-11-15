"""R420 (research round 2, agent "r2_noise") - the r933 tn0 replicas with the label-noise grid extended to
flip_y = 0.02 (and 0.025 / 0.03), plus many public subsets per replica for an ABC on S003's public 0.99923.

Why: r936 combined the replicas with R156's 4-point grid {0, 0.0025, 0.005, 0.01}, which has no mass above 0.01.
R922 (verify_gen) showed that the likelihood of flip_y is flat-ish up to ~0.015 (LR(0.01) = 0.37, LR(0.015) = 0.12,
LR(0.02) = 0.034). This run adds the missing levels on the SAME replicas so research/r425 can re-weight them with
R922's flat-prior likelihood.

Design = research/r933_replica_candidates.py with --train-noise 0 --round-decimals 8 (cand's decision-relevant
group "tn0": train labels clean, test labels flipped with sklearn flip_y semantics, nested random numbers), on the
same seeds (136-357). The worker below is r933's worker restricted to that mode (r933 cannot be imported: it runs on
import). It uses the verified libraries unchanged: lib_gen_replica (sample_parametric / fit_parametric), r931_cand_lib
(LabelFree, all_scores, SubsetAUC) and lib_mix (expected_auc). Because the label noise is nested (U < f with one U
per row and seed) and the public subsets are drawn from the same generator, the arrays at the 4 original levels and
the first 16 public subsets must equal the archived r933 arrays bit for bit; research/r425 checks that against the
packed copy research/r935_r933_pooled.npz.

Per replica (all 9 procedures + Bayes, one label vector because train labels are clean):
  pub_auc / priv_auc / pub_auc_r8 / priv_auc_r8 (levels x 16 draws x procs)   r933 layout, raw and "%.8f" scores
  extra (levels x procs x 4)                                                  r933 layout
  s003_pub8 (levels x n_abc)       S003's public AUC (csv scores) on n_abc public subsets (first 16 = r933's draws)
  abc rows                         every (level, draw) with |S003 public - 0.99923| <= --abc-store: public and
                                   private AUC (csv scores) of the ABC procedures, public n1, wrong public labels

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2noise --high-mem -- research/r420_replica_flip_grid.py --seed-start 136 --n-seeds 74
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

FLIPS = sorted({round(x, 4) for x in np.arange(0, 0.02001, 0.001)} | {0.0025, 0.025, 0.03})
ABC_PROCS = ["s003_gmm2_hv", "r105_gmm2_block_mle", "r255_bqda_eb_hv", "r209_selftrain_qda_lw", "r210_qda_block_nll",
             "r201_qda_block", "bayes"]

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--seed-start", type=int, default=136)
p.add_argument("--n-seeds", type=int, default=74)
p.add_argument("--flips", type=float, nargs="*", default=FLIPS)
p.add_argument("--n-draws", type=int, default=16, help="draws stored in the r933 layout (draw 0 = replica's own)")
p.add_argument("--n-abc", type=int, default=512, help="public subsets per replica for the ABC (first n-draws shared)")
p.add_argument("--obs", type=float, default=0.99923)
p.add_argument("--abc-store", type=float, default=3e-4)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--quick", action="store_true", help="smoke test: 5x1 CV, 2 levels, few draws")
a = p.parse_args()

PROCS = C.PROCS
EVAL = PROCS + ["bayes"]
scheme = Scheme.from_dict(yaml.safe_load(open(paths.PROJECT_YAML, encoding="utf-8"))["official_cv"])
if a.quick:
    scheme = Scheme(kind="kfold", n_splits=5, n_repeats=1, seed=2026)
    a.flips, a.n_draws, a.n_abc = [0.0, 0.01], 3, 20
assert 0.0 in a.flips and a.n_abc >= a.n_draws
seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
tag = f"s{seeds[0]}-{seeds[-1]}" + ("_quick" if a.quick else "")
J_ABC = [EVAL.index(k) for k in ABC_PROCS]

HYP = ("Written before running. (1) At the 4 original levels and the first 16 public subsets the arrays equal the "
       "archived r933 tn0 arrays bit for bit (same seeds, nested label noise). (2) S003's private AUC (csv scores) has "
       "mean ~1 - f/2 at every level (R922: 0.99244 at 0.015, 0.98995 at 0.02) and sd ~6.6e-4 / 7.8e-4 there. "
       "(3) The paired private differences vs S003 stay flat in f up to 0.02 (r201 ~ -2.6e-4, r209 ~ -7e-5, GMM family "
       "~ -1e-6). (4) Public AUC of S003 within +-1.5e-4 of 0.99923 occurs in ~1-3 % of (level, subset) pairs at "
       "f = 0.0025-0.01 and much less at f = 0 or f >= 0.02. (5) Conditional on S003's public, the public AUCs of r201 "
       "and r209 change by < 1e-4 when the grid is extended (their public difference to S003 comes from boundary "
       "misorderings that do not depend on f), so B006 (r201 ~0.9988) and B007 (r209 ~0.9991) stay about right.")


def fmt8(s):
    """Scores as the submission CSV stores them (float_format '%.8f', parsed back) - as r933 --round-decimals 8."""
    return np.array([float("%.8f" % v) for v in np.asarray(s, dtype=float)])


X_real = unlabeled_features(a.data)
y_real = data.load(a.data).y
ps = G.fit_parametric(X_real, y_real)


def worker(seed):
    t0 = time.perf_counter()
    rep = G.sample_parametric(ps, seed, flip_y=0.0)
    X = rep.X_all
    n_tr = len(rep.y_train)
    lf = C.LabelFree(X, n_tr)
    clus = np.r_[rep.cluster_train, rep.cluster_test].astype(int)
    rng = np.random.default_rng([int(seed), 93301])                 # as r933: one U and one new label per row
    U, newlab = rng.random(len(X)), rng.integers(0, 2, len(X))
    n_te = len(rep.y_test)
    drng = np.random.default_rng([int(seed), 93302])                # as r933: draw 0 = replica's own public set
    pub_masks = [np.isin(np.arange(n_te), rep.public_idx)]
    for _ in range(a.n_abc - 1):
        m = np.zeros(n_te, bool)
        m[drng.choice(n_te, G.N_PUBLIC, replace=False)] = True
        pub_masks.append(m)
    pi = 1.0 / (1.0 + np.exp(-np.clip(rep.bayes_test, -700, 700)))
    bnd = (pi > 0.1) & (pi < 0.9)
    ytr = clus[:n_tr]                                                # tn0: train labels clean at every level
    if len(np.unique(ytr)) < 2:
        raise RuntimeError("one-class train labels")
    sc, info = C.all_scores(lf, rep.X_train, rep.X_test, ytr, scheme)
    sc["bayes"] = rep.bayes_test
    S = {k: (sc[k], C.SubsetAUC(sc[k]), C.SubsetAUC(fmt8(sc[k]))) for k in EVAL}
    nL, nD, nP = len(a.flips), a.n_draws, len(EVAL)
    PUB = np.full((nL, nD, nP), np.nan)
    PRIV, PUB8, PRIV8 = np.full_like(PUB, np.nan), np.full_like(PUB, np.nan), np.full_like(PUB, np.nan)
    EXTRA = np.full((nL, nP, 4), np.nan)
    S003PUB8 = np.full((nL, a.n_abc), np.nan)
    abc = []
    n_flip_test = []
    t_auc = time.perf_counter()
    for li, f in enumerate(a.flips):
        yl = clus.copy()
        m = U < f
        m[:n_tr] = False
        yl[m] = newlab[m]
        yte = yl[n_tr:]
        wrong = yte != clus[n_tr:]
        n_flip_test.append(int(wrong.sum()))
        for d in range(nD):
            pm = pub_masks[d]
            for j, k in enumerate(EVAL):
                PUB[li, d, j] = S[k][1](yte, pm)
                PRIV[li, d, j] = S[k][1](yte, ~pm)
                PUB8[li, d, j] = S[k][2](yte, pm)
                PRIV8[li, d, j] = S[k][2](yte, ~pm)
        priv0 = ~pub_masks[0]
        pr = (1 - f) * pi + f / 2
        for j, k in enumerate(EVAL):
            s = S[k][0]
            EXTRA[li, j, 0] = S[k][1](clus[n_tr:], priv0)
            EXTRA[li, j, 1] = L.expected_auc(s[priv0], pr[priv0])
            EXTRA[li, j, 2] = kendalltau(s[bnd], pi[bnd]).statistic if bnd.sum() > 2 else np.nan
            EXTRA[li, j, 3] = kendalltau(s[bnd], S["s003_gmm2_hv"][0][bnd]).statistic if bnd.sum() > 2 else np.nan
        s8 = S["s003_gmm2_hv"][2]
        for d in range(a.n_abc):
            pm = pub_masks[d]
            v = PUB8[li, d, 0] if d < nD else s8(yte, pm)
            S003PUB8[li, d] = v
            if abs(v - a.obs) <= a.abc_store + 1e-12:
                row = [li, d, int((yte[pm] == 1).sum()), int(wrong[pm].sum())]
                for j in J_ABC:
                    if d < nD:
                        row += [PUB8[li, d, j], PRIV8[li, d, j]]
                    else:
                        row += [S[EVAL[j]][2](yte, pm), S[EVAL[j]][2](yte, ~pm)]
                abc.append(row)
    meta = {"seed": int(seed), "n_flip_test": n_flip_test, "n_bayes_boundary": int(bnd.sum()),
            "n_highvar": int(len(lf.hv)), "highvar": [int(c) for c in lf.hv],
            "chosen_reg_r201_counts": pd.Series(info["chosen_reg"]["r201_qda_block"]).value_counts().to_dict(),
            "n_distinct_scores_r8": {k: int(len(np.unique(fmt8(S[k][0])))) for k in EVAL},
            "seconds_fit": float(t_auc - t0), "seconds_auc": float(time.perf_counter() - t_auc),
            "seconds": float(time.perf_counter() - t0)}
    abc = np.array(abc, dtype=float).reshape(-1, 4 + 2 * len(J_ABC))
    abc = np.c_[np.full(len(abc), seed), abc]
    return PUB, PRIV, EXTRA, PUB8, PRIV8, S003PUB8, abc, meta


cfg = {"name": f"r420_replica_flip_grid_{tag}", "design": "r933 --train-noise 0 --round-decimals 8, extended levels",
       "variant": "param", "seeds": [seeds[0], seeds[-1]], "flips": list(map(float, a.flips)), "n_draws": a.n_draws,
       "n_abc": a.n_abc, "obs_public_s003": a.obs, "abc_store_window": a.abc_store, "abc_procs": ABC_PROCS,
       "procedures": EVAL, "source_runs": C.RUN_IDS, "cv": scheme.to_dict(), "test_pred": "fold_mean",
       "code": "research/r931_cand_lib.py + research/lib_gen_replica.py + lib_mix.expected_auc",
       "reproduces": ["20261022-165947_r933_replicas_param_s136-246_tn0_r8_cv",
                      "20261022-165009_r933_replicas_param_s247-357_tn0_r8_cv"], "quick": a.quick}
with Run(f"r420_replica_flip_grid_{tag}", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(worker)(s) for s in seeds)
    names = ["pub_auc", "priv_auc", "extra", "pub_auc_r8", "priv_auc_r8", "s003_pub8"]
    for i, nm in enumerate(names):
        run.save_array(nm, np.stack([r[i] for r in res]))
    run.save_array("seeds", np.array(seeds))
    abc_cols = ["seed", "level", "draw", "n1_public", "n_wrong_public"] + [f"{x}_{k}" for k in ABC_PROCS
                                                                          for x in ("pub8", "priv8")]
    ABC = np.vstack([r[6] for r in res])
    run.save_array("abc_rows", ABC)
    (run.dir / "abc_columns.json").write_text(json.dumps(abc_cols), encoding="utf-8")
    metas = [r[7] for r in res]
    (run.dir / "replica_meta.json").write_text(json.dumps(metas, default=str), encoding="utf-8")
    # quick per-run summary (pooled analysis: research/r425)
    PRIV8 = np.stack([r[4] for r in res])
    S8 = np.stack([r[5] for r in res])
    rows = []
    for li, f in enumerate(a.flips):
        base = PRIV8[:, li, :, 0]
        acc = np.abs(S8[:, li] - a.obs)
        r = {"flip_y": f, "s003_private_mean": float(base.mean()), "s003_private_sd": float(base[:, 0].std(ddof=1))
             if len(seeds) > 1 else np.nan,
             "accept_pm5e-5": float((acc <= 5e-5 + 1e-12).mean()), "accept_pm1.5e-4": float((acc <= 1.5e-4 + 1e-12).mean()),
             "accept_pm3e-4": float((acc <= 3e-4 + 1e-12).mean()),
             "mean_flipped_test": float(np.mean([m["n_flip_test"][li] for m in metas]))}
        for k in ("r201_qda_block", "r209_selftrain_qda_lw", "r255_bqda_eb_hv", "r105_gmm2_block_mle", "bayes"):
            r[f"diff_{k}"] = float((PRIV8[:, li, :, EVAL.index(k)] - base).mean())
        rows.append(r)
    summ = pd.DataFrame(rows)
    run.save_table("quick_summary", summ)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False), flush=True)
    secs = [m["seconds"] for m in metas]
    run.finish(extra_summary={"n_seeds": len(seeds), "wall_seconds": time.time() - t0, "n_abc_rows": int(len(ABC)),
                              "replica_seconds_mean": float(np.mean(secs)), "replica_seconds_max": float(np.max(secs)),
                              "auc_seconds_mean": float(np.mean([m["seconds_auc"] for m in metas])),
                              "flips": list(map(float, a.flips)), "procs": EVAL})
