"""R421 (research round 2, agent "r2_noise") - R922's label-noise likelihood on the REAL test features, extended in two ways:

  (b) the public 395 rows are modelled as a STRATIFIED split (fixed number of observed positives n1*) instead of a
      uniformly random subset; does that move the flip_y support or S003's private interval?
  (a2) the public AUCs of S005 (r201_qda_block) and S006 (r209_selftrain_qda_lw) CONDITIONAL on S003's public 0.99923,
      computed jointly on the same simulated public subsets and labels (exact 5-decimal window), as a non-replica
      cross-check of the replica ABC (research/r420 + r425) and of beliefs B006 / B007.

Label model and machinery = research/r922_label_noise_private.py (verify_gen): clean class c_i ~ Bern(pi_i), pi = a
converged label-free K=2 full GMM on the PCA basis of the structured columns (R922's fit_posterior, copied verbatim);
with probability flip_y a label is redrawn uniformly (P(wrong) = f/2). Public likelihood by simulation conditional on
k wrong public labels (R922's subsets / tie-aware Mann-Whitney code, copied), here with the full histogram over
(k, observed public n1, 2D) kept so that the stratified split is the conditional on n1 = n1*. r922 cannot be imported
(it runs on import), so its helper functions are copied unchanged; the random-split window probabilities must
reproduce R922's P_obs_given_k_exact within Monte-Carlo error.

Stratified split: Kaggle would allocate the 395 public rows proportionally to the observed test labels (sklearn
_approximate_mode: n1* = round(395 N1 / 19750)). N1 = number of observed test positives ~ Normal(sum p_i, sum p_i(1-p_i)),
p_i = (1-f) pi_i + f/2, so n1* is 198 (most likely) or 197; both are reported and mixed with their probability.

Score vectors: S003 recreated as in R922 (TransductiveGMM("highvar", 2), official CV fold mean); r201 / r209 recreated
with research/r931_cand_lib.py (bit-identical to the archived test_pred.npy per r932). All three are formatted "%.8f"
like the submission CSVs and their value hashes are compared with submissions/S003, S005, S006 (hashes passed in).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2noise --high-mem -- research/r421_public_split_joint.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import hashlib
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
from scipy import stats  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("r931_cand_lib")
from ssc import data, paths  # noqa: E402
from ssc.submission import to_unit_interval  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import TransductiveGMM, unlabeled_features  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--obs", type=float, default=0.99923)
p.add_argument("--n-public", type=int, default=395)
p.add_argument("--kmax", type=int, default=10)
p.add_argument("--n-public-draws", type=int, default=12_000_000)
p.add_argument("--n-private-draws", type=int, default=3000)
p.add_argument("--batch", type=int, default=4000)
p.add_argument("--store-window", type=float, default=1.5e-4)
p.add_argument("--hash-s003", default="c70057e06406a2d6")
p.add_argument("--hash-s005", default="b1bc5699fd161b05")
p.add_argument("--hash-s006", default="284a27641b3cada1")
p.add_argument("--quick", action="store_true", help="tiny smoke test")
a = p.parse_args()
if a.quick:
    a.n_public_draws, a.n_private_draws, a.batch = 32000, 20, 2000

CAP = 1200                                         # 2D cap of the public histogram (as R922)
N1LO, N1HI = 150, 245                              # observed public n1 range kept in the histogram
F_PRIV = np.r_[np.round(np.arange(0, 0.0301, 0.0005), 4), [0.035, 0.04, 0.05, 0.06, 0.08, 0.1]]   # = R922
F_FINE = np.round(np.arange(0, 0.10001, 0.00025), 5)                                                 # = R922
WINDOWS = {"exact_round5": 5e-6, "pm5e-5": 5e-5, "pm1.5e-4": 1.5e-4}
SCORES = ["s003", "r201", "r209"]
B006 = {"interval80": (0.9980, 0.9997), "point": 0.9988, "p_higher": 0.20, "bad_below": 0.9960}
B007 = {"interval80": (0.9987, 0.9996), "point": 0.9991, "p_higher": 0.265, "p_equal": 0.04, "bad_below": 0.9975}


# ----------------------------------------------------------------------------- R922 helpers (copied verbatim)
def subsets(rng, B, N, m):
    """B uniform random m-subsets of range(N), each sorted (unique values of iid draws are a uniform random subset)."""
    M = int(m * 1.62) + 20
    idx = rng.integers(0, N, size=(B, M))
    idx.sort(axis=1)
    dup = np.zeros(idx.shape, bool)
    dup[:, 1:] = idx[:, 1:] == idx[:, :-1]
    assert ((~dup).sum(1) >= m).all()
    keys = rng.random((B, M))
    keys[dup] = 2.0
    sel = np.argpartition(keys, m - 1, axis=1)[:, :m]
    out = np.take_along_axis(idx, sel, 1)
    out.sort(axis=1)
    return out


def fit_posterior(X_all, y_train, n_tr, n_init=10):
    """Converged label-free K=2 full GMM on the PCA basis of the linearly structured columns (own implementation)."""
    Z = (X_all - X_all.mean(0)) / X_all.std(0)
    Cm = (Z.T @ Z) / len(Z)
    off = np.abs(Cm - np.diag(np.diag(Cm))).max(1)
    S = np.where(off > 0.1)[0]
    ev, V = np.linalg.eigh(Cm[np.ix_(S, S)])
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    r = int((ev > 1e-9 * ev[0]).sum())
    W = Z[:, S] @ V[:, :r]
    gm = GaussianMixture(2, covariance_type="full", n_init=n_init, tol=1e-10, reg_covar=1e-10, max_iter=20000,
                         random_state=1).fit(W)
    lp = gm._estimate_weighted_log_prob(W)
    lo = lp[:, 1] - lp[:, 0]
    if (lo[:n_tr][y_train == 1].mean()) < (lo[:n_tr][y_train == 0].mean()):
        lo = -lo
    return lo, {"structured": S.tolist(), "rank": r, "converged": bool(gm.converged_), "n_iter": int(gm.n_iter_),
                "weights": gm.weights_.tolist(), "loglik_per_row": float(gm.score(W))}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


# ----------------------------------------------------------------------------- new code
def tie_bounds(g, m):
    """Per row of a (b, m) matrix of non-decreasing group ids: index of the first / last element of each tie group
    (R922's start / end construction, factored out)."""
    b = g.shape[0]
    ar = np.arange(m)
    is_start = np.ones((b, m), bool)
    is_start[:, 1:] = g[:, 1:] != g[:, :-1]
    start = np.maximum.accumulate(np.where(is_start, ar, 0), axis=1)
    is_end = np.ones((b, m), bool)
    is_end[:, :-1] = g[:, 1:] != g[:, :-1]
    end = np.minimum.accumulate(np.where(is_end, ar, m)[:, ::-1], axis=1)[:, ::-1]
    return start, end


def u2_sorted(y, start, end):
    """2 x Mann-Whitney U (ties 1/2) for rows already sorted by score; y bool (b, m). As R922's public_chunk."""
    neg = (~y).astype(np.int32)
    cneg = np.cumsum(neg, axis=1)
    below = np.take_along_axis(cneg - neg, start, 1)
    upto = np.take_along_axis(cneg, end, 1)
    return ((2 * below + (upto - below)) * y).sum(1)


def public_chunk(seed, n_draws, B, pi_sorted, gid, others, m, kmax, obs, store):
    """R922's public simulation (random m-subset, clean labels ~ Bern(pi), exactly k wrong public labels), keeping
    the histogram over (k, observed n1, 2D) of S003 and, for draws with S003's public AUC within `store` of obs, the
    2D of the other score vectors on the same rows and labels."""
    rng = np.random.default_rng([421, int(seed)])
    N = len(pi_sorted)
    nn1 = N1HI - N1LO + 1
    hist = np.zeros((kmax + 1, nn1, CAP + 1), np.int64)
    n1all = np.zeros((kmax + 1, m + 1), np.int64)
    acc = []
    done = 0
    while done < n_draws:
        b = min(B, n_draws - done)
        pos = subsets(rng, b, N, m)
        g = gid[pos]
        c = rng.random((b, m)) < pi_sorted[pos]
        rk = np.argsort(np.argsort(rng.random((b, m)), axis=1), axis=1)
        start, end = tie_bounds(g, m)
        for k in range(kmax + 1):
            y = c ^ (rk < k)
            u2 = u2_sorted(y, start, end)
            n1 = y.sum(1)
            d2 = 2 * n1 * (m - n1) - u2
            n1all[k] += np.bincount(n1, minlength=m + 1)
            inb = (n1 >= N1LO) & (n1 <= N1HI)
            cell = (n1[inb] - N1LO) * (CAP + 1) + np.minimum(d2[inb], CAP)
            hist[k] += np.bincount(cell, minlength=nn1 * (CAP + 1)).reshape(nn1, CAP + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                auc = 1.0 - d2 / (2.0 * n1 * (m - n1))
            hit = np.flatnonzero(np.abs(auc - obs) <= store + 1e-12)
            if len(hit):
                cols = [np.full(len(hit), k), n1[hit], d2[hit]]
                for go in others:
                    G = go[pos[hit]]
                    o = np.argsort(G, axis=1, kind="stable")
                    Gs = np.take_along_axis(G, o, 1)
                    ys = np.take_along_axis(y[hit], o, 1)
                    s_, e_ = tie_bounds(Gs, m)
                    cols.append(2 * n1[hit] * (m - n1[hit]) - u2_sorted(ys, s_, e_))
                acc.append(np.stack(cols, 1).astype(np.int32))
        done += b
    acc = np.vstack(acc) if acc else np.zeros((0, 3 + len(others)), np.int32)
    return hist, n1all, acc


def n1_star(N1, N, m):
    """sklearn _approximate_mode for 2 classes: proportional allocation, remainder to the larger fractional part."""
    x = m * N1 / N
    fl = np.floor(x)
    fr = x - fl
    return (fl + (fr > 0.5) + ((fr == 0.5) & (np.random.default_rng(0).random(np.shape(x)) < 0.5))).astype(int)


def private_chunk(seed, fs, n_draws, pi, orders, m_pub, strat, B=100):
    """Private AUC (complement of the public rows) of every score vector on the same labels.
    strat=False: uniform random public subset (as R922); True: n1* observed positives + (m - n1*) observed negatives."""
    rng = np.random.default_rng([4212, int(seed), int(strat)])
    N = len(pi)
    out = []
    ar = None
    for f in fs:
        done = 0
        while done < n_draws:
            b = min(B, n_draws - done)
            if ar is None or len(ar) != b:
                ar = np.arange(b)
            c = rng.random((b, N)) < pi
            w = rng.random((b, N)) < f / 2
            y = c ^ w
            if strat:
                N1 = y.sum(1)
                n1s = n1_star(N1, N, m_pub)
                key = rng.random((b, N))
                kp = np.sort(np.where(y, key, 2.0), axis=1)[ar, n1s - 1]
                kn = np.sort(np.where(~y, key, 2.0), axis=1)[ar, m_pub - n1s - 1]
                pub = (y & (key <= kp[:, None])) | (~y & (key <= kn[:, None]))
                priv = ~pub
            else:
                priv = np.ones((b, N), bool)
                pub = subsets(rng, b, N, m_pub)
                priv[ar[:, None], pub] = False
            res = [np.full(b, f), (w & priv).sum(1), (y & ~priv).sum(1)]
            for o, st in orders:
                yo, po = y[:, o], priv[:, o]
                P = np.add.reduceat((yo & po).astype(np.int32), st, axis=1)
                Q = np.add.reduceat((~yo & po).astype(np.int32), st, axis=1)
                below = np.cumsum(Q, axis=1) - Q
                u2 = (P * (2 * below + Q)).sum(1).astype(float)
                res.append(u2 / (2.0 * P.sum(1) * Q.sum(1)))
            out.append(np.stack(res, 1))
            done += b
    return np.vstack(out)


def window_mask_d2(n1, m, w, obs):
    """Boolean (n1 values x 2D values) mask: public AUC 1 - (2D/2)/(n1 n0) inside the window."""
    d = np.arange(CAP + 1)[None, :] / 2.0
    nn = (n1 * (m - n1)).astype(float)[:, None]
    auc = 1.0 - d / nn
    auc[:, CAP] = -1.0
    if w <= 5e-6 + 1e-15:
        return (auc >= obs - 5e-6 - 1e-12) & (auc < obs + 5e-6 - 1e-12)
    return np.abs(auc - obs) <= w + 1e-12


def in_window(auc, w, obs=None):
    """Same window rule as window_mask_d2, applied to exact AUC values."""
    obs = a.obs if obs is None else obs
    if w <= 5e-6 + 1e-15:
        return (auc >= obs - 5e-6 - 1e-12) & (auc < obs + 5e-6 - 1e-12)
    return np.abs(auc - obs) <= w + 1e-12


# ----------------------------------------------------------------------------- main
tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r421_public_split_joint", "obs_public_s003": a.obs, "n_public": a.n_public, "kmax": a.kmax,
       "n_public_draws": a.n_public_draws, "n_private_draws_per_f": a.n_private_draws, "store_window": a.store_window,
       "windows": WINDOWS, "f_private_grid": F_PRIV.tolist(), "n1_hist_range": [N1LO, N1HI],
       "posterior": "R922 fit_posterior (converged K=2 full GMM, PCA basis, tol 1e-10, reg 1e-10)",
       "split_models": ["random 395-subset (R922)", "stratified: n1* = 197", "stratified: n1* = 198",
                        "stratified: n1* mixed by P(round(395 N1/19750))"],
       "priors": ["uniform_0_0.1 (primary, R922)", "default_aware (R922)"],
       "beliefs_checked": {"B006": B006, "B007": B007}, "reuses": ["research/r922_label_noise_private.py (helpers)",
                                                                   "research/r931_cand_lib.py"], "quick": a.quick}
HYP = ("Written before running. (b) Stratification fixes the observed public n1 at 198 (or 197) instead of letting it "
       "vary with sd ~10; the AUC window for 0.99923 is the single value 2D = 60 for every n1 near 197.5, and the 2D "
       "distribution given k depends on n1 only through n1*n0 (+-0.3 %), so P(0.99923 | k) changes by < 10 % for each "
       "k, the flip_y support [0, ~0.015] and MLE ~0.003 stay, and S003's private 90 % interval moves by < 3e-4. "
       "(a2) Conditional on S003's exact public 0.99923 and with real-feature labels, r201's public AUC is about "
       "0.9985-0.9990 (B006 point 0.9988) and r209's about 0.9990-0.9993 (B007 point 0.9991); both are below S003's in "
       "most draws (P(higher) 0.1-0.3), and the random-split and stratified answers differ by < 1e-4. The random-split "
       "window probabilities reproduce R922's (1.24e-4, 2.44e-3, 4.1e-4 for k = 0, 1, 2) within MC error.")

with Run("r421_public_split_joint", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t_start = time.time()
    ds = data.load(a.data)
    y_tr = np.asarray(ds.y).astype(int)
    n_tr = len(y_tr)
    X_all = np.asarray(unlabeled_features(a.data), dtype=float)
    scheme = Scheme.from_dict(yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))["official_cv"])
    if a.quick:
        scheme = Scheme(kind="kfold", n_splits=5, n_repeats=1, seed=2026)

    # ---- score vectors as Kaggle saw them ("%.8f" strings of the submission CSVs)
    splits = make_splits(y_tr, scheme)
    cvr = cross_validate(TransductiveGMM("highvar", 2, source=a.data), ds.X, y_tr, splits, scheme,
                         X_test=ds.X_test, n_jobs=1)
    lf = C.LabelFree(X_all, n_tr, need=["r209_selftrain_qda_lw"])
    sc, info = C.all_scores(lf, ds.X.to_numpy(float), ds.X_test.to_numpy(float), y_tr, scheme,
                            procs=["r201_qda_block", "r209_selftrain_qda_lw"])
    raw = {"s003": cvr.test_pred, "r201": sc["r201_qda_block"], "r209": sc["r209_selftrain_qda_lw"]}
    want = {"s003": a.hash_s003, "r201": a.hash_s005, "r209": a.hash_s006}
    vals, hashes = {}, {}
    for k, v in raw.items():
        u, tr = to_unit_interval(v)
        strs = [f"{x:.8f}" for x in u]
        hashes[k] = {"hash": hashlib.sha256("\n".join(strs).encode()).hexdigest()[:16], "transform": tr,
                     "n_distinct": int(len(set(strs)))}
        hashes[k]["matches_submission"] = hashes[k]["hash"] == want[k]
        vals[k] = np.array([float(x) for x in strs])
    print("score hashes:", json.dumps(hashes), flush=True)

    # ---- posterior and train-label likelihood (R922)
    lo, ginfo = fit_posterior(X_all, y_tr, n_tr, n_init=1 if a.quick else 10)
    pi_all = sigmoid(lo)
    pi_tr, pi_te = pi_all[:n_tr], pi_all[n_tr:]
    p_obs_tr = np.where(y_tr == 1, pi_tr, 1 - pi_tr)
    ll_tr = np.array([np.log((1 - f) * p_obs_tr + f / 2).sum() for f in F_FINE])
    Ltr = np.exp(ll_tr - ll_tr.max())

    # ---- sorted structures (S003 order) and dense rank ids of the other scores in that order
    order = np.argsort(vals["s003"], kind="stable")
    ss = vals["s003"][order]
    gid = (np.cumsum(np.r_[True, ss[1:] != ss[:-1]]) - 1).astype(np.int32)
    others = [(rankdata(vals[k][order], method="dense") - 1).astype(np.int32) for k in ("r201", "r209")]
    m = a.n_public

    # ---- public simulation (joint)
    t0 = time.time()
    nchunk = a.n_jobs * 4
    per = int(np.ceil(a.n_public_draws / nchunk))
    res = Parallel(n_jobs=a.n_jobs)(delayed(public_chunk)(s, per, a.batch, pi_te[order], gid, others, m, a.kmax,
                                                          a.obs, a.store_window) for s in range(nchunk))
    H = np.sum([r[0] for r in res], axis=0)
    N1ALL = np.sum([r[1] for r in res], axis=0)
    ACC = np.vstack([r[2] for r in res])
    ndraw = int(N1ALL[0].sum())
    print(f"public sim: {ndraw:,} draws per k in {time.time() - t0:.0f} s; {len(ACC):,} stored rows", flush=True)
    run.save_array("public_hist_k_n1_d2", H.astype(np.int32))
    run.save_array("public_n1_counts", N1ALL)
    run.save_array("public_accepted_rows", ACC)       # columns k, n1, d2_s003, d2_r201, d2_r209

    # window probabilities per k: random split, stratified n1* = 197 / 198
    n1v = np.arange(N1LO, N1HI + 1)
    wp_rows = []
    PW = {}
    for wname, w in WINDOWS.items():
        M = window_mask_d2(n1v, m, w, a.obs)
        hitk = (H * M[None]).sum(2)                   # (k, n1)
        PW[("random", wname)] = hitk.sum(1) / ndraw
        for n1s in (197, 198):
            PW[(f"strat{n1s}", wname)] = hitk[:, n1s - N1LO] / N1ALL[:, n1s]
        for k in range(a.kmax + 1):
            r = {"window": wname, "k_wrong_public": k, "P_random": PW[("random", wname)][k],
                 "hits_random": int(hitk[k].sum()), "P_strat197": PW[("strat197", wname)][k],
                 "P_strat198": PW[("strat198", wname)][k], "hits_strat198": int(hitk[k, 198 - N1LO]),
                 "share_draws_n1_198": float(N1ALL[k, 198] / ndraw)}
            wp_rows.append(r)
    WP = pd.DataFrame(wp_rows)
    run.save_table("window_probs", WP)
    WP.to_csv(tables / "r421_window_probs.csv", index=False)
    print(WP[WP.window == "exact_round5"].to_string(index=False), flush=True)

    # P(n1* = 198 | f): N1 ~ Normal(sum p, sum p(1-p)), p = (1-f) pi + f/2, n1* = round(395 N1 / 19750)
    Nte = len(pi_te)

    def p_n1s(f):
        pp = (1 - f) * pi_te + f / 2
        mu, sd = pp.sum(), np.sqrt((pp * (1 - pp)).sum())
        thr = (197.5 * Nte / m)                       # n1* = 198 iff N1 >= thr (to within the 0.5 tie rule)
        p198 = 1 - stats.norm.cdf((np.ceil(thr) - 0.5 - mu) / sd)
        return {197: 1 - p198, 198: p198}

    kk = np.arange(a.kmax + 1)
    lik_rows = []
    for f, lt in zip(F_FINE, Ltr):
        bk = stats.binom.pmf(kk, m, f / 2)
        pn = p_n1s(f)
        row = {"flip_y": f, "L_train_rel": lt, "p_n1star_198": pn[198]}
        for wname in WINDOWS:
            row[f"Lpub_random_{wname}"] = float((bk * PW[("random", wname)]).sum())
            row[f"Lpub_strat197_{wname}"] = float((bk * PW[("strat197", wname)]).sum())
            row[f"Lpub_strat198_{wname}"] = float((bk * PW[("strat198", wname)]).sum())
            row[f"Lpub_stratmix_{wname}"] = pn[197] * row[f"Lpub_strat197_{wname}"] + pn[198] * row[f"Lpub_strat198_{wname}"]
        lik_rows.append(row)
    LK = pd.DataFrame(lik_rows)
    SPLITS = ["random", "strat197", "strat198", "stratmix"]
    for sp in SPLITS:
        LK[f"Lcomb_{sp}"] = LK.L_train_rel * LK[f"Lpub_{sp}_exact_round5"]
        LK[f"Lcomb_{sp}_rel"] = LK[f"Lcomb_{sp}"] / LK[f"Lcomb_{sp}"].max()
    run.save_table("likelihood_fine", LK)
    LK.to_csv(tables / "r421_likelihood_fine.csv", index=False)

    # ---- private simulation (S003, r201, r209 on the same labels), random and stratified public split
    t0 = time.time()
    ords = []
    for k in SCORES:
        o = np.argsort(vals[k], kind="stable")
        v = vals[k][o]
        ords.append((o, np.flatnonzero(np.r_[True, v[1:] != v[:-1]])))
    nper = int(np.ceil(a.n_private_draws / a.n_jobs))
    PRV = {}
    for strat in (False, True):
        out = Parallel(n_jobs=a.n_jobs)(delayed(private_chunk)(j, F_PRIV, nper, pi_te, ords, m, strat)
                                        for j in range(a.n_jobs))
        D = pd.DataFrame(np.vstack(out), columns=["flip_y", "n_wrong_private", "n1_public"] +
                         [f"priv_{k}" for k in SCORES])
        D["split"] = "stratified" if strat else "random"
        PRV[strat] = D
    print(f"private sim: {sum(len(v) for v in PRV.values()):,} label draws in {time.time() - t0:.0f} s", flush=True)
    PR = pd.concat(PRV.values())
    PR["d_r201"] = PR.priv_r201 - PR.priv_s003
    PR["d_r209"] = PR.priv_r209 - PR.priv_s003
    agg = {"n": ("priv_s003", "size"), "s003_mean": ("priv_s003", "mean"), "s003_sd": ("priv_s003", "std"),
           "s003_q05": ("priv_s003", lambda v: v.quantile(.05)), "s003_q95": ("priv_s003", lambda v: v.quantile(.95)),
           "r201_mean": ("priv_r201", "mean"), "r209_mean": ("priv_r209", "mean"),
           "d_r201_mean": ("d_r201", "mean"), "d_r201_sd": ("d_r201", "std"),
           "d_r209_mean": ("d_r209", "mean"), "d_r209_sd": ("d_r209", "std"),
           "p_r201_beats_s003": ("d_r201", lambda v: float((v > 0).mean())),
           "p_r209_beats_s003": ("d_r209", lambda v: float((v > 0).mean())),
           "mean_n1_public": ("n1_public", "mean")}
    PS = PR.groupby(["split", "flip_y"]).agg(**agg).reset_index()
    run.save_table("private_by_flip", PS)
    PS.to_csv(tables / "r421_private_by_flip.csv", index=False)

    # ---- posteriors over f and posterior predictive of S003's private AUC (R922's nearest-grid mapping)
    def prior_w(name):
        w = np.zeros_like(F_FINE)
        if name == "uniform_0_0.1":
            w[:] = 1.0
            return w / w.sum()
        w[1:] = 0.3 / (len(F_FINE) - 1)              # default_aware (R922)
        w[0] += 0.2
        w[np.argmin(np.abs(F_FINE - 0.01))] += 0.5
        return w

    fpriv = np.array(sorted(PR.flip_y.unique()))
    near = np.abs(F_FINE[:, None] - fpriv[None, :]).argmin(1)
    rng = np.random.default_rng(42199)
    sup_rows = []
    for sp in SPLITS:
        Lc = LK[f"Lcomb_{sp}"].to_numpy()
        imax = int(np.argmax(Lc))
        rel = Lc / Lc.max()
        s8, s32 = F_FINE[rel >= 1 / 8], F_FINE[rel >= 1 / 32]
        priv_split = PRV[sp != "random"]
        by_f = {f: g for f, g in priv_split.groupby("flip_y")}
        for pname in ("uniform_0_0.1", "default_aware"):
            w = prior_w(pname) * Lc
            w = w / w.sum()
            cdf = np.cumsum(w)
            wf = np.bincount(near, weights=w, minlength=len(fpriv))
            parts = []
            for j, f in enumerate(fpriv):
                nd = int(round(wf[j] * 200000))
                if nd:
                    g = by_f[f]
                    parts.append(g.iloc[rng.integers(0, len(g), nd)][["priv_s003", "priv_r201", "priv_r209"]].to_numpy())
            pv = np.vstack(parts)
            sup_rows.append({
                "split": sp, "prior": pname, "flip_mle": float(F_FINE[imax]),
                "support_1_8_lo": float(s8.min()), "support_1_8_hi": float(s8.max()),
                "support_1_32_lo": float(s32.min()), "support_1_32_hi": float(s32.max()),
                "LR_f0": float(rel[0]), "LR_f0.01": float(rel[np.argmin(np.abs(F_FINE - 0.01))]),
                "LR_f0.015": float(rel[np.argmin(np.abs(F_FINE - 0.015))]),
                "LR_f0.02": float(rel[np.argmin(np.abs(F_FINE - 0.02))]),
                "post_mean_f": float((w * F_FINE).sum()), "post_q05_f": float(F_FINE[np.searchsorted(cdf, .05)]),
                "post_q95_f": float(F_FINE[min(np.searchsorted(cdf, .95), len(F_FINE) - 1)]),
                "s003_private_mean": float(pv[:, 0].mean()), "s003_private_q05": float(np.quantile(pv[:, 0], .05)),
                "s003_private_q50": float(np.quantile(pv[:, 0], .5)), "s003_private_q95": float(np.quantile(pv[:, 0], .95)),
                "P_s003_private_ge_0.999": float((pv[:, 0] >= 0.999 - 5e-6).mean()),
                "P_abs_pub_minus_priv_le_0.001": float((np.abs(pv[:, 0] - a.obs) <= 0.001).mean()),
                "r201_private_mean": float(pv[:, 1].mean()), "r209_private_mean": float(pv[:, 2].mean()),
                "d_r201_mean": float((pv[:, 1] - pv[:, 0]).mean()), "d_r209_mean": float((pv[:, 2] - pv[:, 0]).mean())})
    SUP = pd.DataFrame(sup_rows)
    run.save_table("support_and_private", SUP)
    SUP.to_csv(tables / "r421_support_and_private.csv", index=False)
    (tables / "r421_support_and_private.md").write_text(SUP.to_markdown(index=False, floatfmt=".5g"), encoding="utf-8")
    print(SUP.to_string(index=False), flush=True)

    # ---- conditional public predictions of r201 / r209 given S003's public (joint simulation)
    k_, n1_, d0, d1, d2_ = (ACC[:, i].astype(float) for i in range(5))
    nn = n1_ * (m - n1_)
    auc0, auc1, auc2 = 1 - d0 / (2 * nn), 1 - d1 / (2 * nn), 1 - d2_ / (2 * nn)
    pred_rows = []
    for pname in ("uniform_0_0.1", "default_aware"):
        pw = prior_w(pname) * Ltr                     # prior x train likelihood; the public enters via acceptance
        pik = np.array([(pw * stats.binom.pmf(k, m, F_FINE / 2)).sum() for k in range(a.kmax + 1)])
        for sp in ("random", "strat198", "strat197"):
            for wname, w in WINDOWS.items():
                inw = in_window(auc0, w)
                if sp == "random":
                    sel = inw
                    wt = pik[k_.astype(int)] / ndraw
                else:
                    n1s = int(sp[-3:])
                    sel = inw & (n1_ == n1s)
                    wt = pik[k_.astype(int)] / np.maximum(N1ALL[k_.astype(int), n1s], 1)
                if sel.sum() == 0:
                    continue
                W = wt[sel] / wt[sel].sum()
                kpost = np.bincount(k_[sel].astype(int), weights=W, minlength=a.kmax + 1)
                for name, av, bel in (("r201", auc1[sel], B006), ("r209", auc2[sel], B007)):
                    o = np.argsort(av)
                    cw = np.cumsum(W[o])

                    def q(t, o=o, cw=cw, av=av):
                        return float(av[o][min(np.searchsorted(cw, t), len(o) - 1)])
                    lo80, hi80 = bel["interval80"]
                    dv = av - auc0[sel]
                    pred_rows.append({
                        "prior": pname, "split": sp, "window": wname, "score": name, "n_rows": int(sel.sum()),
                        "eff_n": float(1 / (W ** 2).sum()), "P_k0": kpost[0], "P_k1": kpost[1], "P_k2": kpost[2],
                        "P_k3plus": float(kpost[3:].sum()), "s003_public_mean": float((W * auc0[sel]).sum()),
                        "public_mean": float((W * av).sum()),
                        "public_sd": float(np.sqrt((W * (av - (W * av).sum()) ** 2).sum())),
                        "q10": q(.10), "q50": q(.50), "q90": q(.90),
                        "diff_vs_s003_mean": float((W * dv).sum()), "P_higher_than_s003": float(W[dv > 0].sum()),
                        "P_equal_s003": float(W[dv == 0].sum()),
                        "P_in_belief_80pct_interval": float(W[(av >= lo80 - 5e-6) & (av <= hi80 + 5e-6)].sum()),
                        "P_below_file_problem_threshold": float(W[av < bel["bad_below"]].sum()),
                        "belief_point": bel["point"]})
    PP = pd.DataFrame(pred_rows)
    run.save_table("public_pred", PP)
    PP.to_csv(tables / "r421_public_pred.csv", index=False)
    (tables / "r421_public_pred.md").write_text(PP.to_markdown(index=False, floatfmt=".5g"), encoding="utf-8")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(PP[PP.prior == "uniform_0_0.1"].to_string(index=False), flush=True)

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for sp, ls in zip(SPLITS, ["-", "--", ":", "-."]):
            ax[0].plot(F_FINE, LK[f"Lcomb_{sp}_rel"], ls, label=sp)
        ax[0].set_xlim(0, 0.04)
        ax[0].axhline(1 / 8, color="#888", lw=0.8)
        ax[0].set_xlabel("flip_y")
        ax[0].set_ylabel("combined likelihood (rel. to max)")
        ax[0].set_title("flip_y likelihood: random vs stratified public split")
        ax[0].legend(fontsize=8)
        sub = PP[(PP.prior == "uniform_0_0.1") & (PP.window == "exact_round5")]
        for i, (name, g) in enumerate(sub.groupby("score")):
            for j, (_, r) in enumerate(g.iterrows()):
                yv = i + 0.15 * (j - 1)
                ax[1].plot([r.q10, r.q90], [yv, yv], color="#2a78d6")
                ax[1].plot(r.public_mean, yv, "o", color="#eb6834")
                ax[1].text(r.q90 + 2e-5, yv, r.split, fontsize=7, va="center")
        for i, bel in enumerate((B006, B007)):
            ax[1].plot(bel["interval80"], [i - 0.35] * 2, color="#0b0b0b", lw=3)
        ax[1].axvline(a.obs, color="#888", ls=":")
        ax[1].set_yticks([0, 1])
        ax[1].set_yticklabels(["S005 r201", "S006 r209"])
        ax[1].set_xlabel("public AUC given S003's 0.99923 (10-90 %, dot = mean); black = belief 80 %")
        fig.tight_layout()
        run.save_figure("r421_split_joint", fig)
        fig.savefig(figs / "r421_split_joint.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("figure failed:", e)

    exact = WP[WP.window == "exact_round5"].set_index("k_wrong_public")
    out = {"score_hashes": hashes, "posterior_fit": ginfo, "n_public_draws_per_k": ndraw,
           "n_test_boundary_0.1_0.9": int(((pi_te > .1) & (pi_te < .9)).sum()),
           "sum_pi_test": float(pi_te.sum()), "P_n1star_198_at_f0.003": float(p_n1s(0.003)[198]),
           "P_obs_given_k_exact_random": exact.P_random.round(7).tolist()[:6],
           "P_obs_given_k_exact_strat198": exact.P_strat198.round(7).tolist()[:6],
           "P_obs_given_k_exact_strat197": exact.P_strat197.round(7).tolist()[:6],
           "support": SUP[SUP.prior == "uniform_0_0.1"].set_index("split")[
               ["flip_mle", "support_1_8_hi", "support_1_32_hi", "LR_f0", "LR_f0.01", "post_mean_f", "s003_private_mean",
                "s003_private_q05", "s003_private_q95", "P_s003_private_ge_0.999"]].round(6).to_dict(orient="index"),
           "public_pred_exact": PP[(PP.prior == "uniform_0_0.1") & (PP.window == "exact_round5")].set_index(
               ["split", "score"])[["public_mean", "q10", "q90", "P_higher_than_s003", "P_equal_s003",
                                    "P_in_belief_80pct_interval"]].round(6).reset_index().to_dict(orient="records"),
           "seconds": time.time() - t_start}
    run.finish(extra_summary=out)
    for k_o, v_o in out.items():
        print(f"{k_o}: {v_o}")
