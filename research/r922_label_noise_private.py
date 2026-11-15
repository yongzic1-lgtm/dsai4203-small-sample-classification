"""R922 (verifier "gen") - label-noise rate and the private-LB distribution of S003, on the REAL test features.

Independent re-implementation (no lib_gen_replica / lib_mix code) of the claims in research/reports/gen.md (R150, R153,
R156) and mix.md (R103, R106): "the public 0.99923 implies ~0.2 % wrong labels (flip_y ~ 0.004) and an expected private
AUC of S003 ~ 0.998", and of belief B004 (private >= 0.999 with |public - private| <= 0.001, 70 %).

Model of the labels (make_classification semantics): the clean class c_i of a row has the Bayes posterior pi_i given the
features (approximated by a converged label-free K=2 Gaussian mixture on the structured block, orientation from the train
labels); with probability flip_y the observed label is replaced by a fair coin, i.e. P(y_i != c_i) = flip_y / 2,
independently of everything else. Features are FIXED (the real 19,750 test rows); only labels and the unknown public
subset (395 random test rows) are random. No replica data sets are needed for this part.

  1. S003 scores: re-created on the VM with ssc TransductiveGMM("highvar", 2) + official 5x20 CV fold mean, written with
     the same "%.8f" format as the submitted csv; the hash of the formatted values is compared with the local
     submissions/S003_*.csv (hash c70057e06406a2d6 over the 19,750 target strings). The ranking that Kaggle scored is
     heavily tied (8918 rows share the minimum, 6416 the maximum), which matters for the public-score likelihood.
  2. Train-label evidence: P(y_train | flip_y) = prod_i [(1 - f) pi_i(y_i) + f / 2]; count of confident contradictions.
  3. Public-score likelihood P(public AUC rounds to 0.99923 | k wrong labels among the 395 public rows), k = 0..KMAX,
     by simulation (random public subset, clean labels ~ Bern(pi), k uniformly chosen wrong rows, AUC with mid-ranks on
     the submitted scores). L_pub(f) = sum_k Binom(k; 395, f/2) P(obs | k). Several observation windows are reported
     (exact 5-decimal rounding, +-5e-5, +-1.5e-4 as in R153/R156, +-4e-4 as in R106).
  4. Private AUC (19,355 rows) distribution of S003 for each f on a grid (simulation).
  5. Posterior over f for several priors -> posterior predictive of the private AUC, P(B004).
  Sensitivity: posterior from S003's own mixture instead of the converged one; tempered posteriors (log-odds x 0.8, 1.25).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vgen --high-mem -- research/r922_label_noise_private.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.mixture import GaussianMixture  # noqa: E402

from ssc import data, paths  # noqa: E402
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
p.add_argument("--n-public-draws", type=int, default=6_000_000)
p.add_argument("--n-public-draws-sens", type=int, default=1_500_000)
p.add_argument("--n-private-draws", type=int, default=3000)
p.add_argument("--batch", type=int, default=4000)
p.add_argument("--quick", action="store_true", help="tiny smoke test")
a = p.parse_args()
if a.quick:
    a.n_public_draws, a.n_public_draws_sens, a.n_private_draws, a.batch = 16000, 8000, 40, 2000

S003_VALUES_HASH = "c70057e06406a2d6"   # sha256[:16] of "\n".join(target strings) of submissions/S003_*.csv
CAP = 1200                              # 2*D cap for the public histogram (D = discordant pairs incl. half ties)
F_PRIV = np.r_[np.round(np.arange(0, 0.0301, 0.0005), 4), [0.035, 0.04, 0.05, 0.06, 0.08, 0.1]]
F_FINE = np.round(np.arange(0, 0.10001, 0.00025), 5)
SEED_OFF = {"conv": 1000, "s003mix": 2000, "temp0.8": 3000, "temp1.25": 4000}
WINDOWS = {"exact_round5": 5e-6, "pm5e-5": 5e-5, "pm1.5e-4_R153": 1.5e-4, "pm4e-4_R106": 4e-4}


# ----------------------------------------------------------------------------- helpers
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


def public_chunk(seed, n_draws, B, pi_sorted, gid, m, kmax):
    rng = np.random.default_rng([922, int(seed)])
    N = len(pi_sorted)
    nb = (m + 1) * (CAP + 1)
    hist = np.zeros((kmax + 1) * nb, np.int64)
    ar = np.arange(m)
    done = 0
    while done < n_draws:
        b = min(B, n_draws - done)
        pos = subsets(rng, b, N, m)
        g = gid[pos]
        c = rng.random((b, m)) < pi_sorted[pos]
        rk = np.argsort(np.argsort(rng.random((b, m)), axis=1), axis=1)
        is_start = np.ones((b, m), bool)
        is_start[:, 1:] = g[:, 1:] != g[:, :-1]
        start = np.maximum.accumulate(np.where(is_start, ar, 0), axis=1)
        is_end = np.ones((b, m), bool)
        is_end[:, :-1] = g[:, 1:] != g[:, :-1]
        end = np.minimum.accumulate(np.where(is_end, ar, m)[:, ::-1], axis=1)[:, ::-1]
        for k in range(kmax + 1):
            y = c ^ (rk < k)
            neg = (~y).astype(np.int32)
            cneg = np.cumsum(neg, axis=1)
            below = np.take_along_axis(cneg - neg, start, 1)
            upto = np.take_along_axis(cneg, end, 1)
            u2 = ((2 * below + (upto - below)) * y).sum(1)          # 2 * Mann-Whitney U (ties 1/2)
            n1 = y.sum(1)
            d2 = 2 * n1 * (m - n1) - u2
            ok = (n1 > 0) & (n1 < m)
            cell = k * nb + n1[ok] * (CAP + 1) + np.minimum(d2[ok], CAP)
            hist += np.bincount(cell, minlength=hist.size)
        done += b
    return hist.reshape(kmax + 1, m + 1, CAP + 1)


def run_public(pi_sorted, gid, n_draws, label):
    t0 = time.time()
    per = int(np.ceil(n_draws / (a.n_jobs * 4)))
    seeds = range(SEED_OFF[label], SEED_OFF[label] + a.n_jobs * 4)
    hs = Parallel(n_jobs=a.n_jobs)(delayed(public_chunk)(s, per, a.batch, pi_sorted, gid, a.n_public, a.kmax)
                                   for s in seeds)
    H = np.sum(hs, axis=0)
    print(f"public sim [{label}]: {H.sum() // (a.kmax + 1):,} draws per k in {time.time() - t0:.0f} s", flush=True)
    return H


def window_probs(H, m):
    """P(public AUC in window | k) for every window, plus P(AUC <= obs)."""
    n1 = np.arange(m + 1)[:, None].astype(float)
    d = np.arange(CAP + 1)[None, :] / 2.0
    nn = n1 * (m - n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = 1.0 - d / nn
    auc[:, CAP] = -1.0                                          # capped bin: far below the window
    tot = H.sum(axis=(1, 2)).astype(float)
    out = {}
    for name, w in WINDOWS.items():
        if name == "exact_round5":
            mask = (auc >= a.obs - 5e-6 - 1e-12) & (auc < a.obs + 5e-6 - 1e-12)
        else:
            mask = np.abs(auc - a.obs) <= w + 1e-12
        out[name] = (H * mask[None]).sum(axis=(1, 2)) / tot
        out[name + "_hits"] = (H * mask[None]).sum(axis=(1, 2))
    out["p_le_obs"] = (H * (auc <= a.obs + 5e-6)[None]).sum(axis=(1, 2)) / tot
    out["p_eq_1"] = H[:, :, 0].sum(1) / tot
    return pd.DataFrame(out, index=pd.Index(range(H.shape[0]), name="k_wrong_public"))


def private_chunk(seed, fs, n_draws, pi_sorted, starts, m_pub, B=100):
    rng = np.random.default_rng([9221, int(seed)])
    N = len(pi_sorted)
    rows = []
    for f in fs:
        done = 0
        while done < n_draws:
            b = min(B, n_draws - done)
            c = rng.random((b, N)) < pi_sorted
            w = rng.random((b, N)) < f / 2
            y = c ^ w
            priv = np.ones((b, N), bool)
            pub = subsets(rng, b, N, m_pub)
            priv[np.arange(b)[:, None], pub] = False
            P = np.add.reduceat((y & priv).astype(np.int32), starts, axis=1)
            Q = np.add.reduceat((~y & priv).astype(np.int32), starts, axis=1)
            below = np.cumsum(Q, axis=1) - Q
            u2 = (P * (2 * below + Q)).sum(1).astype(float)
            n1, n0 = P.sum(1).astype(float), Q.sum(1).astype(float)
            auc = u2 / (2 * n1 * n0)
            nw = (w & priv).sum(1)
            for v, k in zip(auc, nw):
                rows.append((f, v, int(k)))
            done += b
    return rows


def run_private(pi_sorted, starts, fs, n_draws, label):
    t0 = time.time()
    jobs = []
    # split draws, not f values, so every worker sees all f (balanced)
    nper = int(np.ceil(n_draws / (a.n_jobs)))
    for j in range(a.n_jobs):
        jobs.append(delayed(private_chunk)(SEED_OFF[label] + j, fs, nper, pi_sorted, starts, a.n_public))
    res = Parallel(n_jobs=a.n_jobs)(jobs)
    df = pd.DataFrame([r for x in res for r in x], columns=["flip_y", "private_auc", "n_wrong_private"])
    print(f"private sim [{label}]: {len(df):,} AUCs in {time.time() - t0:.0f} s", flush=True)
    return df


def fit_posterior(X_all, y_train, n_tr, n_init=10):
    """Converged label-free K=2 full GMM on the PCA basis of the linearly structured columns (own implementation)."""
    Z = (X_all - X_all.mean(0)) / X_all.std(0)
    C = (Z.T @ Z) / len(Z)
    off = np.abs(C - np.diag(np.diag(C))).max(1)
    S = np.where(off > 0.1)[0]
    ev, V = np.linalg.eigh(C[np.ix_(S, S)])
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


# ----------------------------------------------------------------------------- main
tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r922_label_noise_private", "obs_public": a.obs, "n_public": a.n_public, "kmax": a.kmax,
       "n_public_draws": a.n_public_draws, "n_public_draws_sens": a.n_public_draws_sens,
       "n_private_draws_per_f": a.n_private_draws, "f_private_grid": F_PRIV.tolist(), "windows": WINDOWS,
       "posterior": "converged K=2 full GMM, PCA basis of structured columns (|corr|>0.1), tol 1e-10, reg 1e-10",
       "s003_reproduction": "ssc TransductiveGMM('highvar', 2) official CV fold mean, %.8f", "quick": a.quick,
       "verifies": ["R150", "R153", "R156", "R103", "R106", "B004"]}
HYP = ("(a) Wrong-label counts drive the public score: with the tied S003 ranking a single confidently wrong label "
       "costs ~0.003 public AUC, while clean-label boundary rows cost ~1e-4, so 0.99923 (about 30 discordant pairs) "
       "is unlikely under flip_y = 0 but the likelihood is nearly flat for flip_y in ~0.002-0.012 and only falls "
       "slowly above; (b) the train labels add a factor ~exp(-125 f); (c) the posterior mean of flip_y depends strongly "
       "on the prior (gen's 4-point grid truncated at 0.01 understates the uncertainty); the private AUC is ~1 - f/2 "
       "with sd ~3-6e-4, so a 90 % predictive interval is roughly 0.994-0.9995 and B004 (private >= 0.999) has "
       "probability well below 70 % (expected 15-35 %).")

with Run("r922_label_noise_private", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    t_start = time.time()
    ds = data.load(a.data)
    y_tr = ds.y
    n_tr = len(y_tr)
    X_all = unlabeled_features(a.data)

    # ---- 1. S003 scores (official CV fold mean, formatted like the submission)
    scheme = Scheme.from_dict(yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))["official_cv"])
    if a.quick:
        scheme = Scheme(kind="kfold", n_splits=5, n_repeats=1, seed=2026)
    splits = make_splits(y_tr, scheme)
    est = TransductiveGMM("highvar", 2, source=a.data)
    cv = cross_validate(est, ds.X, y_tr, splits, scheme, X_test=ds.X_test, n_jobs=1)
    tp = cv.test_pred
    strs = [f"{v:.8f}" for v in tp]
    vh = hashlib.sha256("\n".join(strs).encode()).hexdigest()[:16]
    s003 = np.array([float(v) for v in strs])
    s003_match = vh == S003_VALUES_HASH
    print(f"S003 reproduction: cv auc {cv.fold_scores.auc.mean():.6f}; values hash {vh} "
          f"({'MATCHES' if s003_match else 'DIFFERS FROM'} submitted {S003_VALUES_HASH})", flush=True)
    est.fit(ds.X, y_tr)
    R_own = est.gmm_.predict_proba(est._transform(np.asarray(X_all)))
    k1 = int(np.argmax(est.component_rate_))
    lo_s003 = np.log(np.clip(R_own[:, k1], 1e-300, 1)) - np.log(np.clip(R_own[:, 1 - k1], 1e-300, 1))

    # ---- 2. posterior (converged, independent) and train-label evidence
    lo, ginfo = fit_posterior(np.asarray(X_all), y_tr, n_tr, n_init=1 if a.quick else 10)
    pi_all = sigmoid(lo)
    pi_tr, pi_te = pi_all[:n_tr], pi_all[n_tr:]
    pi_s003_te = sigmoid(lo_s003[n_tr:])
    agree = {"max_abs_diff_pi_test_conv_vs_s003": float(np.abs(pi_te - pi_s003_te).max()),
             "n_test_absdiff_gt_0.01": int((np.abs(pi_te - pi_s003_te) > 0.01).sum()),
             "spearman_s003score_vs_pi_on_boundary": float(stats.spearmanr(s003[(pi_te > .01) & (pi_te < .99)],
                                                                         pi_te[(pi_te > .01) & (pi_te < .99)])[0]),
             "n_test_boundary_0.1_0.9": int(((pi_te > .1) & (pi_te < .9)).sum()),
             "n_test_boundary_0.01_0.99": int(((pi_te > .01) & (pi_te < .99)).sum()),
             "sum_pi_all_rows": float(pi_all.sum()), "sd_sum_pi": float(np.sqrt((pi_all * (1 - pi_all)).sum())),
             "expected_clean_bayes_error_test": float(np.minimum(pi_te, 1 - pi_te).mean())}
    p_obs_tr = np.where(y_tr == 1, pi_tr, 1 - pi_tr)
    contra = pd.DataFrame({"train_row": np.arange(n_tr), "y": y_tr, "pi_class1": pi_tr, "p_observed_label": p_obs_tr,
                           "pi_class1_s003_mixture": sigmoid(lo_s003[:n_tr])}).sort_values("p_observed_label")
    run.save_table("train_label_check", contra)
    contra.head(20).to_csv(tables / "r922_train_label_check.csv", index=False)
    thr_counts = {f"n_train_p_obs_lt_{t:g}": int((p_obs_tr < t).sum()) for t in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001)}
    ll_tr = np.array([np.log((1 - f) * p_obs_tr + f / 2).sum() for f in F_FINE])
    print("train label check:", thr_counts, flush=True)

    # ---- sorted structures of the submitted ranking
    order = np.argsort(s003, kind="stable")
    ss = s003[order]
    new = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    tie_info = {"n_unique_scores": int(new.sum()), "n_at_min": int((s003 == ss[0]).sum()),
                "n_at_max": int((s003 == ss[-1]).sum()),
                "pi_mean_at_min": float(pi_te[s003 == ss[0]].mean()), "pi_mean_at_max": float(pi_te[s003 == ss[-1]].mean())}
    print("ties:", tie_info, flush=True)

    # ---- 3. public-score likelihood given k wrong public labels
    variants = {"conv": pi_te[order]}
    H = {"conv": run_public(variants["conv"], gid, a.n_public_draws, "conv")}
    sens = {"s003mix": pi_s003_te[order], "temp0.8": sigmoid(0.8 * lo[n_tr:])[order],
            "temp1.25": sigmoid(1.25 * lo[n_tr:])[order]}
    for name, v in sens.items():
        H[name] = run_public(v, gid, a.n_public_draws_sens, name)
    WP = {}
    for name, h in H.items():
        wp = window_probs(h, a.n_public)
        wp.insert(0, "posterior_variant", name)
        WP[name] = wp
    wp_all = pd.concat(WP.values()).reset_index()
    run.save_table("public_window_probs", wp_all)
    wp_all.to_csv(tables / "r922_public_window_probs.csv", index=False)
    print(wp_all.to_string(), flush=True)

    def lik_pub(f, wp, col):
        kk = np.arange(a.kmax + 1)
        return float((stats.binom.pmf(kk, a.n_public, f / 2) * wp[col].to_numpy()).sum())

    lik_rows = []
    for f, llt in zip(F_FINE, ll_tr):
        row = {"flip_y": f, "loglik_train": llt}
        for name, wp in WP.items():
            for col in list(WINDOWS) + ["p_le_obs"]:
                row[f"Lpub_{name}_{col}"] = lik_pub(f, wp, col)
        lik_rows.append(row)
    LK = pd.DataFrame(lik_rows)
    LK["Ltrain_rel"] = np.exp(LK.loglik_train - LK.loglik_train.max())
    LK["Lcomb_main"] = LK.Ltrain_rel * LK["Lpub_conv_exact_round5"]
    LK["Lcomb_main_rel"] = LK.Lcomb_main / LK.Lcomb_main.max()
    run.save_table("likelihood_fine", LK)
    LK.to_csv(tables / "r922_likelihood_fine.csv", index=False)

    # ---- 4. private AUC distribution per f
    PR = run_private(pi_te[order], starts, F_PRIV, a.n_private_draws, "conv")
    PRs = run_private(pi_s003_te[order], starts, np.array([0.0, 0.005, 0.01, 0.02]), max(a.n_private_draws // 3, 20),
                      "s003mix")
    psum = PR.groupby("flip_y").private_auc.agg(
        mean="mean", sd="std", q05=lambda v: v.quantile(.05), q50="median", q95=lambda v: v.quantile(.95),
        p_ge_0999=lambda v: float((v >= 0.999 - 5e-6).mean()), p_ge_0998=lambda v: float((v >= 0.998 - 5e-6).mean()),
        p_within_0001_of_public=lambda v: float((np.abs(v - a.obs) <= 0.001).mean())).reset_index()
    psum["mean_wrong_private"] = PR.groupby("flip_y").n_wrong_private.mean().to_numpy()
    psum["approx_1_minus_f_over_2"] = 1 - psum.flip_y / 2
    pss = PRs.groupby("flip_y").private_auc.agg(mean_s003mix="mean", sd_s003mix="std").reset_index()
    run.save_table("private_by_flip", psum)
    run.save_table("private_by_flip_s003mix", pss)
    psum.to_csv(tables / "r922_private_by_flip.csv", index=False)
    print(psum.to_string(), flush=True)
    print(pss.to_string(), flush=True)

    # ---- 5. priors -> posterior over f -> posterior predictive of the private AUC
    fgrid = F_FINE
    L = LK.Lcomb_main.to_numpy()
    dfine = fgrid[1] - fgrid[0]

    def prior_weights(name):
        w = np.zeros_like(fgrid)
        if name == "uniform_0_0.1":
            w[:] = 1.0
        elif name == "uniform_0_0.03":
            w[fgrid <= 0.03 + 1e-12] = 1.0
        elif name == "loguniform_1e-4_0.1":
            m = fgrid >= 1e-4
            w[m] = 1.0 / fgrid[m]
        elif name == "gen_grid4":
            for f0 in (0.0, 0.0025, 0.005, 0.01):
                w[np.argmin(np.abs(fgrid - f0))] = 1.0
        elif name == "default_aware":   # 0.2 author set flip_y=0; 0.5 sklearn default 0.01; 0.3 uniform (0, 0.1]
            w[1:] = 0.3 / (len(fgrid) - 1)
            w[0] += 0.2
            w[np.argmin(np.abs(fgrid - 0.01))] += 0.5
            return w
        return w / w.sum()

    priv_by_f = {f: g.private_auc.to_numpy() for f, g in PR.groupby("flip_y")}
    fpriv = np.array(sorted(priv_by_f))
    pri_rows = []
    pred_samples = {}
    rng = np.random.default_rng(92299)
    for pname in ("uniform_0_0.1", "uniform_0_0.03", "loguniform_1e-4_0.1", "gen_grid4", "default_aware"):
        w = prior_weights(pname) * L
        w = w / w.sum()
        cdf = np.cumsum(w)
        q = lambda t: float(fgrid[min(np.searchsorted(cdf, t), len(fgrid) - 1)])  # noqa: E731
        # posterior predictive: map each fine f to the nearest simulated private grid value
        near = np.abs(fgrid[:, None] - fpriv[None, :]).argmin(1)
        wf = np.bincount(near, weights=w, minlength=len(fpriv))
        draws = []
        for j, f in enumerate(fpriv):
            nd = int(round(wf[j] * 200000))
            if nd:
                draws.append(rng.choice(priv_by_f[f], nd, replace=True))
        pv = np.concatenate(draws)
        pred_samples[pname] = pv
        pri_rows.append({"prior": pname, "post_mean_flip": float((w * fgrid).sum()), "post_median_flip": q(0.5),
                         "post_q05_flip": q(0.05), "post_q95_flip": q(0.95), "post_p_flip_eq_0": float(w[0]),
                         "post_p_flip_le_0.002": float(w[fgrid <= 0.002 + 1e-12].sum()),
                         "post_p_flip_ge_0.01": float(w[fgrid >= 0.01 - 1e-12].sum()),
                         "pred_private_mean": float(pv.mean()), "pred_private_sd": float(pv.std()),
                         "pred_private_q05": float(np.quantile(pv, .05)), "pred_private_q50": float(np.quantile(pv, .5)),
                         "pred_private_q95": float(np.quantile(pv, .95)),
                         "P_B004_private_ge_0.999": float((pv >= 0.999 - 5e-6).mean()),
                         "P_private_ge_0.998": float((pv >= 0.998 - 5e-6).mean()),
                         "P_abs_public_minus_private_le_0.001": float((np.abs(pv - a.obs) <= 0.001).mean())})
    PRI = pd.DataFrame(pri_rows)
    run.save_table("posterior_by_prior", PRI)
    PRI.to_csv(tables / "r922_posterior_by_prior.csv", index=False)
    (tables / "r922_posterior_by_prior.md").write_text(PRI.to_markdown(index=False, floatfmt=".5f"), encoding="utf-8")
    print(PRI.to_string(), flush=True)

    # likelihood-ratio table at named flip values
    named = [0.0, 0.001, 0.0025, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
    lr = LK.set_index("flip_y").loc[[fgrid[np.argmin(np.abs(fgrid - f0))] for f0 in named]]
    imax = int(np.argmax(L))
    lr_tab = pd.DataFrame({"flip_y": named, "L_train_rel": lr.Ltrain_rel.to_numpy(),
                           "L_public_exact": lr["Lpub_conv_exact_round5"].to_numpy(),
                           "L_public_R153window": lr["Lpub_conv_pm1.5e-4_R153"].to_numpy(),
                           "L_public_R106window": lr["Lpub_conv_pm4e-4_R106"].to_numpy(),
                           "L_public_exact_s003mix": lr["Lpub_s003mix_exact_round5"].to_numpy(),
                           "L_public_exact_temp0.8": lr["Lpub_temp0.8_exact_round5"].to_numpy(),
                           "L_public_exact_temp1.25": lr["Lpub_temp1.25_exact_round5"].to_numpy(),
                           "L_comb_rel_to_max": lr.Lcomb_main_rel.to_numpy()})
    run.save_table("likelihood_named", lr_tab)
    lr_tab.to_csv(tables / "r922_likelihood_named.csv", index=False)
    (tables / "r922_likelihood_named.md").write_text(lr_tab.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")
    print(lr_tab.to_string(), flush=True)
    # 'support' interval: f with combined likelihood >= 1/8 and >= 1/32 of the maximum
    sup8 = fgrid[LK.Lcomb_main_rel.to_numpy() >= 1 / 8]
    sup32 = fgrid[LK.Lcomb_main_rel.to_numpy() >= 1 / 32]

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        ax[0].plot(fgrid, LK.Ltrain_rel, label="train labels")
        lp_ = LK["Lpub_conv_exact_round5"]
        ax[0].plot(fgrid, lp_ / lp_.max(), label="public 0.99923 (exact rounding)")
        lp2 = LK["Lpub_conv_pm1.5e-4_R153"]
        ax[0].plot(fgrid, lp2 / lp2.max(), ":", label="public, R153 window")
        ax[0].plot(fgrid, LK.Lcomb_main_rel, "k", lw=2, label="combined")
        ax[0].set_xlabel("flip_y"); ax[0].set_ylabel("relative likelihood"); ax[0].legend(fontsize=8)
        ax[0].set_title("Likelihood of flip_y")
        ax[1].plot(psum.flip_y, psum["mean"], "k")
        ax[1].fill_between(psum.flip_y, psum.q05, psum.q95, alpha=.3)
        ax[1].axhline(0.999, color="r", ls="--", lw=1); ax[1].axhline(a.obs, color="g", ls=":", lw=1)
        ax[1].set_xlim(0, 0.03); ax[1].set_ylim(0.982, 1.0005)
        ax[1].set_xlabel("flip_y"); ax[1].set_ylabel("S003 private AUC (mean, 90%)"); ax[1].set_title("Private AUC given flip_y")
        for pname, pv in pred_samples.items():
            ax[2].hist(pv, bins=np.linspace(0.990, 1.0001, 120), histtype="step", density=True, label=pname)
        ax[2].axvline(0.999, color="r", ls="--", lw=1)
        ax[2].set_xlabel("S003 private AUC"); ax[2].legend(fontsize=7); ax[2].set_title("Posterior predictive")
        fig.tight_layout()
        run.save_figure("r922_label_noise", fig)
        fig.savefig(figs / "r922_label_noise.png", dpi=150, bbox_inches="tight")
    except Exception as e:  # noqa: BLE001
        print("figure failed:", e)

    main = PRI.set_index("prior")
    out = {"s003_values_hash": vh, "s003_reproduction_matches_submission": bool(s003_match),
           "s003_cv_auc_reproduced": float(cv.fold_scores.auc.mean()), "posterior_fit": ginfo, "posterior_agreement": agree,
           "ties": tie_info, **thr_counts,
           "flip_mle_combined": float(fgrid[imax]), "flip_support_1_8": [float(sup8.min()), float(sup8.max())],
           "flip_support_1_32": [float(sup32.min()), float(sup32.max())],
           "LR_flip0_vs_mle": float(L[0] / L[imax]), "LR_flip0.01_vs_mle": float(L[np.argmin(np.abs(fgrid - 0.01))] / L[imax]),
           "LR_flip0.02_vs_mle": float(L[np.argmin(np.abs(fgrid - 0.02))] / L[imax]),
           "P_obs_given_k_exact": WP["conv"]["exact_round5"].round(6).tolist(),
           "posterior_by_prior": main.round(6).to_dict(orient="index"), "seconds": time.time() - t_start}
    run.finish(extra_summary=out)
    for k_, v_ in out.items():
        print(f"{k_}: {v_}")
