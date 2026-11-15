"""Replica generator for the DSAI4203 small-sample-classification-challenge data (agent "gen", R150-R199).

WHAT THE REAL DATA ARE (evidence: runs 20261022-064606_r150_taxonomy, 20261022-065325_r152_calibration,
20261022-065605_r154_class_structure; report research/reports/gen.md)
    sklearn.datasets.make_classification(n_samples=20000, n_features=150, n_informative=12,
        n_redundant=8, n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None,
        hypercube=True, class_sep ~ 0.71 (95%: 0.61-0.81), flip_y ~ 0.0025-0.005 (R156; 0 is unlikely),
        shift=0.0, scale=1.0, shuffle=True)
    followed by a per-column affine map  x_j <- a_j * x_j + b_j  with a_j ~ U(0.5, 3), b_j ~ U(-5, 5),
    values rounded to 6 decimals, and a random 250-row train / 19,750-row test split.
    Real columns: informative {6,15,30,46,52,61,64,80,100,112,142,144} (centroids differ only in
    46, 80, 100, 112: hamming distance 4); redundant (exact linear combinations of the informative ones)
    {8,17,26,49,57,67,77,98}; the other 130 are independent N(b_j, a_j^2) noise.
    Validation (20261022-072608_r151_replica_validation): parametric replicas reproduce the std profile,
    correlation spectrum, block size/rank, BIC, boundary fraction and the R003 model ranking (97%).
    Simulation results (20261022-073058_r153_param, 20261022-075419_r153_mc_cs0.71_h4): Bayes private AUC
    0.99995 without label noise, ~1 - flip_y/2 with it; gmm2_hv is Bayes-optimal within 3e-7.

TWO REPLICA VARIANTS
    (a) sample_parametric(fit_parametric(X_all, y_train), seed)   parametric bootstrap of the REAL data:
        per-class Gaussians fitted (label-free GMM, labels only orient the components) in the 12-dim
        informative coordinates, the 8 redundant columns rebuilt with the real exact linear map, and the
        130 noise columns ~ N(real mean, real std^2); same column positions and scales as the real data.
    (b) sample_make_classification(seed, class_sep=..., flip_y=...)   the generator hypothesis itself:
        sklearn's own make_classification (shuffle=False, captured internals) + random row/column
        permutation + the affine map above. New random A_k, B, centroids, scales for every seed.
    Both return a Replica with known test labels and the TRUE Bayes log-odds (computed from the true
    generating parameters), a Kaggle-like public subset (395 test rows) and the column roles.

API
    Structure (label-free; uses only the 20,000 x 150 feature matrix)
      identify_structure(X_all, ...) -> dict      feature roles, rank, the 8 dependencies, hypercube test,
                                                   scale/class_sep/hamming estimates, basis search table
      fit_parametric(X_all, y_train=None, struct=None) -> ParametricStructure
    Sampling
      sample_parametric(ps, seed, flip_y=0.0, ...) -> Replica
      sample_make_classification(seed, class_sep=0.7, flip_y=0.0, n_informative=12, n_redundant=8,
            n_repeated=0, n_clusters_per_class=1, hypercube=True, scale_range=(0.5, 3.0),
            shift_range=(-5.0, 5.0), hamming=None, ...) -> Replica
      Replica: X_train (250,150), y_train, X_test (19750,150), y_test (observed labels, flips included),
               cluster_train/cluster_test (clean class), bayes_train/bayes_test (true log-odds of clean
               class 1), public_idx (positions in X_test), roles (per column), params (dict);
               .X_all, .private_mask, .frames() -> (train_df, test_df) in the Kaggle csv layout.
    Procedures trained like ours (array based; same settings as src/ssc and research/r003)
      highvar_columns(X_all, ratio=1.5); structured_columns(X_all) -> (cols, maxcorr, r2)
      UnsupervisedGMM(X_all, columns="highvar"|"all"|"structured"|"informative"|idx, K=2,
                      pca_dims=None|int|"rank", tol=1e-3).fit()
          .R (responsibilities of all rows), .bic, .loglik, .scores(train_idx, y_train) (= ssc TransductiveGMM
          score), .responsibilities(X_new)
      GMM_SPECS: named variants used in R153 (gmm2_hv, gmm2_hv_tight, gmm2_hv_pca, gmm4_hv, gmm2_all,
                 gmm2_struct_pca, gmm2_struct)
      make_supervised(name, X_all)  name in l1_all, lda_hv, qda_hv, logreg_hv (sklearn pipelines)
      cv_ranking(rep, models, n_repeats=4) -> list of dicts (R003-style CV AUC + test AUC per model)
      TransductiveGMMTight(columns="informative", tol=1e-8, ...)  sklearn estimator for the official CV on the
          real data (like ssc TransductiveGMM, plus the informative-column rule and a converged EM; R155)
    Metrics / diagnostics
      auc_fast(y, s); bayes_auc(rep, subset="private"|"public"|"all", labels="observed"|"clean");
      lb_resample(y_test, scores, n_public=395, n_draws) -> public/private AUC per random public subset;
      summary_stats(X_all, y_train, informative_cols=None) (R151 statistics); within_class_mardia(L, post);
      make_classification_with_truth(...) -> X, y, {centroids, A, B, cluster_id, sizes, Z} (bit-identical to
          sklearn's make_classification(shift=0, scale=1, shuffle=False); captured via a recording RandomState)

    Label noise: replicas are drawn with flip_y=0 by default; pass flip_y=... (sklearn semantics: a row is
    re-labelled uniformly at random with prob flip_y, i.e. ~flip_y/2 labels actually change). R153 instead
    draws clean replicas and applies nested flips itself so that one set of mixture fits serves all levels.

USAGE
    import sys; sys.path.insert(0, "research")
    from ssc.transductive import unlabeled_features; from ssc import data
    import lib_gen_replica as G
    X_all = unlabeled_features("raw"); y = data.load("raw").y
    ps = G.fit_parametric(X_all, y)                     # ~10 s
    rep = G.sample_parametric(ps, seed=1)               # one replica competition with known labels
    rep2 = G.sample_make_classification(seed=1, class_sep=0.7, hamming=4)
    print(G.bayes_auc(rep), G.bayes_auc(rep2))

No archive writes happen here; scripts r15x_*.py call these functions and archive the results.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.mixture import GaussianMixture

N_FEATURES = 150
N_TOTAL = 20000
N_TRAIN = 250
N_PUBLIC = 395            # 2 % of 19,750 test rows
DECIMALS = 6              # the Kaggle csv stores 6 decimals
REAL_INFORMATIVE = (6, 15, 30, 46, 52, 61, 64, 80, 100, 112, 142, 144)
REAL_REDUNDANT = (8, 17, 26, 49, 57, 67, 77, 98)


# ============================================================================ metrics
def auc_fast(y, s) -> float:
    """ROC AUC via average ranks (ties count 1/2), identical to sklearn.metrics.roc_auc_score."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ============================================================================ structure (label-free)
def _weighted_moments(X, w):
    w = w / w.sum()
    m = w @ X
    Xc = X - m
    C = (Xc * w[:, None]).T @ Xc
    return m, C


def structured_columns(X_all, corr_thr: float = 0.05, r2_thr: float = 0.5):
    """Columns that are linearly related to some other column (label-free).
    Null distribution at n=20,000: |corr| ~ N(0, 0.0071), max over 11k pairs ~ 0.03."""
    X = np.asarray(X_all, float)
    n, p = X.shape
    Z = (X - X.mean(0)) / X.std(0)
    C = (Z.T @ Z) / n
    off = np.abs(C - np.diag(np.diag(C)))
    maxcorr = off.max(1)
    P = np.linalg.inv(C + 1e-9 * np.eye(p))
    r2 = 1.0 - 1.0 / np.diag(P)          # R^2 of column j regressed on all others
    S = np.where((maxcorr > corr_thr) | (r2 > r2_thr))[0]
    return S, maxcorr, r2


def identify_structure(X_all, corr_thr: float = 0.05, z_thr: float = 4.0, rank_tol: float = 1e-9,
                       gmm_n_init: int = 5, random_state: int = 0, max_combos: int = 300_000,
                       top_k_table: int = 25) -> dict:
    """Label-free reverse engineering of a make_classification-type matrix.

    1. structured set S (linearly related columns) vs independent noise columns;
    2. numerical rank r of S (= n_informative), repeated (|corr| ~ 1) columns;
    3. 2-component full-covariance GMM in the PCA basis of S -> posterior p (orientation arbitrary);
    4. basis search: informative set = the r-subset of S that maximises log det of the pooled
       within-class correlation matrix (A_k^T A_k with iid U(-1,1) entries has expectation (r/3) I, so the
       true informative columns are nearly uncorrelated within class, while every redundant column is a
       dense combination of them); all C(|S|, r) subsets are scored (125,970 for 20 choose 12);
    5. hypercube test: per-column class-mean difference z-scores; under hypercube=True the informative
       columns whose two centroids agree have EXACTLY zero mean difference (|z| ~ |N(0,1)|), the others
       differ by 2*class_sep (unscaled), redundant ones almost surely have non-zero differences;
    6. scales a_i = sqrt(pooled within-class var / (r/3)), redundant coefficients B = beta * a_I / a_R
       (should lie in [-1, 1]), class_sep = mean |delta u| / 2 over the active (differing) coordinates.
    """
    X = np.asarray(X_all, float)
    n, p = X.shape
    mu, sd = X.mean(0), X.std(0)
    S, maxcorr, r2 = structured_columns(X, corr_thr)
    N = np.setdiff1d(np.arange(p), S)
    Zs = (X[:, S] - mu[S]) / sd[S]
    Cs = (Zs.T @ Zs) / n
    ev = np.sort(np.linalg.eigvalsh(Cs))[::-1]
    r = int((ev > rank_tol * ev[0]).sum())

    # repeated columns (exact duplicates up to affine map)
    repeated = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if abs(Cs[i, j]) > 1 - 1e-8:
                repeated.append((int(S[i]), int(S[j])))
    rep_drop = sorted({b for _, b in repeated})
    S_core = np.array([c for c in S if c not in rep_drop])

    # PCA basis of S, 2-component GMM
    _, _, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    W = Zs @ Vt[:r].T
    gm = GaussianMixture(2, covariance_type="full", n_init=gmm_n_init, random_state=random_state,
                         reg_covar=1e-6, max_iter=1000).fit(W)
    post = gm.predict_proba(W)[:, 1]

    XS = X[:, S_core]
    n1, n0 = post.sum(), (1 - post).sum()
    m1, C1 = _weighted_moments(XS, post)
    m0, C0 = _weighted_moments(XS, 1 - post)
    v1, v0 = np.diag(C1), np.diag(C0)
    z = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    d_std = (m1 - m0) / np.sqrt(0.5 * (v0 + v1))
    Cw = 0.5 * (C0 + C1)
    loc = {c: k for k, c in enumerate(S_core)}
    zero_all = [c for c in S_core if abs(z[loc[c]]) < z_thr]
    nonzero_all = [c for c in S_core if abs(z[loc[c]]) >= z_thr]

    # basis search: all r-subsets of S_core (unconstrained; the hypercube zero-difference columns are
    # only forced into the basis when the full search is too large)
    if math.comb(len(S_core), r) <= max_combos:
        pool, fixed = list(S_core), []
    else:
        fixed = zero_all[:r]
        pool = [c for c in S_core if c not in fixed]
    k = r - len(fixed)
    n_comb = math.comb(len(pool), k) if 0 <= k <= len(pool) else 0
    dsd = np.sqrt(np.diag(Cw))
    Rw = Cw / np.outer(dsd, dsd)
    fixed_idx = [loc[c] for c in fixed]
    rows = []
    if 0 < n_comb <= max_combos:
        combos = list(itertools.combinations(pool, k))
        for start in range(0, len(combos), 20000):
            chunk = combos[start:start + 20000]
            idx = np.array([fixed_idx + [loc[c] for c in cb] for cb in chunk])
            mats = Rw[idx[:, :, None], idx[:, None, :]]
            sign, ld = np.linalg.slogdet(mats)
            ld = np.where(sign > 0, ld, -np.inf)
            rows.extend(zip(chunk, ld))
        rows.sort(key=lambda t: -t[1])
        best = list(fixed) + list(rows[0][0])
        margin = float(rows[0][1] - rows[1][1]) if len(rows) > 1 else float("inf")
    elif n_comb == 0 or k == 0:
        best, margin = list(fixed), float("nan")
    else:  # greedy pivoted selection of the most independent columns
        best, margin = list(fixed), float("nan")
        while len(best) < r:
            cand = [c for c in pool if c not in best]
            scores = []
            for c in cand:
                ii = [loc[b] for b in best + [c]]
                scores.append(np.linalg.slogdet(Rw[np.ix_(ii, ii)])[1])
            best.append(cand[int(np.argmax(scores))])
    informative = sorted(int(c) for c in best)
    redundant = sorted(int(c) for c in S_core if c not in informative)
    # hypercube test inside the identified basis: informative columns split into exact zeros and actives
    zero = [c for c in informative if abs(z[loc[c]]) < z_thr]
    nonzero = [c for c in S_core if c not in zero]
    hypercube_consistent = 0 < len(zero) < r and all(abs(z[loc[c]]) >= z_thr for c in informative if c not in zero)

    # exact dependencies and scale estimates
    I_idx = [loc[c] for c in informative]
    R_idx = [loc[c] for c in redundant]
    Xc = XS - XS.mean(0)
    beta, *_ = np.linalg.lstsq(Xc[:, I_idx], Xc[:, R_idx], rcond=None)
    resid = Xc[:, R_idx] - Xc[:, I_idx] @ beta
    dep_r2 = 1 - resid.var(0) / Xc[:, R_idx].var(0)
    const = XS[:, R_idx].mean(0) - XS[:, I_idx].mean(0) @ beta
    q = r / 3.0
    a_I = np.sqrt(np.diag(Cw)[I_idx] / q)
    Bs = beta * a_I[:, None]
    a_R = np.sqrt((Bs ** 2).sum(0) / q)
    B = Bs / a_R[None, :]
    delta_x = (m1 - m0)
    delta_u = delta_x[I_idx] / a_I
    active = [c for c in informative if c not in zero]
    act_idx = [informative.index(c) for c in active]
    cs_hat = float(np.mean(np.abs(delta_u[act_idx])) / 2) if act_idx else float("nan")
    Su0 = C0[np.ix_(I_idx, I_idx)] / np.outer(a_I, a_I)
    Su1 = C1[np.ix_(I_idx, I_idx)] / np.outer(a_I, a_I)
    offd = np.r_[Su0[np.triu_indices(r, 1)], Su1[np.triu_indices(r, 1)]]

    table = [{"rank": i + 1, "added_columns": " ".join(str(c) for c in cb), "logdet_within_corr": float(ld)}
             for i, (cb, ld) in enumerate(rows[:top_k_table])]
    return {
        "n_rows": n, "structured": [int(c) for c in S], "noise": [int(c) for c in N], "rank": r,
        "eigenvalues_S": ev, "repeated_pairs": repeated, "maxcorr": maxcorr, "r2_on_others": r2,
        "S_core": [int(c) for c in S_core], "z_mean_diff": dict(zip([int(c) for c in S_core], z)),
        "d_std_mean_diff": dict(zip([int(c) for c in S_core], d_std)),
        "zero_diff": [int(c) for c in zero], "nonzero_diff": [int(c) for c in nonzero],
        "zero_diff_any_S": [int(c) for c in zero_all], "nonzero_diff_any_S": [int(c) for c in nonzero_all],
        "z_zero_max": float(max([abs(z[loc[c]]) for c in zero], default=float("nan"))),
        "z_active_min": float(min([abs(z[loc[c]]) for c in active], default=float("nan"))),
        "z_redundant_min": float(min([abs(z[loc[c]]) for c in redundant], default=float("nan"))),
        "hypercube_consistent": bool(hypercube_consistent), "n_basis_candidates": int(n_comb),
        "basis_margin_logdet": margin, "basis_table": table,
        "informative": informative, "redundant": redundant, "active": [int(c) for c in active],
        "hamming": len(active), "class_sep_hat": cs_hat, "delta_u": delta_u,
        "beta": beta, "const": const, "dep_r2": dep_r2, "dep_max_abs_resid": np.abs(resid).max(0),
        "a_informative": a_I, "a_redundant": a_R, "B_unscaled": B,
        "u_cov_offdiag_sd": float(offd.std()), "u_cov_diag_ratio": np.diag(Su1) / np.diag(Su0),
        "gmm_weights": gm.weights_, "gmm_bic": float(gm.bic(W)), "posterior": post,
        "noise_mean": mu[N], "noise_std": sd[N],
    }


# ============================================================================ variant (a): parametric
@dataclass
class ParametricStructure:
    cols_I: np.ndarray
    cols_R: np.ndarray
    cols_N: np.ndarray
    mu_I: np.ndarray
    sd_I: np.ndarray
    M_R: np.ndarray           # latent (standardised informative coords) -> redundant columns
    mu_R: np.ndarray
    mu_N: np.ndarray
    sd_N: np.ndarray
    means: np.ndarray         # (2, r) latent class means, row 1 = positive class
    covs: np.ndarray          # (2, r, r)
    weights: np.ndarray       # fitted component weights (class 0, class 1)
    n_features: int = N_FEATURES
    info: dict = field(default_factory=dict)


def fit_parametric(X_all, y_train=None, n_train: int = N_TRAIN, struct: dict | None = None,
                   n_init: int = 5, random_state: int = 0) -> ParametricStructure:
    """Fit the parametric replica to the real 20,000-row matrix (train rows first).
    Labels (optional) only decide which mixture component is called class 1."""
    X = np.asarray(X_all, float)
    st = struct or identify_structure(X)
    I, R, N = np.array(st["informative"]), np.array(st["redundant"]), np.array(st["noise"])
    mu_I, sd_I = X[:, I].mean(0), X[:, I].std(0)
    L = (X[:, I] - mu_I) / sd_I
    gm = GaussianMixture(2, covariance_type="full", n_init=n_init, random_state=random_state,
                         reg_covar=1e-6, max_iter=1000, tol=1e-6).fit(L)
    order = np.array([0, 1])
    if y_train is not None:
        Rt = gm.predict_proba(L[:n_train])
        y = np.asarray(y_train).astype(int)
        if Rt[y == 1, 1].mean() < Rt[y == 1, 0].mean():
            order = np.array([1, 0])
    mu_R = X[:, R].mean(0)
    M_R, *_ = np.linalg.lstsq(L, X[:, R] - mu_R, rcond=None)
    return ParametricStructure(I, R, N, mu_I, sd_I, M_R, mu_R, X[:, N].mean(0), X[:, N].std(0),
                               gm.means_[order], gm.covariances_[order], gm.weights_[order],
                               X.shape[1], {"gmm_bic": float(gm.bic(L)), "structure": {
                                   k: st[k] for k in ("informative", "redundant", "rank", "class_sep_hat", "hamming")}})


def _mvn_logpdf(Xc, cov):
    """log N(x; 0, cov) for rows of the centred matrix Xc."""
    Lc = np.linalg.cholesky(cov)
    sol = np.linalg.solve(Lc, Xc.T)
    return -0.5 * (sol ** 2).sum(0) - np.log(np.diag(Lc)).sum() - 0.5 * Xc.shape[1] * np.log(2 * np.pi)


@dataclass
class Replica:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    cluster_train: np.ndarray
    cluster_test: np.ndarray
    bayes_train: np.ndarray
    bayes_test: np.ndarray
    public_idx: np.ndarray
    roles: np.ndarray
    params: dict

    @property
    def X_all(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_test])

    @property
    def private_mask(self) -> np.ndarray:
        m = np.ones(len(self.y_test), bool)
        m[self.public_idx] = False
        return m

    def frames(self):
        import pandas as pd
        cols = [str(i) for i in range(self.X_train.shape[1])]
        tr = pd.DataFrame(self.X_train, columns=cols)
        tr.insert(0, "target", self.y_train)
        tr.insert(0, "id", np.arange(len(tr)))
        te = pd.DataFrame(self.X_test, columns=cols)
        te.insert(0, "id", np.arange(len(tr), len(tr) + len(te)))
        return tr, te


def _finish_replica(X, y, cluster, logodds, roles, params, rng, n_train, n_public, decimals,
                    rowperm=None) -> Replica:
    if rowperm is None:
        rowperm = rng.permutation(len(y))
    X, y, cluster, logodds = X[rowperm], y[rowperm], cluster[rowperm], logodds[rowperm]
    if decimals is not None:
        X = np.round(X, decimals)
    n_test = len(y) - n_train
    pub = np.sort(rng.choice(n_test, size=n_public, replace=False))
    return Replica(X[:n_train], y[:n_train], X[n_train:], y[n_train:], cluster[:n_train], cluster[n_train:],
                   logodds[:n_train], logodds[n_train:], pub, np.asarray(roles), params)


def sample_parametric(ps: ParametricStructure, seed: int, flip_y: float = 0.0, balanced: bool = True,
                      n_total: int = N_TOTAL, n_train: int = N_TRAIN, n_public: int = N_PUBLIC,
                      decimals: int | None = DECIMALS) -> Replica:
    """Variant (a): parametric bootstrap of the real data (fixed fitted parameters, new rows)."""
    rng = np.random.default_rng([20261022, int(seed)])
    if balanced:
        n1 = n_total // 2
    else:
        n1 = int(rng.binomial(n_total, ps.weights[1]))
    n0 = n_total - n1
    cluster = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    r = ps.means.shape[1]
    L = np.empty((n_total, r))
    for k, (a, b) in enumerate([(0, n0), (n0, n_total)]):
        Lc = np.linalg.cholesky(ps.covs[k])
        L[a:b] = ps.means[k] + rng.standard_normal((b - a, r)) @ Lc.T
    X = np.empty((n_total, ps.n_features))
    X[:, ps.cols_I] = ps.mu_I + ps.sd_I * L
    X[:, ps.cols_R] = ps.mu_R + L @ ps.M_R
    X[:, ps.cols_N] = ps.mu_N + ps.sd_N * rng.standard_normal((n_total, len(ps.cols_N)))
    y = cluster.copy()
    flip = rng.random(n_total) < flip_y
    y[flip] = rng.integers(0, 2, flip.sum())
    logodds = (_mvn_logpdf(L - ps.means[1], ps.covs[1]) - _mvn_logpdf(L - ps.means[0], ps.covs[0])
               + np.log(n1 / n0))
    roles = np.array(["noise"] * ps.n_features, dtype=object)
    roles[ps.cols_I] = "informative"
    roles[ps.cols_R] = "redundant"
    params = {"variant": "parametric", "seed": int(seed), "flip_y": flip_y, "n_flipped_true": int((y != cluster).sum()),
              "balanced": balanced}
    return _finish_replica(X, y, cluster, logodds, roles, params, rng, n_train, n_public, decimals)


# ============================================================================ variant (b): make_classification
class _RecordingRS(np.random.RandomState):
    """RandomState that records the arrays returned by uniform / standard_normal, so the internal
    parameters of sklearn's make_classification (A_k, B, informative z) can be read back."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.log = []

    def uniform(self, low=0.0, high=1.0, size=None):
        out = super().uniform(low, high, size)
        self.log.append(("uniform", out))
        return out

    def standard_normal(self, size=None):
        out = super().standard_normal(size)
        self.log.append(("standard_normal", out))
        return out


def make_classification_with_truth(n_samples=N_TOTAL, n_features=N_FEATURES, n_informative=12, n_redundant=8,
                                   n_repeated=0, n_clusters_per_class=1, weights=None, flip_y=0.0,
                                   class_sep=1.0, hypercube=True, random_state=0):
    """sklearn.datasets.make_classification(..., shift=0, scale=1, shuffle=False) plus its internal
    parameters: centroids (n_clusters, r), A (n_clusters, r, r), B (r, n_red), cluster id per row."""
    from sklearn.datasets import make_classification
    rs = _RecordingRS(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated, n_classes=2,
                               n_clusters_per_class=n_clusters_per_class, weights=weights, flip_y=flip_y,
                               class_sep=class_sep, hypercube=hypercube, shift=0.0, scale=1.0,
                               shuffle=False, random_state=rs)
    r = n_informative
    n_clusters = 2 * n_clusters_per_class
    k0 = next(i for i, (nm, a) in enumerate(rs.log) if nm == "standard_normal" and a.shape == (n_samples, r))
    Z = rs.log[k0][1]
    unis = [a for nm, a in rs.log[k0 + 1:] if nm == "uniform"]
    A = np.array([2 * u - 1 for u in unis[:n_clusters]])
    B = 2 * unis[n_clusters] - 1 if n_redundant > 0 else np.zeros((r, 0))
    w = [1.0 / 2] * 2 if weights is None else list(weights) + ([1.0 - sum(weights)] if len(weights) == 1 else [])
    sizes = [int(n_samples * w[k % 2] / n_clusters_per_class) for k in range(n_clusters)]
    for i in range(n_samples - sum(sizes)):
        sizes[i % n_clusters] += 1
    starts = np.r_[0, np.cumsum(sizes)]
    cluster_id = np.repeat(np.arange(n_clusters), sizes)
    centroids = np.array([X[starts[k], :r] - Z[starts[k]] @ A[k] for k in range(n_clusters)])
    return X, y, {"centroids": centroids, "A": A, "B": B, "cluster_id": cluster_id, "sizes": np.array(sizes),
                  "Z": Z}


def sample_make_classification(seed: int, class_sep: float = 0.7, flip_y: float = 0.0, n_informative: int = 12,
                               n_redundant: int = 8, n_repeated: int = 0, n_clusters_per_class: int = 1,
                               weights=None, hypercube: bool = True, scale_range=(0.5, 3.0),
                               shift_range=(-5.0, 5.0), hamming: int | None = None, n_total: int = N_TOTAL,
                               n_features: int = N_FEATURES, n_train: int = N_TRAIN, n_public: int = N_PUBLIC,
                               decimals: int | None = DECIMALS, max_tries: int = 2000) -> Replica:
    """Variant (b): sklearn make_classification + random row/column permutation + affine map
    x_j <- a_j x_j + b_j (a ~ U(scale_range), b ~ U(shift_range)). hamming=h re-draws the generator
    seed until the two class centroids differ in exactly h coordinates (n_clusters_per_class=1)."""
    r = n_informative
    for t in range(max_tries):
        gseed = int(seed) if hamming is None else int((int(seed) * 7919 + t) % (2 ** 31 - 1))
        X, y, tr = make_classification_with_truth(n_total, n_features, r, n_redundant, n_repeated,
                                                  n_clusters_per_class, weights, flip_y, class_sep, hypercube,
                                                  random_state=gseed)
        h = int((np.abs(tr["centroids"][0] - tr["centroids"][1]) > 1e-9).sum())
        if hamming is None or n_clusters_per_class != 1 or h == hamming:
            break
    else:
        raise RuntimeError(f"no generator seed with hamming={hamming} in {max_tries} tries")
    u = X[:, :r].copy()
    cid = tr["cluster_id"]
    cls_of = np.arange(2 * n_clusters_per_class) % 2
    cluster = cls_of[cid]
    logp = np.stack([_mvn_logpdf(u - tr["centroids"][k], tr["A"][k].T @ tr["A"][k]) + np.log(tr["sizes"][k])
                     for k in range(len(tr["sizes"]))], 1)
    lse = lambda M: M.max(1) + np.log(np.exp(M - M.max(1, keepdims=True)).sum(1))  # noqa: E731
    logodds = lse(logp[:, cls_of == 1]) - lse(logp[:, cls_of == 0])
    rng = np.random.default_rng([20261023, int(seed)])
    colperm = rng.permutation(n_features)
    a = rng.uniform(*scale_range, n_features)
    b = rng.uniform(*shift_range, n_features)
    X = X[:, colperm] * a + b
    nr = r + n_redundant
    roles = np.where(colperm < r, "informative",
                     np.where(colperm < nr, "redundant", np.where(colperm < nr + n_repeated, "repeated", "noise")))
    params = {"variant": "make_classification", "seed": int(seed), "generator_seed": gseed, "class_sep": class_sep,
              "flip_y": flip_y, "hamming": h, "n_informative": r, "n_redundant": n_redundant,
              "n_repeated": n_repeated, "n_clusters_per_class": n_clusters_per_class, "hypercube": hypercube,
              "scale_range": list(scale_range), "shift_range": list(shift_range), "a": a, "b": b,
              "colperm": colperm, "n_flipped_true": int((y != cluster).sum()),
              "centroids": tr["centroids"], "A": tr["A"], "B": tr["B"], "cluster_sizes": tr["sizes"]}
    return _finish_replica(X, y.astype(int), cluster.astype(int), logodds, roles, params, rng, n_train, n_public,
                           decimals)


# ============================================================================ procedures
def highvar_columns(X_all, ratio: float = 1.5) -> np.ndarray:
    std = np.asarray(X_all).std(0)
    return np.where(std > ratio * np.median(std))[0]


def resolve_columns(X_all, columns) -> np.ndarray:
    if columns is None or (isinstance(columns, str) and columns == "all"):
        return np.arange(np.asarray(X_all).shape[1])
    if isinstance(columns, str) and columns.startswith("highvar"):
        return highvar_columns(X_all, float(columns.split(":", 1)[1]) if ":" in columns else 1.5)
    if isinstance(columns, str) and columns == "structured":
        return structured_columns(X_all)[0]
    return np.asarray(columns, int)


class UnsupervisedGMM:
    """Label-free part of ssc.transductive.TransductiveGMM on an in-memory matrix (train rows first)."""

    def __init__(self, X_all, columns="highvar", K: int = 2, pca_dims=None, n_init: int = 5,
                 reg_covar: float = 1e-6, random_state: int = 0, covariance_type: str = "full",
                 tol: float = 1e-3, max_iter: int = 1000):
        self.X_all = np.asarray(X_all, float)
        self.columns, self.K, self.pca_dims = columns, K, pca_dims
        self.n_init, self.reg_covar, self.random_state, self.covariance_type = n_init, reg_covar, random_state, covariance_type
        self.tol, self.max_iter = tol, max_iter   # sklearn / ssc default tol 1e-3 (loose); 1e-8 = converged MLE

    def fit(self):
        if isinstance(self.columns, str) and self.columns == "informative":
            cols = np.asarray(identify_structure(self.X_all)["informative"])
        else:
            cols = resolve_columns(self.X_all, self.columns)
        B = self.X_all[:, cols]
        self.mu_, self.sd_ = B.mean(0), B.std(0)
        Z = (B - self.mu_) / self.sd_
        d = self.pca_dims
        self.Vt_ = None
        if d:
            _, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            if d == "rank":
                d = int((s ** 2 > 1e-9 * s[0] ** 2).sum())
            self.Vt_ = Vt[:d]
            Z = Z @ self.Vt_.T
        self.cols_, self.dims_ = cols, Z.shape[1]
        self.gmm_ = GaussianMixture(self.K, covariance_type=self.covariance_type, n_init=self.n_init,
                                    random_state=self.random_state, reg_covar=self.reg_covar, tol=self.tol,
                                    max_iter=self.max_iter).fit(Z)
        self.converged_, self.n_iter_ = bool(self.gmm_.converged_), int(self.gmm_.n_iter_)
        self.R = self.gmm_.predict_proba(Z)
        self.bic = float(self.gmm_.bic(Z))
        self.loglik = float(self.gmm_.score(Z) * len(Z))
        return self

    def rates(self, train_idx, y_train, alpha: float = 1.0) -> np.ndarray:
        y = np.asarray(y_train).astype(int)
        Rt = self.R[train_idx]
        p0 = y.mean()
        return (Rt.T @ y + alpha * p0) / (Rt.sum(0) + alpha)

    def scores(self, train_idx, y_train, alpha: float = 1.0) -> np.ndarray:
        """Score of every row = sum_k P(k|x) * smoothed class-1 rate of component k (as in ssc)."""
        return np.clip(self.R @ self.rates(train_idx, y_train, alpha), 0, 1)

    def responsibilities(self, X) -> np.ndarray:
        """P(k | x) for new rows given as raw 150-column feature rows."""
        Z = (np.asarray(X, float)[:, self.cols_] - self.mu_) / self.sd_
        if self.Vt_ is not None:
            Z = Z @ self.Vt_.T
        return self.gmm_.predict_proba(Z)


_UNSUP_CACHE: dict = {}


class TransductiveGMMTight(ClassifierMixin, BaseEstimator):
    """sklearn estimator like ssc.transductive.TransductiveGMM (mixture on train+test FEATURES of the competition
    data, labels only map components to classes, Beta-smoothed component rates), with two extra options:
    columns="informative" (the 12 label-free identified informative columns, R150) and an EM tolerance
    (tol=1e-8 runs EM to convergence; ssc uses sklearn's default 1e-3). Inputs X = raw 150 columns."""

    def __init__(self, columns="informative", n_components: int = 2, pca_dims=None, n_init: int = 5,
                 alpha: float = 1.0, reg_covar: float = 1e-6, tol: float = 1e-8, max_iter: int = 5000,
                 source: str = "raw", random_state: int = 0):
        self.columns, self.n_components, self.pca_dims, self.n_init = columns, n_components, pca_dims, n_init
        self.alpha, self.reg_covar, self.tol, self.max_iter = alpha, reg_covar, tol, max_iter
        self.source, self.random_state = source, random_state

    def _unsup(self):
        key = (self.source, str(self.columns), self.n_components, str(self.pca_dims), self.n_init, self.reg_covar,
               self.tol, self.max_iter, self.random_state)
        if key not in _UNSUP_CACHE:
            from ssc.transductive import unlabeled_features
            _UNSUP_CACHE[key] = UnsupervisedGMM(unlabeled_features(self.source), self.columns, self.n_components,
                                                self.pca_dims, self.n_init, self.reg_covar, self.random_state,
                                                tol=self.tol, max_iter=self.max_iter).fit()
        return _UNSUP_CACHE[key]

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        g = self._unsup()
        R = g.responsibilities(X)
        p0 = y.mean()
        self.component_rate_ = (R.T @ y + self.alpha * p0) / (R.sum(0) + self.alpha)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        p1 = np.clip(self._unsup().responsibilities(X) @ self.component_rate_, 0.0, 1.0)
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StdSubset(TransformerMixin, BaseEstimator):
    """Select columns and standardise them with fixed (train+test) statistics, like ssc SubspaceModel."""

    def __init__(self, cols=(), mu=(), sd=()):
        self.cols, self.mu, self.sd = cols, mu, sd

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (np.asarray(X, float)[:, list(self.cols)] - np.asarray(self.mu)) / np.asarray(self.sd)


def make_supervised(name: str, X_all):
    """R003 supervised candidates on an in-memory matrix."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    if name == "l1_all":
        from ssc.models import build_pipeline
        return build_pipeline({"model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 1.0}}})
    X_all = np.asarray(X_all, float)
    hv = highvar_columns(X_all)
    sub = StdSubset(tuple(int(c) for c in hv), tuple(X_all[:, hv].mean(0)), tuple(X_all[:, hv].std(0)))
    est = {"lda_hv": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
           "qda_hv": QuadraticDiscriminantAnalysis(reg_param=0.1),
           "logreg_hv": LogisticRegression(C=1.0, max_iter=5000)}[name]
    return Pipeline([("sub", sub), ("model", est)])


GMM_SPECS = {   # name -> UnsupervisedGMM kwargs (R003 + variants)
    "gmm2_hv": dict(columns="highvar", K=2),
    "gmm2_hv_tight": dict(columns="highvar", K=2, tol=1e-8, max_iter=5000),
    "gmm2_hv_pca": dict(columns="highvar", K=2, pca_dims="rank"),
    "gmm4_hv": dict(columns="highvar", K=4),
    "gmm2_all": dict(columns="all", K=2, n_init=2),
    "gmm2_struct_pca": dict(columns="structured", K=2, pca_dims="rank"),
    "gmm2_struct": dict(columns="structured", K=2),
}


def cv_ranking(rep: Replica, models=("l1_all", "lda_hv", "qda_hv", "logreg_hv", "gmm2_hv"), n_repeats: int = 4,
               seed: int = 2026, gmm_cache: dict | None = None) -> list[dict]:
    """R003-style evaluation on one replica: repeated stratified 5-fold CV AUC on the 250 train rows
    (same split rule as the official protocol, fewer repeats) and the fold-mean test score's AUC on the
    private / public parts of the replica's test set."""
    from ssc.validation import Scheme, cross_validate, make_splits
    scheme = Scheme(kind="kfold", n_splits=5, n_repeats=n_repeats, seed=seed)
    splits = make_splits(rep.y_train, scheme)
    X_all = rep.X_all
    n_tr = len(rep.y_train)
    priv = rep.private_mask
    out = []
    for name in models:
        if name in GMM_SPECS:
            g = (gmm_cache or {}).get(name) or UnsupervisedGMM(X_all, **GMM_SPECS[name]).fit()
            if gmm_cache is not None:
                gmm_cache[name] = g
            aucs, test = [], np.zeros(len(rep.y_test))
            for s in splits:
                sc = g.scores(s.train_idx, rep.y_train[s.train_idx])
                aucs.append(auc_fast(rep.y_train[s.val_idx], sc[s.val_idx]))
                test += sc[n_tr:]
            test /= len(splits)
        else:
            cv = cross_validate(make_supervised(name, X_all), rep.X_train, rep.y_train, splits, scheme,
                                X_test=rep.X_test, n_jobs=1)
            aucs, test = cv.fold_scores["auc"].to_numpy(), cv.test_pred
        a = np.asarray(aucs).reshape(n_repeats, -1).mean(1)
        out.append({"model": name, "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std_repeat": float(a.std(ddof=1)),
                    "test_auc_private": auc_fast(rep.y_test[priv], test[priv]),
                    "test_auc_public": auc_fast(rep.y_test[~priv], test[~priv]), "_test": test})
    return out


# ============================================================================ evaluation helpers
def bayes_auc(rep: Replica, subset: str = "private", labels: str = "observed") -> float:
    y = rep.y_test if labels == "observed" else rep.cluster_test
    m = {"private": rep.private_mask, "public": ~rep.private_mask, "all": np.ones(len(y), bool)}[subset]
    return auc_fast(y[m], rep.bayes_test[m])


def lb_resample(y_test, scores: dict, n_public: int = N_PUBLIC, n_draws: int = 200, seed: int = 0) -> list[dict]:
    """Draw n_draws random public subsets of the test rows (Kaggle-like 2 % public LB); for every draw and
    model return the public AUC and the private AUC (the complement)."""
    rng = np.random.default_rng(seed)
    y_test = np.asarray(y_test)
    n = len(y_test)
    rows = []
    for d in range(n_draws):
        mask = np.zeros(n, bool)
        mask[rng.choice(n, n_public, replace=False)] = True
        for k, v in scores.items():
            rows.append({"draw": d, "model": k, "public": auc_fast(y_test[mask], v[mask]),
                         "private": auc_fast(y_test[~mask], v[~mask])})
    return rows


def within_class_mardia(L, post, thr: float = 0.999, n_sub: int = 3000, seed: int = 0) -> dict:
    """Mardia multivariate kurtosis z and skewness chi2/df for the confident rows of each class in the
    latent coordinates L (truncation at the boundary biases both; compare with Gaussian replicas)."""
    out = {}
    rng = np.random.default_rng(seed)
    r = L.shape[1]
    res = []
    for mask in (post > thr, post < 1 - thr):
        Lc = L[mask]
        m = Lc.mean(0)
        Sinv = np.linalg.inv(np.cov(Lc, rowvar=False, bias=True))
        D = np.einsum("ij,jk,ik->i", Lc - m, Sinv, Lc - m)
        zk = ((D ** 2).mean() - r * (r + 2)) / np.sqrt(8 * r * (r + 2) / len(Lc))
        sub = Lc[rng.choice(len(Lc), min(n_sub, len(Lc)), replace=False)] - m
        b1 = ((sub @ Sinv @ sub.T) ** 3).mean()
        res.append((float(zk), float(len(sub) * b1 / 6 / (r * (r + 1) * (r + 2) / 6))))
    out["mardia_kurt_z_absmax"] = max(abs(z) for z, _ in res)
    out["mardia_skew_chi2_over_df_max"] = max(c for _, c in res)
    return out


def summary_stats(X_all, y_train, n_train: int = N_TRAIN, n_eig: int = 15, informative_cols=None) -> dict:
    """Label-free (and train-label purity) summary statistics used to compare replicas with the real data."""
    X = np.asarray(X_all, float)
    std = X.std(0)
    hv = highvar_columns(X)
    C = np.corrcoef(X, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    S, maxcorr, _ = structured_columns(X)
    Zs = (X[:, S] - X[:, S].mean(0)) / X[:, S].std(0)
    evS = np.linalg.eigvalsh(Zs.T @ Zs / len(Zs))
    rankS = int((evS > 1e-9 * evS.max()).sum())
    Zb = (X[:, hv] - X[:, hv].mean(0)) / X[:, hv].std(0)
    evb = np.linalg.eigvalsh(Zb.T @ Zb / len(Zb))
    rank_hv = int((evb > 1e-9 * evb.max()).sum())
    g = UnsupervisedGMM(X, "highvar", 2).fit()
    lab = g.R[:n_train].argmax(1)
    y = np.asarray(y_train)
    rates = sorted(float(y[lab == k].mean()) if (lab == k).any() else float("nan") for k in range(2))
    post = g.R[:, 1]
    g3 = UnsupervisedGMM(X, "highvar", 3, n_init=2).fit()
    noise = np.setdiff1d(np.arange(X.shape[1]), S)
    from ssc.features import abs_correlation
    return {
        "std_q10": float(np.quantile(std, .1)), "std_median": float(np.median(std)),
        "std_q90": float(np.quantile(std, .9)), "std_max": float(std.max()),
        "n_highvar": int(len(hv)), "rank_highvar": rank_hv, "n_structured": int(len(S)), "rank_structured": rankS,
        "n_corr_eig_lt_1e-6": int((ev < 1e-6).sum()), **{f"corr_eig_{i + 1}": float(ev[i]) for i in range(n_eig)},
        "gmm2_hv_train_rate_lo": rates[0], "gmm2_hv_train_rate_hi": rates[1],
        "gmm2_hv_weight_min": float(g.gmm_.weights_.min()),
        "gmm_bic_per_row_k3_minus_k2": (g3.bic - g.bic) / len(X),
        "frac_post_0.1_0.9": float(((post > 0.1) & (post < 0.9)).mean()),
        "mean_min_post": float(np.minimum(post, 1 - post).mean()),
        "noise_max_abs_corr_target_train": float(abs_correlation(X[:n_train][:, noise], y).max()) if len(noise) else float("nan"),
        "struct_max_abs_corr_target_train": float(abs_correlation(X[:n_train][:, S], y).max()),
        **(_mardia_for(X, informative_cols) if informative_cols is not None else {}),
    }


def _mardia_for(X, cols):
    cols = np.asarray(cols, int)
    L = (X[:, cols] - X[:, cols].mean(0)) / X[:, cols].std(0)
    g = UnsupervisedGMM(X, cols, 2, tol=1e-6).fit()
    return within_class_mardia(L, g.R[:, 1])
