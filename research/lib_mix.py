"""Shared helpers for the 'mix' research line (research/r100-r149).

Label-free tools on the 20,000 train+test feature rows (labels never enter these):
  * standardisation and a PCA basis that drops exact linear redundancies,
  * per-feature class-dependence scores given soft component labels r (marginal and
    conditional-on-a-feature-set likelihood-ratio tests, with the BIC threshold df*log n),
  * Gaussian-mixture fitting, an own EM implementation that also supports label
    constraints (semi-supervised EM) and warm starts from responsibilities,
  * sklearn-compatible transductive estimators for ssc.validation.cross_validate,
  * a parametric generator (replica datasets with known labels) for the bootstrap check.

Labels enter only inside estimator.fit(X_train_fold, y_train_fold), exactly like
ssc.transductive.TransductiveGMM (D007).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.mixture import GaussianMixture

from ssc import paths
from ssc.transductive import unlabeled_features

BLOCK = (8, 15, 26, 30, 46, 49, 64, 67, 77, 80, 98, 100, 112)
N_TRAIN = 250
S003_RUN = "20261022-050708_r003_gmm2_hv"


# ----------------------------------------------------------------------------- basics
def standardize(X):
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / sd, mu, sd


def pca_basis(Z, rel_tol: float = 1e-8):
    """Eigen-basis of the covariance of Z (columns already standardised), dropping
    directions with eigenvalue <= rel_tol * max (exact linear redundancies; the CSV is
    rounded to 6 decimals, so exact relations leave eigenvalues of ~1e-13)."""
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    keep = w > rel_tol * w[0]
    return V[:, keep], w


@lru_cache(maxsize=256)
def basis(columns: tuple, source: str = "raw", rel_tol: float = 1e-8, whiten: bool = False):
    """(cols, mu, sd, W, U_all): U_all = coordinates of all 20,000 rows in the
    non-degenerate PCA basis of the standardised columns."""
    X_all = unlabeled_features(source)
    cols = np.array(columns, dtype=int)
    Z, mu, sd = standardize(X_all[:, cols])
    W, w = pca_basis(Z, rel_tol)
    if whiten:
        W = W / np.sqrt(w[: W.shape[1]])
    U = (Z - Z.mean(0)) @ W
    return cols, mu, sd, W, U


def transform(X, columns: tuple, source: str = "raw", rel_tol: float = 1e-8, whiten: bool = False):
    cols, mu, sd, W, _ = basis(columns, source, rel_tol, whiten)
    Z = (np.asarray(X, dtype=float)[:, cols] - mu) / sd
    # Z.mean(0) over all rows is 0 by construction of the standardisation
    return Z @ W


@lru_cache(maxsize=4)
def _train_row_index(source: str = "raw"):
    X_all = unlabeled_features(source)
    return {X_all[i].tobytes(): i for i in range(N_TRAIN)}


def rows_in_all(X, source: str = "raw") -> np.ndarray:
    """Positions of the given (training) rows inside the 20,000-row matrix."""
    idx = _train_row_index(source)
    X = np.asarray(X, dtype=float)
    out = np.array([idx.get(r.tobytes(), -1) for r in X])
    if (out < 0).any():
        raise ValueError("some rows are not training rows of the unlabeled matrix")
    return out


# ----------------------------------------------------------------------------- feature scores
def _wls_resvar(D, X, w):
    """Weighted least squares of every column of X on design D; returns the weighted
    residual variance per column."""
    Dw = D * w[:, None]
    G = D.T @ Dw
    R = Dw.T @ X
    B = np.linalg.solve(G + 1e-10 * np.eye(len(G)) * np.trace(G) / len(G), R)
    ss = (w[:, None] * X**2).sum(0) - (R * B).sum(0)
    return np.maximum(ss / w.sum(), 1e-300)


def class_dependence(X_cand, r, U=None):
    """Label-free class-dependence of each candidate column given soft labels r (P(class 1)).

    For each column x: two-component expected complete-data likelihood-ratio test of
    x | U, c ~ N(a_c + b_c'U, s_c^2) (class-specific) against x | U ~ N(a + b'U, s^2)
    (common), LRT = n log s^2 - n1 log s1^2 - n0 log s0^2, df = dim(U) + 2.
    With U=None the test is marginal (df = 2). Also returns the pooled R^2 of x on U.
    dBIC = df * log(n) - LRT (< 0: the class-specific conditional model is preferred)."""
    X_cand = np.asarray(X_cand, dtype=float)
    n = len(X_cand)
    D = np.ones((n, 1)) if U is None else np.c_[np.ones(n), U]
    one = np.ones(n)
    s_all = _wls_resvar(D, X_cand, one)
    s1 = _wls_resvar(D, X_cand, r)
    s0 = _wls_resvar(D, X_cand, 1 - r)
    n1 = r.sum()
    n0 = n - n1
    lrt = n * np.log(s_all) - n1 * np.log(s1) - n0 * np.log(s0)
    df = D.shape[1] + 1
    r2 = 1 - s_all / X_cand.var(0)
    # marginal effect sizes (for the table)
    m1 = (r[:, None] * X_cand).sum(0) / n1
    m0 = ((1 - r)[:, None] * X_cand).sum(0) / n0
    v1 = (r[:, None] * (X_cand - m1) ** 2).sum(0) / n1
    v0 = ((1 - r)[:, None] * (X_cand - m0) ** 2).sum(0) / n0
    return {"lrt": lrt, "df": df, "dbic": df * np.log(n) - lrt, "r2": r2,
            "std_mean_diff": (m1 - m0) / np.sqrt((v1 + v0) / 2), "log_var_ratio": np.log(v1 / v0),
            "resvar_ratio": s1 / s0}


# ----------------------------------------------------------------------------- EM
def _chol_logpdf(U, mean, cov):
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (U - mean).T)
    return -0.5 * (sol**2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * U.shape[1] * np.log(2 * np.pi)


def component_logpdf(U, means, covs):
    return np.column_stack([_chol_logpdf(U, means[k], covs[k]) for k in range(len(means))])


def m_step(U, resp, cov_type="full", reg=1e-6):
    nk = resp.sum(0) + 1e-12
    weights = nk / nk.sum()
    means = (resp.T @ U) / nk[:, None]
    d = U.shape[1]
    covs = []
    for k in range(resp.shape[1]):
        Xc = U - means[k]
        C = (resp[:, k, None] * Xc).T @ Xc / nk[k]
        covs.append(C)
    covs = np.array(covs)
    if cov_type == "tied":
        covs = np.repeat((covs * weights[:, None, None]).sum(0)[None], len(covs), 0)
    elif cov_type == "diag":
        covs = np.array([np.diag(np.diag(C)) for C in covs])
    covs = covs + reg * np.eye(d)[None]
    return weights, means, covs


def em(U, resp, cov_type="full", reg=1e-6, max_iter=2000, tol=1e-9, label_idx=None, label_cls=None,
       comp_class=None, eps=0.0):
    """EM for a Gaussian mixture started from responsibilities `resp` (n x K).

    Semi-supervised option: rows label_idx (positions in U) carry class labels label_cls;
    component k belongs to class comp_class[k]. Their E-step uses P(y | k) = 1 - eps if
    comp_class[k] == y else eps (eps = 0: hard constraint). Returns dict with parameters,
    the unconstrained posterior of every row (labels NOT applied: used for scoring), the
    observed-data log-likelihood (with label terms) and the iteration count."""
    n, K = resp.shape
    lab_term = None
    if label_idx is not None and len(label_idx):
        comp_class = np.asarray(comp_class)
        match = comp_class[None, :] == np.asarray(label_cls)[:, None]
        with np.errstate(divide="ignore"):
            lab_term = np.where(match, np.log1p(-eps) if eps > 0 else 0.0, np.log(eps) if eps > 0 else -np.inf)
    prev = -np.inf
    for it in range(max_iter):
        weights, means, covs = m_step(U, resp, cov_type, reg)
        lp = component_logpdf(U, means, covs) + np.log(weights)[None]
        lp_c = lp.copy()
        if lab_term is not None:
            lp_c[label_idx] = lp_c[label_idx] + lab_term
        norm = logsumexp(lp_c, axis=1)
        ll = norm.mean()
        resp = np.exp(lp_c - norm[:, None])
        if abs(ll - prev) < tol:
            break
        prev = ll
    post = np.exp(lp - logsumexp(lp, axis=1)[:, None])
    return {"weights": weights, "means": means, "covs": covs, "post": post, "resp": resp,
            "loglik": float(ll), "n_iter": it + 1}


def n_params(K, d, cov_type):
    cov = {"full": K * d * (d + 1) / 2, "tied": d * (d + 1) / 2, "diag": K * d, "spherical": K}[cov_type]
    return int(K - 1 + K * d + cov)


# ----------------------------------------------------------------------------- shared label-free fits
@lru_cache(maxsize=128)
def sk_gmm(columns: tuple, K: int = 2, cov_type: str = "full", n_init: int = 5, seed: int = 0,
           reg: float = 1e-6, tol: float = 1e-3, source: str = "raw", whiten: bool = False):
    *_, U = basis(columns, source, whiten=whiten)
    return GaussianMixture(K, covariance_type=cov_type, n_init=n_init, random_state=seed, reg_covar=reg,
                           tol=tol, max_iter=5000).fit(U)


@lru_cache(maxsize=128)
def polished_posterior(columns: tuple, K: int = 2, cov_type: str = "full", n_init: int = 5, seed: int = 0,
                       reg: float = 1e-6, source: str = "raw"):
    """sklearn GMM (n_init restarts, default tol) polished by own EM to tol 1e-10 per row.
    Returns (posterior n x K, mean log-likelihood, params dict)."""
    *_, U = basis(columns, source)
    g = sk_gmm(columns, K, cov_type, n_init, seed, reg, 1e-3, source)
    fit = em(U, g.predict_proba(U), cov_type, reg, tol=1e-10)
    return fit["post"], fit["loglik"], fit


def component_rates(R_train, y, alpha=1.0):
    p0 = y.mean()
    return (R_train.T @ y + alpha * p0) / (R_train.sum(0) + alpha)


# ----------------------------------------------------------------------------- estimators
class MixGMM(ClassifierMixin, BaseEstimator):
    """Transductive GMM on an arbitrary label-free feature set (PCA basis without exact
    redundancies), optional EM polishing to full convergence and an ensemble of restarts
    (seeds). Score = sum_k P(k|x) * rate_k (rates from the fold's training labels)."""

    def __init__(self, columns=BLOCK, K=2, cov_type="full", reg=1e-6, n_init=5, seeds=(0,), polish=True,
                 alpha=1.0, source="raw"):
        self.columns = columns
        self.K = K
        self.cov_type = cov_type
        self.reg = reg
        self.n_init = n_init
        self.seeds = seeds
        self.polish = polish
        self.alpha = alpha
        self.source = source

    def _post_all(self, seed):
        cols = tuple(int(c) for c in self.columns)
        if self.polish:
            return polished_posterior(cols, self.K, self.cov_type, self.n_init, seed, self.reg, self.source)[0]
        *_, U = basis(cols, self.source)
        return sk_gmm(cols, self.K, self.cov_type, self.n_init, seed, self.reg, 1e-3, self.source).predict_proba(U)

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        idx = rows_in_all(X, self.source)
        self.rates_ = [component_rates(self._post_all(s)[idx], y, self.alpha) for s in self.seeds]
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def score_all(self):
        return np.mean([self._post_all(s) @ r for s, r in zip(self.seeds, self.rates_)], axis=0)

    def predict_proba(self, X):
        # rows are matched to the 20,000-row posterior through their coordinates
        cols = tuple(int(c) for c in self.columns)
        out = np.zeros(len(X))
        for s, rate in zip(self.seeds, self.rates_):
            if self.polish:
                fit = polished_posterior(cols, self.K, self.cov_type, self.n_init, s, self.reg, self.source)[2]
                U = transform(X, cols, self.source)
                lp = component_logpdf(U, fit["means"], fit["covs"]) + np.log(fit["weights"])[None]
                R = np.exp(lp - logsumexp(lp, axis=1)[:, None])
            else:
                g = sk_gmm(cols, self.K, self.cov_type, self.n_init, s, self.reg, 1e-3, self.source)
                R = g.predict_proba(transform(X, cols, self.source))
            out += R @ rate
        p1 = np.clip(out / len(self.seeds), 0, 1)
        return np.c_[1 - p1, p1]


class SemiSupEM(ClassifierMixin, BaseEstimator):
    """Semi-supervised EM: mixture on all 20,000 rows; the fold's training rows carry
    their labels as constraints (components grouped by class: k_per_class components per
    class, eps = assumed label-noise probability). Started from the unsupervised fit
    (components mapped to classes by the fold labels). Score = P(class 1 | x) without the
    label term (so validation/test rows are treated exactly alike)."""

    def __init__(self, columns=BLOCK, k_per_class=1, cov_type="full", reg=1e-6, eps=0.01, n_init=5,
                 source="raw"):
        self.columns = columns
        self.k_per_class = k_per_class
        self.cov_type = cov_type
        self.reg = reg
        self.eps = eps
        self.n_init = n_init
        self.source = source

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        cols = tuple(int(c) for c in self.columns)
        *_, U = basis(cols, self.source)
        K = 2 * self.k_per_class
        post0, _, _ = polished_posterior(cols, K, self.cov_type, self.n_init, 0, self.reg, self.source)
        idx = rows_in_all(X, self.source)
        rate = component_rates(post0[idx], y)
        # map: the k_per_class components with the highest class-1 rate -> class 1
        comp_class = np.zeros(K, int)
        comp_class[np.argsort(rate)[::-1][: self.k_per_class]] = 1
        fit = em(U, post0, self.cov_type, self.reg, tol=1e-10, label_idx=idx, label_cls=y,
                 comp_class=comp_class, eps=self.eps)
        self.fit_ = fit
        self.comp_class_ = comp_class
        self.cols_ = cols
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        U = transform(X, self.cols_, self.source)
        f = self.fit_
        lp = component_logpdf(U, f["means"], f["covs"]) + np.log(f["weights"])[None]
        R = np.exp(lp - logsumexp(lp, axis=1)[:, None])
        p1 = np.clip(R[:, self.comp_class_ == 1].sum(1), 0, 1)
        return np.c_[1 - p1, p1]


# ----------------------------------------------------------------------------- metrics
def cv_extra_metrics(cv, y, clip=1e-6):
    """Per-repeat log-loss / Brier of the out-of-fold scores (averaged over repeats) and the
    number of training rows on the wrong side of 0.5 (averaged)."""
    ll, br, wrong = [], [], []
    for row in cv.oof:
        p = np.clip(row, clip, 1 - clip)
        ll.append(log_loss(y, p, labels=[0, 1]))
        br.append(brier_score_loss(y, row))
        wrong.append(int(((row >= 0.5).astype(int) != y).sum()))
    return {"cv_logloss": float(np.mean(ll)), "cv_logloss_std_repeat": float(np.std(ll, ddof=1)),
            "cv_brier": float(np.mean(br)), "cv_brier_std_repeat": float(np.std(br, ddof=1)),
            "cv_n_wrong_side_mean": float(np.mean(wrong)), "logloss_clip": clip}


@lru_cache(maxsize=2)
def s003_pred(source="raw"):
    """S003 test predictions (R003 gmm2_hv). Loaded from the real archive when present
    (laptop); on a Colab VM (no experiments/ folder) recomputed exactly with the same code,
    splits and seeds: TransductiveGMM('highvar', 2), official CV, fold_mean."""
    f = paths.ROOT / "experiments" / S003_RUN / "test_pred.npy"
    if source == "raw" and f.exists():
        return np.load(f)
    import yaml
    from ssc import data
    from ssc.transductive import TransductiveGMM
    from ssc.validation import Scheme, cross_validate, make_splits
    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
    scheme = Scheme.from_dict(official)
    ds = data.load(source)
    cv = cross_validate(TransductiveGMM("highvar", 2, source=source), ds.X, ds.y, make_splits(ds.y, scheme), scheme,
                        X_test=ds.X_test)
    return cv.test_pred


def boundary_stats(pred, ref=None, lo=0.1, hi=0.9):
    """How a candidate re-orders the S003 boundary rows (S003 prediction in (lo, hi))."""
    from scipy.stats import kendalltau, spearmanr
    ref = s003_pred() if ref is None else ref
    b = (ref > lo) & (ref < hi)
    out = {"n_boundary_s003": int(b.sum()),
           "n_boundary_self": int(((pred > lo) & (pred < hi)).sum()),
           "spearman_all": float(spearmanr(pred, ref)[0]),
           "spearman_boundary": float(spearmanr(pred[b], ref[b])[0]),
           "kendall_boundary": float(kendalltau(pred[b], ref[b])[0]),
           "n_side_flips_all": int(((pred >= 0.5) != (ref >= 0.5)).sum()),
           "frac_pos_half": float((pred >= 0.5).mean())}
    return out


def expected_auc(score, p):
    """Expected AUC of `score` if the labels were Bernoulli(p) independently (p = some
    model's posterior): sum_{i,j} p_i (1-p_j) 1[s_i > s_j] / sum_{i!=j} p_i (1-p_j),
    computed in O(n log n) via ranks (ties count 1/2)."""
    from scipy.stats import rankdata
    s = np.asarray(score)
    p = np.asarray(p, dtype=float)
    o = np.argsort(s, kind="mergesort")
    s_o, p_o = s[o], p[o]
    q_o = 1 - p_o
    # for each i, sum of q_j over j with s_j < s_i, plus half of ties (excluding itself)
    ranks = rankdata(s_o, method="min") - 1  # number strictly below
    cq = np.concatenate([[0], np.cumsum(q_o)])
    below = cq[ranks]
    rmax = rankdata(s_o, method="max")
    tie = cq[rmax] - cq[ranks] - q_o
    num = (p_o * (below + 0.5 * tie)).sum()
    den = p.sum() * q_o.sum() - (p * (1 - p)).sum()
    return float(num / den)


def auc(y, s):
    return float(roc_auc_score(y, s))


# ============================================================================= array-level API
# (same algorithms as above, on an arbitrary 20,000 x 150 matrix: used for replica datasets)
def highvar_cols(X_all, ratio=1.5):
    s = X_all.std(0)
    return tuple(int(j) for j in np.where(s > ratio * np.median(s))[0])


def basis_arr(X_all, cols, rel_tol=1e-8):
    cols = np.array(cols, dtype=int)
    Z, mu, sd = standardize(X_all[:, cols])
    W, w = pca_basis(Z, rel_tol)
    return (Z - Z.mean(0)) @ W


def gmm_fit_arr(U, K=2, cov_type="full", n_init=5, seed=0, reg=1e-6, polish=True, max_iter=2000):
    g = GaussianMixture(K, covariance_type=cov_type, n_init=n_init, random_state=seed, reg_covar=reg,
                        tol=1e-3, max_iter=5000).fit(U)
    if not polish:
        return {"post": g.predict_proba(U), "loglik": float(g.score(U)), "sk": g}
    return em(U, g.predict_proba(U), cov_type, reg, tol=1e-10, max_iter=max_iter)


def oriented(post, y_train):
    """P(class 1) for K=2 from a 2-column posterior, oriented with the training labels."""
    r = post[:, 1]
    if np.corrcoef(r[: len(y_train)], y_train)[0, 1] < 0:
        r = post[:, 0]
    return r


def discover_arr(X_all, start=BLOCK, max_iter=6, K=2, n_init=5, seed=0, verbose=False):
    """Label-free forward discovery (R100 rule): S <- S + {j : conditional LRT of x_j given
    PCA(S) with soft labels from the K=2 GMM on S exceeds (dim+2) log n, and x_j is not an
    exact linear combination of S}. Returns (S, history list)."""
    Z, _, _ = standardize(X_all)
    n, P = Z.shape
    S = list(start)
    hist = []
    for it in range(max_iter):
        U = basis_arr(X_all, S)
        fit = gmm_fit_arr(U, K, "full", n_init, seed)
        r = fit["post"][:, 0]
        cand = [j for j in range(P) if j not in S]
        res = class_dependence(Z[:, cand], r, U)
        ok = (res["dbic"] < 0) & (res["r2"] <= 1 - 1e-7)
        add = [cand[i] for i in np.where(ok)[0]]
        hist.append({"iter": it, "n_S": len(S), "dim": U.shape[1], "added": add})
        if verbose:
            print(it, len(S), add, flush=True)
        if not add:
            break
        S = S + add
    return S, hist


class ReplicaGenerator:
    """Parametric generator fitted to the real 20,000 x 150 feature matrix.

    Signal set S: K-component full-covariance Gaussian mixture in the PCA basis of the
    standardised S columns (component = class). Every other column: class-independent
    Gaussian linear regression on the S coordinates with a full residual covariance
    (keeps correlations and exact linear relations of the rest). Values are
    un-standardised with the real means / sds and rounded to 6 decimals like the CSV.
    Class of component k = majority training label (K=2: the component with the larger
    training class-1 rate is class 1)."""

    def __init__(self, signal_cols, K=2, source="raw", seed=0, n_init=5, reg=1e-6, y_train=None):
        X_all = unlabeled_features(source)
        self.n = len(X_all)
        Z, self.mu, self.sd = standardize(X_all)
        self.S = np.array(signal_cols, dtype=int)
        self.R = np.array([j for j in range(X_all.shape[1]) if j not in set(self.S)], dtype=int)
        ZS = Z[:, self.S]
        self.W, _ = pca_basis(ZS)
        U = (ZS - ZS.mean(0)) @ self.W
        self.fit = gmm_fit_arr(U, K, "full", n_init, seed, reg)
        post = self.fit["post"]
        if y_train is None:
            from ssc import data
            y_train = data.load(source).y
        rate = component_rates(post[:N_TRAIN], np.asarray(y_train))
        self.comp_class = (rate > 0.5).astype(int)
        D = np.c_[np.ones(self.n), U]
        ZR = Z[:, self.R]
        self.B, *_ = np.linalg.lstsq(D, ZR, rcond=None)
        E = ZR - D @ self.B
        Ce = np.cov(E, rowvar=False)
        w, V = np.linalg.eigh(Ce)
        self.Esq = V * np.sqrt(np.clip(w, 0, None))[None]
        self.ZS_mean = ZS.mean(0)

    def sample(self, rng, n=None):
        n = n or self.n
        f = self.fit
        K = len(f["weights"])
        comp = rng.choice(K, size=n, p=f["weights"] / f["weights"].sum())
        U = np.empty((n, len(f["means"][0])))
        for k in range(K):
            m = comp == k
            L_ = np.linalg.cholesky(f["covs"][k])
            U[m] = f["means"][k] + rng.standard_normal((m.sum(), U.shape[1])) @ L_.T
        Z = np.empty((n, len(self.mu)))
        Z[:, self.S] = U @ self.W.T + self.ZS_mean
        Z[:, self.R] = np.c_[np.ones(n), U] @ self.B + rng.standard_normal((n, self.Esq.shape[1])) @ self.Esq.T
        X = np.round(Z * self.sd + self.mu, 6)
        y = self.comp_class[comp]
        # Bayes posterior P(class 1 | x) under the generator (uses the generator's own parameters)
        lp = component_logpdf(U, f["means"], f["covs"]) + np.log(f["weights"])[None]
        post = np.exp(lp - logsumexp(lp, axis=1)[:, None])
        bayes = post[:, self.comp_class == 1].sum(1)
        return X, y, bayes


# ----------------------------------------------------------------------------- more procedures (array level)
def class1_prob(post, idx_train, y_train, alpha=1.0):
    """Score = sum_k P(k | x) rate_k with rates from the labelled rows (TransductiveGMM rule)."""
    rate = component_rates(post[idx_train], np.asarray(y_train), alpha)
    return post @ rate


def qda_logpost(U, p1, reg=1e-6):
    """Weighted Gaussian class model (QDA) fitted with soft labels p1 on all rows;
    returns P(class 1 | x) for all rows."""
    resp = np.c_[1 - p1, p1]
    w, m, c = m_step(U, resp, "full", reg)
    lp = component_logpdf(U, m, c) + np.log(w)[None]
    return np.exp(lp[:, 1] - logsumexp(lp, axis=1))


def selftrain_qda_arr(U, p_init, idx_train, y_train, thr=0.99, n_iter=5, reg=1e-6):
    """Self-training: hard pseudo-labels on rows with P(class 1) > thr or < 1 - thr (true
    labels for the labelled rows), QDA fitted on those rows only, repeated n_iter times.
    (Classification-EM on the confident core: expected to shrink the class covariances.)"""
    p = p_init.copy()
    for _ in range(n_iter):
        lab = np.full(len(U), -1)
        lab[p > thr] = 1
        lab[p < 1 - thr] = 0
        lab[idx_train] = y_train
        m = lab >= 0
        resp = np.c_[(lab[m] == 0), (lab[m] == 1)].astype(float)
        w, mu, c = m_step(U[m], resp, "full", reg)
        lp = component_logpdf(U, mu, c) + np.log(w)[None]
        p = np.exp(lp[:, 1] - logsumexp(lp, axis=1))
    return p


def quad_logit_arr(U, p_soft, idx_train, y_train, C=1.0, train_weight=1.0):
    """Discriminative model on pseudo-labels: logistic regression on degree-2 polynomial
    features of U, fitted on all rows with soft labels p_soft (each row twice, weights p and
    1-p); labelled rows use their true label with weight train_weight. Returns P(class 1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    F = StandardScaler().fit_transform(PolynomialFeatures(2, include_bias=False).fit_transform(U))
    p = p_soft.copy()
    p[idx_train] = y_train
    wts = np.ones(len(U))
    wts[idx_train] = train_weight
    Xd = np.vstack([F, F])
    yd = np.r_[np.ones(len(U)), np.zeros(len(U))]
    wd = np.r_[p * wts, (1 - p) * wts]
    keep = wd > 1e-12
    lr = LogisticRegression(C=C, max_iter=5000).fit(Xd[keep], yd[keep], sample_weight=wd[keep])
    return lr.predict_proba(F)[:, 1]


def bagged_post_arr(U, ref_fit, B=10, seed=0, reg=1e-6):
    """Bootstrap-aggregated mixture posterior: refit (EM from the reference fit's
    responsibilities restricted to the resample) on B bootstrap resamples of the rows and
    average the posterior of every row (components keep the reference labelling)."""
    rng = np.random.default_rng(seed)
    n = len(U)
    acc = np.zeros_like(ref_fit["post"])
    for b in range(B):
        ix = rng.integers(0, n, n)
        f = em(U[ix], ref_fit["post"][ix], "full", reg, tol=1e-9)
        lp = component_logpdf(U, f["means"], f["covs"]) + np.log(f["weights"])[None]
        acc += np.exp(lp - logsumexp(lp, axis=1)[:, None])
    return acc / B


@lru_cache(maxsize=16)
def bagged_posterior(columns: tuple, B: int = 10, seed: int = 0, source: str = "raw"):
    *_, U = basis(columns, source)
    _, _, fit = polished_posterior(columns, 2, "full", 5, 0, 1e-6, source)
    return bagged_post_arr(U, fit, B, seed)


class PostTableModel(ClassifierMixin, BaseEstimator):
    """Generic transductive scorer: a label-free posterior table over the 20,000 rows
    (from `table_fn()`, cached) + the fold's training labels -> scores for any rows that
    are part of the 20,000 (training rows, validation rows, test rows).

    mode: 'rates'      sum_k P(k|x) rate_k (TransductiveGMM rule, K >= 2)
          'selftrain'  self-training QDA started from the K=2 posterior
          'quadlogit'  quadratic logistic regression on soft pseudo-labels
    Rows are located by exact feature values (train rows) or by position in the test file."""

    def __init__(self, columns=BLOCK, table="polished", K=2, mode="rates", thr=0.99, n_iter=5, C=1.0,
                 B=10, source="raw"):
        self.columns = columns
        self.table = table
        self.K = K
        self.mode = mode
        self.thr = thr
        self.n_iter = n_iter
        self.C = C
        self.B = B
        self.source = source

    def _post(self):
        cols = tuple(int(c) for c in self.columns)
        if self.table == "polished":
            return polished_posterior(cols, self.K, "full", 5, 0, 1e-6, self.source)[0]
        if self.table == "bagged":
            return bagged_posterior(cols, self.B, 0, self.source)
        raise ValueError(self.table)

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        idx = rows_in_all(X, self.source)
        post = self._post()
        p = class1_prob(post, idx, y)
        cols = tuple(int(c) for c in self.columns)
        if self.mode == "rates":
            self.score_ = p
        else:
            *_, U = basis(cols, self.source)
            if self.mode == "selftrain":
                self.score_ = selftrain_qda_arr(U, p, idx, y, self.thr, self.n_iter)
            elif self.mode == "quadlogit":
                self.score_ = quad_logit_arr(U, p, idx, y, self.C)
            else:
                raise ValueError(self.mode)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        pos = locate_rows(X, self.source)
        p1 = np.clip(self.score_[pos], 0, 1)
        return np.c_[1 - p1, p1]


@lru_cache(maxsize=4)
def _all_row_index(source: str = "raw"):
    X_all = unlabeled_features(source)
    return {X_all[i].tobytes(): i for i in range(len(X_all))}


def locate_rows(X, source: str = "raw") -> np.ndarray:
    idx = _all_row_index(source)
    X = np.asarray(X, dtype=float)
    out = np.array([idx.get(r.tobytes(), -1) for r in X])
    if (out < 0).any():
        raise ValueError(f"{int((out < 0).sum())} rows are not part of the train+test matrix")
    return out


CONN20 = (6, 8, 15, 17, 26, 30, 46, 49, 52, 57, 61, 64, 67, 77, 80, 98, 100, 112, 142, 144)  # R100 connected set


class SupSelectGMM(ClassifierMixin, BaseEstimator):
    """Supervised in-fold selection + transductive GMM: rank all 150 features by the marginal
    two-class Gaussian LRT on the fold's training rows (hard labels, df 2), keep the top k,
    then K=2 full GMM (polished) on the 20,000 rows in the PCA basis of the selected set."""

    def __init__(self, k=13, source="raw"):
        self.k = k
        self.source = source

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        X_all = unlabeled_features(self.source)
        mu, sd = X_all.mean(0), X_all.std(0)
        Z = (np.asarray(X, dtype=float) - mu) / sd
        res = class_dependence(Z, y.astype(float), None)
        self.cols_ = tuple(sorted(int(j) for j in np.argsort(-res["lrt"])[: self.k]))
        post, _, fit = polished_posterior(self.cols_, 2, "full", 5, 0, 1e-6, self.source)
        idx = rows_in_all(X, self.source)
        self.rate_ = component_rates(post[idx], y)
        self.fit_ = fit
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        U = transform(X, self.cols_, self.source)
        f = self.fit_
        lp = component_logpdf(U, f["means"], f["covs"]) + np.log(f["weights"])[None]
        R = np.exp(lp - logsumexp(lp, axis=1)[:, None])
        p1 = np.clip(R @ self.rate_, 0, 1)
        return np.c_[1 - p1, p1]


# ----------------------------------------------------------------------------- official-CV run helper
def official_run(prefix, name, make, desc, hyp, source="raw", author=None, n_jobs=1, extra_cfg=None):
    """Evaluate one candidate with the OFFICIAL protocol and archive it as an experiment run
    (test_pred.npy = fold_mean) with the label-free / non-saturated extra metrics."""
    import yaml
    from ssc import data
    from ssc.tracking import Run
    from ssc.validation import Scheme, cross_validate, make_splits
    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
    scheme = Scheme.from_dict(official)
    ds = data.load(source)
    splits = make_splits(ds.y, scheme)
    s3 = s003_pred(source)
    ref_post, _, _ = polished_posterior(BLOCK, 2, "full", 5, 0, 1e-6, source)
    ref_test = oriented(ref_post, ds.y)[N_TRAIN:]
    cfg = {"name": f"{prefix}_{name}", "description": desc, "model_code": f"research/{prefix}_*.py + research/lib_mix.py",
           "cv": scheme.to_dict(), "test_pred": "fold_mean", "uses_unlabeled_test_features": True, **(extra_cfg or {})}
    with Run(f"{prefix}_{name}", cfg, kind="experiment", source=source, author=author, hypothesis=hyp) as run:
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
        s = run.save_cv(cv)
        tp = cv.test_pred
        extra = {"official_cv": True, **cv_extra_metrics(cv, ds.y), **boundary_stats(tp, s3),
                 "expected_auc_under_ref": expected_auc(tp, ref_test),
                 "max_abs_diff_vs_s003": float(np.abs(tp - s3).max()), "uses_unlabeled_test_features": True}
        run.finish(extra_summary=extra)
        return {"candidate": name, "run_id": run.run_id, "auc_mean": s["auc_mean"], **extra}
