"""Shared helpers for the foundation-model / Bayesian research scripts (r250-r299, label "fm").

Contents
  * extra Python environments on the Colab VM (TabPFN / TabICL / NumPyro need torch or jax,
    which are not in the pinned project env): one separate uv venv per stack, built from the
    pinned project requirements plus the extra packages, so the shared /content/venv is never
    modified. Scripts re-exec themselves inside that venv.
  * token hygiene: every foundation-model call runs with all HuggingFace / TabPFN tokens
    removed and browser login disabled, so a gated model fails instead of authenticating.
  * BayesQDA: class-conditional Normal-Inverse-Wishart model, Student-t posterior predictive,
    hyperparameters by empirical Bayes (closed-form marginal likelihood) inside each fold.
  * TrainHighVar: the label-free "std > 1.5 x median std" block rule computed on the fitting
    rows only (fully inductive; no unlabeled test rows).
  * TransductiveBGMM: like ssc.transductive.TransductiveGMM but with a variational
    Dirichlet-process Gaussian mixture (sklearn BayesianGaussianMixture).
  * HorseshoeLogReg: regularised-horseshoe logistic regression with NumPyro NUTS.
  * reference_agreement: agreement of test scores with the S003 reference on boundary rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import gammaln, multigammaln
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ON_VM = Path("/content").is_dir() and os.name != "nt"

TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "TABPFN_TOKEN",
              "HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


# --------------------------------------------------------------------------- extra environments
def token_free_env(base: dict | None = None) -> dict:
    """Environment without any model-hub credentials: gated downloads must fail, not log in."""
    env = dict(os.environ if base is None else base)
    for k in TOKEN_VARS:
        env.pop(k, None)
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["HF_TOKEN_PATH"] = "/nonexistent/hf_token"     # ignore any cached HF login
    env["TABPFN_NO_BROWSER"] = "1"                      # never open a PriorLabs login flow
    env["TABPFN_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


def apply_token_free_env() -> None:
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in token_free_env().items()
                       if k.startswith(("HF_HUB_", "HF_TOKEN_PATH", "TABPFN_"))})


def _uv() -> str:
    for c in (shutil.which("uv"), "/usr/local/bin/uv", "/root/.local/bin/uv", "/usr/bin/uv"):
        if c and Path(c).exists():
            return c
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
    c = shutil.which("uv")
    if not c:
        raise RuntimeError("uv not found on the VM")
    return c


def venv_python(name: str) -> Path:
    return Path(f"/content/venv_{name}/bin/python")


def ensure_venv(name: str, packages: list[str], extra_index: str | None = None) -> tuple[Path, dict]:
    """Create /content/venv_<name> = pinned project requirements + `packages` (idempotent).

    First try with the pinned requirements as hard requirements (so numpy/pandas/sklearn keep the
    project versions); if the resolver refuses, fall back to installing `packages` with the pinned
    file as *constraints* dropped, and record that in the returned info."""
    if not ON_VM:
        raise RuntimeError("extra environments are only built on the Colab VM")
    uv = _uv()
    py = venv_python(name)
    reqs = ROOT / "colab" / "requirements-colab.txt"
    want = hashlib.sha256((json.dumps(packages) + reqs.read_text() + str(extra_index)).encode()).hexdigest()[:16]
    marker = py.parent.parent / ".fm_env"
    info = {"venv": str(py.parent.parent), "packages": packages, "hash": want}
    if marker.exists() and json.loads(marker.read_text()).get("hash") == want:
        info.update(json.loads(marker.read_text()))
        return py, info
    if not py.exists():
        subprocess.run([uv, "venv", str(py.parent.parent), "--python", "3.11", "-q"], check=True)
    base = [uv, "pip", "install", "--python", str(py), "-q"]
    if extra_index:
        base += ["--extra-index-url", extra_index]
    r = subprocess.run(base + ["-r", str(reqs)] + packages, capture_output=True, text=True)
    info["pinned_resolution"] = r.returncode == 0
    if r.returncode != 0:
        print("[lib_fm] pinned resolution failed, installing without the project pins:\n" + r.stderr[-2000:])
        subprocess.run(base + packages, check=True)
        subprocess.run(base + ["PyYAML", "tabulate", "matplotlib"], check=False)
    subprocess.run([uv, "pip", "install", "--python", str(py), "-q", "--no-deps", "-e", str(ROOT)], check=True)
    fr = subprocess.run([uv, "pip", "freeze", "--python", str(py)], capture_output=True, text=True)
    info["freeze"] = fr.stdout.splitlines()
    marker.write_text(json.dumps(info))
    return py, info


def reexec_in(py: Path) -> None:
    """Re-run the current script with interpreter `py` (same argv and environment)."""
    # compare venv prefixes, not resolved interpreters (all uv venvs share one base python binary)
    if Path(sys.prefix) == Path(py).parent.parent or os.environ.get("FM_REEXEC") == str(py):
        return
    env = dict(os.environ, FM_REEXEC=str(py))
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(str(py), [str(py)] + sys.argv, env)


def freeze_versions(names=("torch", "tabpfn", "tabicl", "jax", "jaxlib", "numpyro", "huggingface_hub",
                           "scikit-learn", "numpy", "pandas", "scipy")) -> dict:
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


# --------------------------------------------------------------------------- official CV helpers
def code_hash(*files) -> dict:
    """sha256[:16] of this library and the given script files: exact code identity of a run even when
    the job was launched with `remote.py run --no-sync` (meta.json then has no code bundle hash)."""
    out = {}
    for f in (__file__, *files):
        f = Path(f)
        out[f.name] = hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
    return out


def official_setup(source: str):
    """(dataset, scheme, splits) of the official protocol (project.yaml official_cv)."""
    import yaml
    from ssc import data, paths
    from ssc.validation import Scheme, make_splits
    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
    scheme = Scheme.from_dict(official)
    ds = data.load(source)
    return ds, scheme, make_splits(ds.y, scheme)


R003_REFERENCES = {
    "r003_gmm2_hv": "TransductiveGMM K=2 full on the high-variance block (= S003, public 0.99923)",
    "r003_sup_qda_hv": "QDA(reg 0.1) on the high-variance block",
    "r003_sup_logreg_hv": "L2 logistic (C=1) on the high-variance block",
    "r003_sup_l1_all": "L1 logistic C=0.1 on all 150 features",
}


def r003_reference(name: str, ds, splits, scheme, source: str, n_jobs: int = 1):
    """Recompute one R003 configuration with identical code and splits (deterministic), so paired
    comparisons and S003 agreement can be computed on the VM without the local archive."""
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from ssc.models import build_pipeline
    from ssc.transductive import SubspaceModel, TransductiveGMM
    from ssc.validation import cross_validate
    make = {
        "r003_gmm2_hv": lambda: TransductiveGMM("highvar", 2, source=source),
        "r003_sup_qda_hv": lambda: SubspaceModel(QuadraticDiscriminantAnalysis(reg_param=0.1), "highvar", source=source),
        "r003_sup_logreg_hv": lambda: SubspaceModel(LogisticRegression(C=1.0, max_iter=5000), "highvar", source=source),
        "r003_sup_l1_all": lambda: build_pipeline({"model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 1.0}}}),
    }[name]
    nj = 1 if name.startswith("r003_gmm") else n_jobs
    return cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=nj)


def paired_table(cands: dict, refs: dict) -> "list[dict]":
    """Paired comparison (ssc.validation.paired_compare) of every candidate CVResult vs every reference."""
    from ssc.validation import paired_compare
    rows = []
    for cn, cv in cands.items():
        for rn, rcv in refs.items():
            c = paired_compare(cv.fold_scores, rcv.fold_scores)
            rows.append({"candidate": cn, "reference": rn, "auc_candidate": float(cv.fold_scores["auc"].mean()),
                         "auc_reference": float(rcv.fold_scores["auc"].mean()), **c})
    return rows


# --------------------------------------------------------------------------- agreement with S003
def reference_agreement(pred, ref, lo: float = 0.1, hi: float = 0.9) -> dict:
    """Compare test scores `pred` with the reference posterior `ref` (S003 = TransductiveGMM K=2 block).

    boundary rows: lo < ref < hi. 'pseudo_auc_confident' treats the reference's confident
    (non-boundary) rows as labels: an estimate of how well `pred` recovers the mixture classes on
    ~19.5k test rows (only meaningful because S003 scored 0.99923 on the public LB)."""
    from scipy.stats import kendalltau, spearmanr
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    b = (ref > lo) & (ref < hi)
    conf = ~b   # S003 scores live in about [0.004, 0.98] (Beta-smoothed component rates): non-boundary rows
    out = {"n_test": int(len(ref)), "n_boundary": int(b.sum()),
           "spearman_all": float(spearmanr(pred, ref).statistic),
           "pseudo_auc_vs_ref_hard": float(roc_auc_score(ref > 0.5, pred)),
           "pseudo_auc_confident": float(roc_auc_score(ref[conf] > 0.5, pred[conf])) if conf.any() else float("nan"),
           "frac_pred_in_0.1_0.9": float(((pred > lo) & (pred < hi)).mean())}
    if b.sum() > 2:
        out["spearman_boundary"] = float(spearmanr(pred[b], ref[b]).statistic)
        out["kendall_boundary"] = float(kendalltau(pred[b], ref[b]).statistic)
        out["side_agree_boundary"] = float(((pred[b] > 0.5) == (ref[b] > 0.5)).mean())
        out["mean_abs_diff_boundary"] = float(np.abs(pred[b] - ref[b]).mean())
    out["side_agree_all"] = float(((pred > 0.5) == (ref > 0.5)).mean())
    return out


def oof_calibration(oof: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> dict:
    """CV log-loss and Brier score from the out-of-fold matrix (n_repeats, n) of probabilities.
    Only meaningful when the scores are probabilities in [0, 1]."""
    p = np.clip(np.asarray(oof, float), eps, 1 - eps)
    y = np.asarray(y, float)[None, :]
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"cv_logloss": float(np.nanmean(ll)), "cv_brier": float(np.nanmean((p - y) ** 2)),
            "cv_logloss_std_repeat": float(np.nanstd(np.nanmean(ll, 1), ddof=1))}


# --------------------------------------------------------------------------- inductive block rule
class TrainHighVar(TransformerMixin, BaseEstimator):
    """Keep columns with std > ratio * median std, computed on the FITTING rows only, then
    standardise with the fitting rows' mean/std. No unlabeled test rows, no labels."""

    def __init__(self, ratio: float = 1.5):
        self.ratio = ratio

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        sd = X.std(0)
        self.columns_ = np.where(sd > self.ratio * np.median(sd))[0]
        self.mean_ = X[:, self.columns_].mean(0)
        self.scale_ = X[:, self.columns_].std(0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return (np.asarray(X, float)[:, self.columns_] - self.mean_) / self.scale_


# --------------------------------------------------------------------------- Bayesian QDA (NIW)
def _niw_posterior(X, m0, kappa0, nu0, Psi0):
    n, d = X.shape
    xbar = X.mean(0)
    S = (X - xbar).T @ (X - xbar)
    kn, nn = kappa0 + n, nu0 + n
    mn = (kappa0 * m0 + n * xbar) / kn
    diff = (xbar - m0)[:, None]
    Psin = Psi0 + S + (kappa0 * n / kn) * (diff @ diff.T)
    return mn, kn, nn, Psin


def _logdet(A):
    sign, ld = np.linalg.slogdet(A)
    return ld if sign > 0 else -np.inf


def niw_log_marginal(X, m0, kappa0, nu0, Psi0) -> float:
    """log p(X) under x ~ N(mu, Sigma), mu|Sigma ~ N(m0, Sigma/kappa0), Sigma ~ IW(Psi0, nu0)."""
    n, d = X.shape
    _, kn, nn, Psin = _niw_posterior(X, m0, kappa0, nu0, Psi0)
    return float(-0.5 * n * d * np.log(np.pi) + multigammaln(nn / 2, d) - multigammaln(nu0 / 2, d)
                 + 0.5 * nu0 * _logdet(Psi0) - 0.5 * nn * _logdet(Psin) + 0.5 * d * (np.log(kappa0) - np.log(kn)))


def mvt_logpdf(X, loc, shape, df):
    d = len(loc)
    L = np.linalg.cholesky(shape)
    Z = np.linalg.solve(L, (X - loc).T)
    maha = (Z ** 2).sum(0)
    if df > 1e8:  # Gaussian limit
        return -0.5 * d * np.log(2 * np.pi) - np.log(np.diag(L)).sum() - 0.5 * maha
    return (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi)
            - np.log(np.diag(L)).sum() - 0.5 * (df + d) * np.log1p(maha / df))


class BayesQDA(ClassifierMixin, BaseEstimator):
    """Bayesian QDA: per class a Normal-Inverse-Wishart prior, scored with the exact posterior
    predictive (multivariate Student-t). Prior: m0 = training mean, Psi0 = (nu0 - d - 1) * S0 so that
    E[Sigma] = S0. With mode="eb" (kappa0, nu0, S0) are chosen by maximising the summed closed-form
    log marginal likelihood of both classes on the fitting rows (empirical Bayes, type-II ML), over
    a grid; S0 in {pooled within-class covariance, its diagonal, isotropic}. With mode="fixed" the
    given kappa0 / nu0_extra / prior are used.

    Rank: the 13-feature block has rank 12, so the fitting rows are first projected onto their
    non-null principal directions (SVD of the centred fitting rows, no labels)."""

    def __init__(self, mode: str = "eb", kappa0: float = 0.01, nu0_extra: float = 2.0, prior: str = "pooled",
                 kappa_grid=(0.001, 0.01, 0.1, 1.0), nu_extra_grid=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                 prior_grid=("pooled", "diag", "iso"), rank_tol: float = 1e-6, predictive: str = "t"):
        self.predictive = predictive  # "t" = exact posterior predictive; "plugin" = Gaussian at posterior means
        self.mode = mode
        self.kappa0 = kappa0
        self.nu0_extra = nu0_extra
        self.prior = prior
        self.kappa_grid = kappa_grid
        self.nu_extra_grid = nu_extra_grid
        self.prior_grid = prior_grid
        self.rank_tol = rank_tol

    def _S0(self, Z, y, kind):
        d = Z.shape[1]
        pooled = sum(np.cov(Z[y == c].T, bias=False) * ((y == c).sum() - 1) for c in (0, 1)) / (len(y) - 2)
        if kind == "pooled":
            return pooled
        if kind == "diag":
            return np.diag(np.diag(pooled))
        if kind == "iso":
            return np.eye(d) * np.trace(pooled) / d
        raise ValueError(kind)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.center_ = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.center_, full_matrices=False)
        keep = s > self.rank_tol * s[0]
        self.proj_ = Vt[keep].T
        Z = (X - self.center_) @ self.proj_
        d = Z.shape[1]
        m0 = Z.mean(0)
        if self.mode == "eb":
            grid = [(k, ne, pr) for k in self.kappa_grid for ne in self.nu_extra_grid for pr in self.prior_grid]
        else:
            grid = [(self.kappa0, self.nu0_extra, self.prior)]
        best, table = None, []
        S0s = {pr: self._S0(Z, y, pr) for pr in {g[2] for g in grid}}
        for k, ne, pr in grid:
            nu0 = d + 1 + ne
            Psi0 = (nu0 - d - 1) * S0s[pr]
            lml = sum(niw_log_marginal(Z[y == c], m0, k, nu0, Psi0) for c in (0, 1))
            table.append((k, ne, pr, lml))
            if best is None or lml > best[3]:
                best = (k, ne, pr, lml)
        self.eb_table_ = table
        self.kappa0_, self.nu0_extra_, self.prior_, self.log_marginal_ = best
        nu0 = d + 1 + self.nu0_extra_
        Psi0 = (nu0 - d - 1) * S0s[self.prior_]
        self.post_ = []
        for c in (0, 1):
            mn, kn, nn, Psin = _niw_posterior(Z[y == c], m0, self.kappa0_, nu0, Psi0)
            df = nn - d + 1
            if self.predictive == "plugin":   # Gaussian with the posterior-mean covariance (df -> inf limit)
                self.post_.append((mn, Psin / (nn - d - 1), 1e12))
            else:
                self.post_.append((mn, Psin * (kn + 1) / (kn * df), df))
        n1 = y.sum()
        self.log_prior_odds_ = np.log((n1 + 1) / (len(y) - n1 + 1))
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        Z = (np.asarray(X, float) - self.center_) @ self.proj_
        l0 = mvt_logpdf(Z, *self.post_[0])
        l1 = mvt_logpdf(Z, *self.post_[1])
        return l1 - l0 + self.log_prior_odds_

    def predict_proba(self, X):
        z = np.clip(self.decision_function(X), -700, 700)
        p1 = 1 / (1 + np.exp(-z))
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


# --------------------------------------------------------------------------- DP Gaussian mixture
@lru_cache(maxsize=16)
def fit_bgmm(source: str, columns_key: tuple, pca_dims, n_components: int, gamma: float, n_init: int,
             random_state: int, max_iter: int = 3000):
    from sklearn.mixture import BayesianGaussianMixture
    from ssc.transductive import _basis
    *_, Z = _basis(source, columns_key, pca_dims)
    return BayesianGaussianMixture(
        n_components=n_components, covariance_type="full",
        weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=gamma,
        n_init=n_init, max_iter=max_iter, tol=1e-4, init_params="kmeans",
        random_state=random_state).fit(Z)


class TransductiveBGMM(ClassifierMixin, BaseEstimator):
    """Variational Dirichlet-process Gaussian mixture (truncation n_components) fitted on train+test
    FEATURES only; component -> class rates from the fold's training labels with the same Beta
    smoothing as ssc.transductive.TransductiveGMM. Score(x) = sum_k q(k | x) * rate_k."""

    def __init__(self, columns="highvar", n_components: int = 10, gamma: float = 1.0, pca_dims=None,
                 n_init: int = 3, alpha: float = 1.0, source: str = "raw", random_state: int = 0):
        self.columns = columns
        self.n_components = n_components
        self.gamma = gamma
        self.pca_dims = pca_dims
        self.n_init = n_init
        self.alpha = alpha
        self.source = source
        self.random_state = random_state

    def _key(self):
        from ssc.transductive import resolve_columns, unlabeled_features
        return tuple(int(c) for c in resolve_columns(self.columns, unlabeled_features(self.source)))

    def _transform(self, X):
        from ssc.transductive import _basis
        cols, mu, sd, Vt, _ = _basis(self.source, self._key(), self.pca_dims)
        Z = (np.asarray(X, float)[:, cols] - mu) / sd
        return Z @ Vt.T if Vt is not None else Z

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.mix_ = fit_bgmm(self.source, self._key(), self.pca_dims, self.n_components, float(self.gamma),
                             self.n_init, self.random_state)
        R = self.mix_.predict_proba(self._transform(X))
        p0 = y.mean()
        self.component_rate_ = (R.T @ y + self.alpha * p0) / (R.sum(0) + self.alpha)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        R = self.mix_.predict_proba(self._transform(X))
        p1 = np.clip(R @ self.component_rate_, 0.0, 1.0)
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def plugin_mixture_loglik(Z, weights, means, covariances, min_weight: float = 0.0):
    """Log-likelihood of Z under a plug-in Gaussian mixture; components with weight <= min_weight
    are dropped and the rest renormalised. Returns (total loglik, n_components kept)."""
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    keep = np.where(weights > min_weight)[0]
    w = weights[keep] / weights[keep].sum()
    L = np.column_stack([np.log(w[i]) + multivariate_normal(means[k], covariances[k], allow_singular=False).logpdf(Z)
                         for i, k in enumerate(keep)])
    return float(logsumexp(L, axis=1).sum()), len(keep)


def mixture_bic(loglik: float, k: int, d: int, n: int) -> float:
    n_par = (k - 1) + k * d + k * d * (d + 1) / 2
    return float(-2 * loglik + n_par * np.log(n))


# --------------------------------------------------------------------------- horseshoe logistic
class HorseshoeLogReg(ClassifierMixin, BaseEstimator):
    """Logistic regression with the regularised horseshoe prior (Piironen & Vehtari 2017), sampled
    with NumPyro NUTS (non-centred). Inputs should be standardised (put a scaler before it).

        beta_j = z_j * tau * lt_j,  z_j ~ N(0, 1),  lambda_j ~ C+(0, 1),  tau ~ C+(0, tau0),
        lt_j^2 = c^2 lambda_j^2 / (c^2 + tau^2 lambda_j^2),  c^2 ~ Inv-Gamma(nu/2, nu s^2 / 2),
        tau0 = p0 / (p - p0) * sigma / sqrt(n),  sigma = 1 / sqrt(ybar (1 - ybar))  (pseudo-sigma).

    Scores = posterior mean of P(y = 1 | x)."""

    def __init__(self, p0: float = 10.0, slab_scale: float = 2.0, slab_df: float = 4.0,
                 num_warmup: int = 500, num_samples: int = 500, num_chains: int = 1,
                 target_accept: float = 0.95, max_tree_depth: int = 10, seed: int = 0):
        self.p0 = p0
        self.slab_scale = slab_scale
        self.slab_df = slab_df
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept = target_accept
        self.max_tree_depth = max_tree_depth
        self.seed = seed

    @staticmethod
    def _model(X, y, tau0, slab_scale, slab_df):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        n, p = X.shape
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 3.0))
        tau_u = numpyro.sample("tau_u", dist.HalfCauchy(1.0))
        tau = numpyro.deterministic("tau", tau0 * tau_u)
        lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(p)))
        c2_u = numpyro.sample("c2_u", dist.InverseGamma(0.5 * slab_df, 0.5 * slab_df))
        c2 = numpyro.deterministic("c2", slab_scale ** 2 * c2_u)
        lt = jnp.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        z = numpyro.sample("z", dist.Normal(jnp.zeros(p), 1.0))
        beta = numpyro.deterministic("beta", z * tau * lt)
        numpyro.sample("y", dist.Bernoulli(logits=alpha + X @ beta), obs=y)

    def fit(self, X, y):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        import jax
        import numpyro
        from numpyro.infer import MCMC, NUTS
        numpyro.set_host_device_count(max(1, self.num_chains))
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        n, p = X.shape
        ybar = y.mean()
        sigma = 1.0 / np.sqrt(ybar * (1 - ybar))
        tau0 = self.p0 / (p - self.p0) * sigma / np.sqrt(n)
        kernel = NUTS(self._model, target_accept_prob=self.target_accept, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains,
                    progress_bar=False, chain_method="sequential")
        mcmc.run(jax.random.PRNGKey(self.seed), X, y, float(tau0), float(self.slab_scale), float(self.slab_df),
                 extra_fields=("diverging",))
        s = mcmc.get_samples()
        self.alpha_ = np.asarray(s["alpha"])
        self.beta_ = np.asarray(s["beta"])
        self.tau_ = np.asarray(s["tau"])
        self.lam_ = np.asarray(s["lam"])
        self.c2_ = np.asarray(s["c2"])
        extra = mcmc.get_extra_fields()
        self.n_divergent_ = int(np.asarray(extra["diverging"]).sum()) if "diverging" in extra else -1
        self.tau0_ = tau0
        self.sigma_ = sigma
        self.n_fit_ = n
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = p
        if self.num_chains > 1:
            from numpyro.diagnostics import summary
            gs = mcmc.get_samples(group_by_chain=True)
            sm = summary({k: gs[k] for k in ("alpha", "tau", "beta")}, prob=0.9)
            self.rhat_max_ = float(max(np.nanmax(np.asarray(v["r_hat"])) for v in sm.values()))
            self.ess_min_ = float(min(np.nanmin(np.asarray(v["n_eff"])) for v in sm.values()))
        return self

    def predict_proba(self, X, chunk: int = 4096):
        X = np.asarray(X, float)
        out = np.empty(len(X))
        for i in range(0, len(X), chunk):
            eta = X[i:i + chunk] @ self.beta_.T + self.alpha_[None, :]
            out[i:i + chunk] = (1 / (1 + np.exp(-np.clip(eta, -50, 50)))).mean(1)
        return np.c_[1 - out, out]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def shrinkage(self) -> np.ndarray:
        """Posterior mean of 1 - kappa_j, kappa_j = 1 / (1 + n sigma^-2 tau^2 lt_j^2) (Piironen & Vehtari)."""
        lt2 = self.c2_[:, None] * self.lam_ ** 2 / (self.c2_[:, None] + self.tau_[:, None] ** 2 * self.lam_ ** 2)
        kappa = 1.0 / (1.0 + self.n_fit_ / self.sigma_ ** 2 * self.tau_[:, None] ** 2 * lt2)
        return (1 - kappa).mean(0)


class DropCollinear(TransformerMixin, BaseEstimator):
    """Keep a maximal linearly independent subset of columns of the fitting rows (QR with column
    pivoting; |R_kk| > tol * |R_00|). Labels are not used. Removes exact linear dependencies, so a
    regression coefficient vector becomes identifiable."""

    def __init__(self, tol: float = 1e-7):
        self.tol = tol

    def fit(self, X, y=None):
        from scipy.linalg import qr
        X = np.asarray(X, float)
        Xc = X - X.mean(0)
        _, R, piv = qr(Xc, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        rank = int((d > self.tol * d[0]).sum())
        self.columns_ = np.sort(piv[:rank])
        self.dropped_ = np.sort(piv[rank:])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.columns_]


class QuadBlock(TransformerMixin, BaseEstimator):
    """[all standardised features, squares and pairwise products of the block columns] (fold-internal
    standardisation; block = label-free column list passed in)."""

    def __init__(self, block=()):
        self.block = block

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mean_, self.scale_ = X.mean(0), X.std(0)
        Z = (X - self.mean_) / self.scale_
        Q = self._quad(Z)
        self.qmean_, self.qscale_ = Q.mean(0), Q.std(0) + 1e-12
        self.n_features_in_ = X.shape[1]
        return self

    def _quad(self, Z):
        B = Z[:, list(self.block)]
        iu = np.triu_indices(B.shape[1])
        return (B[:, :, None] * B[:, None, :])[:, iu[0], iu[1]]

    def transform(self, X):
        Z = (np.asarray(X, float) - self.mean_) / self.scale_
        Q = (self._quad(Z) - self.qmean_) / self.qscale_
        return np.hstack([Z, Q])
