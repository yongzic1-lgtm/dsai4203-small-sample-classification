"""Transductive / semi-supervised classifiers that use the 19,750 unlabeled test rows.

Why this is legitimate and leakage-free: the unsupervised steps (feature
standardisation, the high-variance feature rule, PCA, Gaussian-mixture fitting)
use only FEATURES of train+test rows, never labels. Labels enter only through
the training part of each CV fold (mapping mixture components to classes), so a
CV score of these models is an honest estimate. Using unlabeled test features is
not hand-labelling (Kaggle foundational rule 4b) and uses no external data.

The label-free parts are identical in every fold, so they are cached per
parameter set (saves refitting the same mixture 100 times).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture

from . import paths
from .data import FEATURES


@lru_cache(maxsize=4)
def unlabeled_features(source: str = "raw") -> np.ndarray:
    """Train + test feature matrix (rows: train then test), as float array."""
    import pandas as pd
    d = paths.data_dir(source)
    tr = pd.read_csv(d / "train.csv")
    te = pd.read_csv(d / "test.csv")
    tr.columns = [str(c) for c in tr.columns]
    te.columns = [str(c) for c in te.columns]
    return np.vstack([tr[FEATURES].to_numpy(float), te[FEATURES].to_numpy(float)])


def resolve_columns(columns, X_all: np.ndarray) -> np.ndarray:
    """'all' | 'highvar' | 'highvar:<ratio>' | list of column indices / names."""
    if columns in (None, "all"):
        return np.arange(X_all.shape[1])
    if isinstance(columns, str) and columns.startswith("highvar"):
        ratio = float(columns.split(":", 1)[1]) if ":" in columns else 1.5
        std = X_all.std(0)
        return np.where(std > ratio * np.median(std))[0]
    return np.array([int(c) for c in columns])


@lru_cache(maxsize=64)
def _basis(source: str, columns_key: tuple, pca_dims: int | None):
    X_all = unlabeled_features(source)
    cols = np.array(columns_key)
    B = X_all[:, cols]
    mu, sd = B.mean(0), B.std(0)
    Z = (B - mu) / sd
    Vt = None
    if pca_dims:
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        Vt = Vt[:pca_dims]
        Z = Z @ Vt.T
    return cols, mu, sd, Vt, Z


@lru_cache(maxsize=64)
def _mixture(source: str, columns_key: tuple, pca_dims: int | None, n_components: int,
             covariance_type: str, n_init: int, random_state: int, reg_covar: float):
    *_, Z = _basis(source, columns_key, pca_dims)
    return GaussianMixture(n_components, covariance_type=covariance_type, n_init=n_init,
                           random_state=random_state, reg_covar=reg_covar, max_iter=1000).fit(Z)


class TransductiveGMM(ClassifierMixin, BaseEstimator):
    """Gaussian mixture fitted on train+test FEATURES (no labels), then each component's
    class-1 rate is estimated from the training labels with Beta(alpha*p0, alpha*(1-p0))
    smoothing. Score(x) = sum_k P(k | x) * rate_k.

    Inputs X must be the raw 150-column feature matrix (use scaler: none, no selector);
    column choice, standardisation and PCA are done internally on train+test features.
    """

    def __init__(self, columns="highvar", n_components: int = 2, covariance_type: str = "full",
                 pca_dims: int | None = None, n_init: int = 5, alpha: float = 1.0,
                 reg_covar: float = 1e-6, source: str = "raw", random_state: int = 0):
        self.columns = columns
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.pca_dims = pca_dims
        self.n_init = n_init
        self.alpha = alpha
        self.reg_covar = reg_covar
        self.source = source
        self.random_state = random_state

    def _key(self):
        cols = resolve_columns(self.columns, unlabeled_features(self.source))
        return tuple(int(c) for c in cols)

    def _transform(self, X):
        cols, mu, sd, Vt, _ = _basis(self.source, self._key(), self.pca_dims)
        Z = (np.asarray(X, dtype=float)[:, cols] - mu) / sd
        return Z @ Vt.T if Vt is not None else Z

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.gmm_ = _mixture(self.source, self._key(), self.pca_dims, self.n_components,
                             self.covariance_type, self.n_init, self.random_state, self.reg_covar)
        R = self.gmm_.predict_proba(self._transform(X))
        p0 = y.mean()
        self.component_rate_ = (R.T @ y + self.alpha * p0) / (R.sum(0) + self.alpha)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        R = self.gmm_.predict_proba(self._transform(X))
        p1 = np.clip(R @ self.component_rate_, 0.0, 1.0)
        return np.c_[1 - p1, p1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SubspaceModel(ClassifierMixin, BaseEstimator):
    """Supervised model on a label-free column subset (e.g. 'highvar'), standardised with
    train+test statistics. Lets supervised baselines use the same subspace as TransductiveGMM."""

    def __init__(self, estimator=None, columns="highvar", pca_dims: int | None = None, source: str = "raw"):
        self.estimator = estimator
        self.columns = columns
        self.pca_dims = pca_dims
        self.source = source

    def _transform(self, X):
        key = tuple(int(c) for c in resolve_columns(self.columns, unlabeled_features(self.source)))
        cols, mu, sd, Vt, _ = _basis(self.source, key, self.pca_dims)
        Z = (np.asarray(X, dtype=float)[:, cols] - mu) / sd
        return Z @ Vt.T if Vt is not None else Z

    def fit(self, X, y):
        from sklearn.base import clone
        self.classes_ = np.unique(y)
        self.estimator_ = clone(self.estimator).fit(self._transform(X), y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._transform(X))

    def decision_function(self, X):
        return self.estimator_.decision_function(self._transform(X))

    def predict(self, X):
        return self.estimator_.predict(self._transform(X))
