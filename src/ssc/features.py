"""Feature scoring and selection.

Every selector here is an sklearn transformer, so it can sit inside a Pipeline
and is re-fitted on the training portion of each fold. Fitting a selector on
all 250 rows and cross-validating afterwards leaks validation labels into the
feature choice (extension E3 measures exactly how much).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


def abs_correlation(X, y, method: str = "pearson") -> np.ndarray:
    """|corr(feature_j, y)| for every column j. Constant columns score 0."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "spearman":
        X = np.apply_along_axis(rankdata, 0, X)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(axis=0) / denom
    return np.nan_to_num(np.abs(r), nan=0.0)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores; ties broken by lower column index."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return np.sort(order[: min(k, len(scores))])


def bootstrap_topk_frequency(X, y, k: int, n_boot: int = 500, random_state: int = 0,
                             method: str = "pearson") -> np.ndarray:
    """Fraction of bootstrap resamples (n rows drawn with replacement) in which
    each feature is among the top-k by |correlation|. Shape (n_features,)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = len(y)
    counts = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        counts[top_k_indices(abs_correlation(X[idx], y[idx], method), k)] += 1
    return counts / n_boot


class CorrelationTopK(SelectorMixin, BaseEstimator):
    """Keep the k features with the largest |correlation| with the target."""

    def __init__(self, k: int | str = 20, method: str = "pearson"):
        self.k = k
        self.method = method

    def fit(self, X, y):
        X = validate_data(self, X, dtype=float)  # sets n_features_in_ (+ feature_names_in_ for DataFrames)
        self.scores_ = abs_correlation(X, y, self.method)
        k = self.n_features_in_ if self.k == "all" else int(self.k)
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[top_k_indices(self.scores_, k)] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


class StabilityTopK(SelectorMixin, BaseEstimator):
    """Keep features whose bootstrap top-k selection frequency >= threshold,
    computed only from the data passed to fit (i.e. inside the fold)."""

    def __init__(self, k: int = 20, n_boot: int = 200, threshold: float = 0.5,
                 method: str = "pearson", random_state: int = 0):
        self.k = k
        self.n_boot = n_boot
        self.threshold = threshold
        self.method = method
        self.random_state = random_state

    def fit(self, X, y):
        X = validate_data(self, X, dtype=float)
        self.frequency_ = bootstrap_topk_frequency(X, y, self.k, self.n_boot, self.random_state, self.method)
        mask = self.frequency_ >= self.threshold
        if not mask.any():  # never return an empty feature set
            mask[np.argmax(self.frequency_)] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


class FixedColumns(SelectorMixin, BaseEstimator):
    """Keep a fixed list of column indices chosen in advance.

    Warning: if the list was chosen by looking at the labels of rows that are
    later used for validation, the CV score is optimistically biased."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        validate_data(self, X, dtype=None)
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[list(self.columns)] = True
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_
