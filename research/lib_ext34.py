"""Shared helpers for agent ext34 (research/r320-r339): E3 leakage and E4 feature-stability follow-ups.

FoldColumnsModel fits one of three model families on a column subset. The subset is either
fixed (chosen without labels, e.g. the label-free high-variance block) or looked up per
training part from a precomputed map {train_key -> columns}; the map is built from the
training part of each official-CV fold only, so the evaluation stays honest.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ssc.transductive import TransductiveGMM

MODEL_FAMILIES = ("logreg", "qda", "gmm")


def train_key(X, y) -> str:
    """Hash of one training part (rows and labels), used to look up its in-fold column set."""
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    y = np.ascontiguousarray(np.asarray(y).astype(np.int8))
    return hashlib.sha1(X.tobytes() + y.tobytes()).hexdigest()[:20]


def base_model(family: str, source: str = "raw"):
    if family == "logreg":  # same base model as E3/E4 (L2, C = 0.1, liblinear, seed 0)
        return Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(C=0.1, solver="liblinear", max_iter=5000, random_state=0))])
    if family == "qda":  # same as R003 sup_qda_hv (reg 0.1), but scaled inside the fold
        return Pipeline([("scale", StandardScaler()), ("model", QuadraticDiscriminantAnalysis(reg_param=0.1))])
    raise ValueError(f"unknown family {family!r}")


class FoldColumnsModel(ClassifierMixin, BaseEstimator):
    """Model family on a column subset: fixed `columns`, or `column_map[train_key(X, y)]`."""

    def __init__(self, family: str = "logreg", columns=None, column_map=None, source: str = "raw"):
        self.family = family
        self.columns = columns
        self.column_map = column_map
        self.source = source

    def _cols(self, X, y) -> np.ndarray:
        if self.columns is not None:
            return np.array(sorted(int(c) for c in self.columns))
        key = train_key(X, y)
        if not self.column_map or key not in self.column_map:
            raise KeyError("training part not found in column_map (splits differ from the precomputed ones?)")
        return np.array(sorted(int(c) for c in self.column_map[key]))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.cols_ = self._cols(X, y)
        self.n_features_in_ = X.shape[1]
        if self.family == "gmm":  # label-free mixture on train+test features of these columns
            self.model_ = TransductiveGMM(columns=[int(c) for c in self.cols_], n_components=2,
                                          source=self.source).fit(X, y)
        else:
            self.model_ = clone(base_model(self.family, self.source)).fit(X[:, self.cols_], y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self.family == "gmm":
            return self.model_.predict_proba(X)
        return self.model_.predict_proba(X[:, self.cols_])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
