"""Validation protocol: split generation, repeated CV, paired model comparison.

Conventions
-----------
* A scheme is (kind, n_splits, n_repeats, seed[, test_size]).
  kind="kfold":   repeat r uses StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).
  kind="holdout": repeat r uses one StratifiedShuffleSplit(test_size, random_state=seed + r).
  Note: sklearn's (Stratified)KFold does NOT shuffle unless shuffle=True; with
  shuffle=False every repeat is the same split, so repeats are meaningless.
* The split signature is a hash of every validation index set. Two runs are
  directly comparable (paired) only if their signatures match.
* The model's own seed (pipeline.random_state) is fixed separately, so the
  spread across repeats measures split-to-split variation only.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

from .models import predict_score

TEST_PRED_MODES = ("fold_mean", "refit", "none")


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class Scheme:
    kind: str = "kfold"          # "kfold" | "holdout"
    n_splits: int = 5            # kfold only
    n_repeats: int = 10
    seed: int = 2026
    test_size: float = 0.2       # holdout only
    shuffle: bool = True         # kfold only; False reproduces sklearn's default (no shuffling)
    stratify: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "Scheme":
        d = dict(d or {})
        if "scheme" in d:  # accept the config key name too
            d["kind"] = d.pop("scheme")
        return cls(**d)

    def to_dict(self) -> dict:
        return dict(kind=self.kind, n_splits=self.n_splits, n_repeats=self.n_repeats, seed=self.seed,
                    test_size=self.test_size, shuffle=self.shuffle, stratify=self.stratify)

    def label(self) -> str:
        if self.kind == "holdout":
            return f"holdout{int(round(self.test_size * 100))}x{self.n_repeats}"
        return f"{self.n_splits}foldx{self.n_repeats}" + ("" if self.shuffle else "_noshuffle")


def make_splits(y, scheme: Scheme) -> list[Split]:
    y = np.asarray(y)
    X_dummy = np.zeros((len(y), 1))
    splits: list[Split] = []
    for r in range(scheme.n_repeats):
        rs = scheme.seed + r
        if scheme.kind == "kfold":
            if scheme.shuffle:
                cv = (StratifiedKFold if scheme.stratify else KFold)(scheme.n_splits, shuffle=True, random_state=rs)
            else:
                cv = (StratifiedKFold if scheme.stratify else KFold)(scheme.n_splits, shuffle=False)
        elif scheme.kind == "holdout":
            cv = (StratifiedShuffleSplit if scheme.stratify else ShuffleSplit)(
                n_splits=1, test_size=scheme.test_size, random_state=rs)
        else:
            raise ValueError(f"unknown scheme kind {scheme.kind!r}")
        for f, (tr, va) in enumerate(cv.split(X_dummy, y)):
            splits.append(Split(r, f, np.asarray(tr), np.asarray(va)))
    return splits


def split_signature(splits: list[Split]) -> str:
    h = hashlib.sha1()
    for s in splits:
        h.update(f"{s.repeat}:{s.fold}:".encode())
        h.update(np.sort(s.val_idx).astype(np.int64).tobytes())
    return h.hexdigest()[:12]


def val_id(val_idx: np.ndarray) -> str:
    """Short hash of one validation index set ('v' prefix keeps read_csv from parsing it as a number)."""
    return "v" + hashlib.sha1(np.sort(np.asarray(val_idx)).astype(np.int64).tobytes()).hexdigest()[:10]


@dataclass
class CVResult:
    fold_scores: pd.DataFrame            # repeat, fold, auc, n_train, n_val, fit_seconds, val_id, constant_pred
    oof: np.ndarray                      # (n_repeats, n_samples); NaN where a row was not validated
    split_signature: str
    scheme: dict
    test_pred: np.ndarray | None = None  # mean of fold models' test scores (or refit, see cross_validate)
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        fs = self.fold_scores
        rep = fs.groupby("repeat")["auc"].mean()
        out = {
            "auc_mean": float(fs["auc"].mean()),
            "auc_std_fold": float(fs["auc"].std(ddof=1)) if len(fs) > 1 else float("nan"),
            "auc_std_repeat": float(rep.std(ddof=1)) if len(rep) > 1 else float("nan"),
            "auc_min_repeat": float(rep.min()),
            "auc_max_repeat": float(rep.max()),
            "n_folds_total": int(len(fs)),
            "split_signature": self.split_signature,
        }
        if self.scheme.get("kind") == "kfold":  # every row validated once per repeat -> pooled OOF AUC
            y = self.extra.get("y")
            if y is not None:
                pooled = [roc_auc_score(y, row) for row in self.oof]
                out["oof_auc_mean"] = float(np.mean(pooled))
                out["oof_auc_std"] = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else float("nan")
        return out


def _fit_one(estimator, X, y, X_test, split: Split):
    t0 = time.perf_counter()
    model = clone(estimator)
    model.fit(X[split.train_idx], y[split.train_idx])
    val = predict_score(model, X[split.val_idx])
    test = predict_score(model, X_test) if X_test is not None else None
    return split, val, test, time.perf_counter() - t0


def cross_validate(estimator, X, y, splits: list[Split], scheme: Scheme | dict | None = None,
                   X_test=None, test_pred: str = "fold_mean", n_jobs: int = 1) -> CVResult:
    """Fit a fresh clone of `estimator` on every split and score AUC on its validation part.

    test_pred: "fold_mean" averages the test scores of all fold models;
               "refit" fits once more on all rows and predicts the test set;
               "none" skips test prediction.

    fold_scores also records each fold's validation-set hash (val_id, used by
    paired_compare) and whether the fold model gave one constant score (constant_pred,
    e.g. L1 logreg with every coefficient at 0, which scores AUC 0.5).
    """
    if test_pred not in TEST_PRED_MODES:
        raise ValueError(f"test_pred must be one of {TEST_PRED_MODES}, got {test_pred!r}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    Xt = None if X_test is None or test_pred != "fold_mean" else np.asarray(X_test, dtype=float)
    scheme_d = scheme.to_dict() if isinstance(scheme, Scheme) else dict(scheme or {})

    results = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(estimator, X, y, Xt, s) for s in splits)

    n_repeats = max(s.repeat for s in splits) + 1
    oof = np.full((n_repeats, len(y)), np.nan)
    rows, test_sum, n_test_models = [], None, 0
    for split, val, test, secs in results:
        oof[split.repeat, split.val_idx] = val
        rows.append({"repeat": split.repeat, "fold": split.fold,
                     "auc": roc_auc_score(y[split.val_idx], val),
                     "n_train": len(split.train_idx), "n_val": len(split.val_idx), "fit_seconds": secs,
                     "val_id": val_id(split.val_idx), "constant_pred": bool(np.ptp(val) < 1e-12)})
        if test is not None:
            test_sum = test if test_sum is None else test_sum + test
            n_test_models += 1

    tp = None
    if test_pred == "fold_mean" and test_sum is not None:
        tp = test_sum / n_test_models
    elif test_pred == "refit" and X_test is not None:
        full = clone(estimator).fit(X, y)
        tp = predict_score(full, np.asarray(X_test, dtype=float))

    # the unfitted template travels in extra so tracking.Run.save_cv can record its get_params(deep=True)
    return CVResult(pd.DataFrame(rows), oof, split_signature(splits), scheme_d, tp, {"y": y, "estimator": estimator})


def run_cv(estimator, X, y, scheme: Scheme, **kwargs) -> CVResult:
    return cross_validate(estimator, X, y, make_splits(y, scheme), scheme, **kwargs)


def paired_compare(fold_scores_a: pd.DataFrame, fold_scores_b: pd.DataFrame,
                   n_train: int | None = None, n_val: int | None = None) -> dict:
    """Compare two models evaluated on IDENTICAL splits (same split signature).

    Returns the mean AUC difference (a - b); its spread across folds and across
    repeats; se_split = std_diff_repeat / sqrt(n_repeats), the split noise of the
    reported mean difference (how far it would move under other split seeds of the
    same rows); the fraction of folds where a wins (ties count 1/2); and the
    Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio 2003, Machine
    Learning 52:239-281), which inflates the variance by n_val/n_train to account
    for overlapping training sets (a data-sampling view, far wider than se_split).
    Identical fold differences give t = 0, p = 1 (all zero) or |t| = inf, p = 0.

    Frames written by cross_validate carry each fold's validation-set hash (val_id):
    when both have it, the folds must match set by set, otherwise ValueError. Older
    frames without val_id are only aligned by their (repeat, fold) numbers.
    """
    m = fold_scores_a.merge(fold_scores_b, on=["repeat", "fold"], suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(fold_scores_a) or len(m) != len(fold_scores_b):
        raise ValueError("fold sets differ: the two runs were not evaluated on the same splits")
    if {"val_id_a", "val_id_b"} <= set(m.columns) and (m["val_id_a"] != m["val_id_b"]).any():
        n_bad = int((m["val_id_a"] != m["val_id_b"]).sum())
        raise ValueError(f"validation sets differ in {n_bad} of {len(m)} folds: the two runs were not "
                         "evaluated on the same splits (other split seed or data)")
    d = (m["auc_a"] - m["auc_b"]).to_numpy()
    rep = (m.assign(d=d).groupby("repeat")["d"].mean()).to_numpy()
    n_tr = n_train or int(m["n_train_a"].iloc[0])
    n_va = n_val or int(m["n_val_a"].iloc[0])
    J = len(d)
    var = d.var(ddof=1) if J > 1 else float("nan")
    if J > 1 and var == 0:  # every fold difference identical: no spread to test against
        t = 0.0 if d.mean() == 0 else float(np.copysign(np.inf, d.mean()))
        p = 1.0 if d.mean() == 0 else 0.0
    else:
        se = np.sqrt((1.0 / J + n_va / n_tr) * var) if J > 1 else float("nan")
        t = d.mean() / se if se and se > 0 else float("nan")
        p = 2 * stats.t.sf(abs(t), df=J - 1) if np.isfinite(t) else float("nan")
    std_rep = float(rep.std(ddof=1)) if len(rep) > 1 else float("nan")
    return {
        "mean_diff": float(d.mean()),
        "std_diff_fold": float(d.std(ddof=1)) if J > 1 else float("nan"),
        "std_diff_repeat": std_rep,
        "se_split": std_rep / np.sqrt(len(rep)) if len(rep) > 1 else float("nan"),
        "win_rate_fold": float((d > 0).mean() + 0.5 * (d == 0).mean()),
        "n_folds": J,
        "n_repeats": int(len(rep)),
        "corrected_t": float(t),
        "corrected_p": float(p),
    }


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC estimate.

    Use it to judge leaderboard noise: the public LB has about 395 rows."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(var))
