"""Shared helpers for the supervised-only / semi-supervised track (label `sup`, scripts r200-r249).

Question of the track: what can the 250 labels alone achieve, and where do the standard
semi-supervised baselines of scikit-learn sit between that and the TransductiveGMM?

Design
------
* Every candidate is evaluated with the OFFICIAL CV (project.yaml official_cv, 5-fold x 20,
  seed 2026) through ssc.validation.cross_validate and archived as one experiment Run with
  test predictions (fold_mean of the 100 fold models).
* Whenever a hyperparameter is tuned, the tuning is NESTED: `Tuned` runs GridSearchCV
  (inner RepeatedStratifiedKFold on the 200-row training fold only, scoring ROC AUC) and
  refits the winner on that training fold. The outer 50-row validation fold never takes part
  in the choice, so the outer CV stays an honest estimate. Each run records the grid size
  (n_configs) so the report can state how many configurations were tried.
* "Strictly supervised" pipelines use the 250 labelled rows only: the high-variance block is
  re-derived INSIDE each training fold from the 200 training rows (`HighVarColumns`, the same
  label-free rule as R001: std > 1.5 x median std), scaling is fitted in-fold, and no test
  feature is touched (uses_unlabeled_test_features = False).
* Semi-supervised wrappers (`PoolSelfTraining`, `PoolGraphSSL`) see all 20,000 feature rows
  (train + test). Rows of the current training fold carry their labels, every other row is
  passed as -1 (unlabelled), so validation-fold labels are never used. The block and its
  standardisation come from train+test features (label-free, as in ssc.transductive).
* `Tuned` writes one small JSON per outer fit (best params, inner AUC, selected columns,
  extra fit info) into a temporary log dir; the parent maps it back to (repeat, fold) by a
  hash of the training matrix and saves it as the run table fold_fit_info.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.utils.validation import check_is_fitted, validate_data

from ssc import data, paths
from ssc.features import abs_correlation, top_k_indices
from ssc.tracking import Run
from ssc.transductive import resolve_columns, unlabeled_features
from ssc.validation import Scheme, cross_validate, make_splits

AUTHOR_TAG = "sup"
CURRENT = {"source": "raw"}  # set by run_candidates before candidates are built (main process)


# --------------------------------------------------------------------------- setup
def setup(source: str = "raw"):
    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
    scheme = Scheme.from_dict(official)
    ds = data.load(source)
    splits = make_splits(ds.y, scheme)
    return scheme, ds, splits


def fold_key(X) -> str:
    return hashlib.sha1(np.ascontiguousarray(X, dtype=float).tobytes()).hexdigest()[:16]


def quiet():
    """Context manager that silences the warnings that flood logs with 100 x inner-CV fits
    (QDA 'Variables are collinear' on the rank-12 block, saga/liblinear convergence)."""
    cm = warnings.catch_warnings()
    cm.__enter__()
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", RuntimeWarning)
    return cm


# --------------------------------------------------------------------------- in-fold transformers
class HighVarColumns(SelectorMixin, BaseEstimator):
    """Label-free block rule of R001 applied to the rows passed to fit (the training fold):
    keep columns with std > ratio * median(std). Optionally add the `extra_k` non-block
    columns with the largest in-fold |corr| with the target (supervised, in-fold).
    Must come BEFORE any scaler (it needs the raw spreads)."""

    def __init__(self, ratio: float = 1.5, extra_k: int = 0):
        self.ratio = ratio
        self.extra_k = extra_k

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=float)
        std = X.std(0)
        block = std > self.ratio * np.median(std)
        if block.sum() < 2:  # no block (e.g. the synthetic sandbox data): keep the 2 widest columns
            block = np.zeros_like(block)
            block[np.argsort(-std)[:2]] = True
        mask = block.copy()
        if self.extra_k and y is not None:
            score = abs_correlation(X, y)
            score[block] = -1.0
            mask[top_k_indices(score, int(self.extra_k))] = True
        self.block_ = np.where(block)[0]
        self.support_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


class RankPCA(TransformerMixin, BaseEstimator):
    """PCA keeping every component with eigenvalue > tol * largest (drops exact linear
    redundancies, e.g. the 13-feature block has rank 12). Fitted in-fold, label-free."""

    def __init__(self, tol: float = 1e-8):
        self.tol = tol

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=float)
        self.mean_ = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        keep = s**2 > self.tol * s[0] ** 2
        self.components_ = Vt[keep]
        self.n_components_ = int(keep.sum())
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = validate_data(self, X, dtype=float, reset=False)
        return (X - self.mean_) @ self.components_.T


class GPCAuto(ClassifierMixin, BaseEstimator):
    """GaussianProcessClassifier (Laplace) with C * RBF kernel built at fit time, so an ARD
    length-scale vector matches however many columns arrive. Kernel hyperparameters are
    fitted by maximising the marginal likelihood on the training fold (type-II ML, no CV)."""

    def __init__(self, ard: bool = False, length_scale: float = 3.0, n_restarts_optimizer: int = 2,
                 random_state: int = 0):
        self.ard = ard
        self.length_scale = length_scale
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        ls = np.full(d, self.length_scale) if self.ard else self.length_scale
        kernel = ConstantKernel(1.0, (1e-3, 1e4)) * RBF(ls, (1e-2, 1e4))
        self.gpc_ = GaussianProcessClassifier(kernel, n_restarts_optimizer=self.n_restarts_optimizer,
                                              random_state=self.random_state, max_iter_predict=200).fit(X, y)
        self.classes_ = self.gpc_.classes_
        k = self.gpc_.kernel_
        self.fit_info_ = {"kernel": str(k), "log_marginal_likelihood": float(self.gpc_.log_marginal_likelihood_value_)}
        return self

    def predict_proba(self, X):
        return self.gpc_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.gpc_.predict(np.asarray(X, dtype=float))


# --------------------------------------------------------------------------- nested tuning wrapper
def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, BaseEstimator):  # repr shows the non-default parameters, e.g. HighVarColumns(ratio=1.6)
        return " ".join(repr(v).split())
    return str(v)


def _support_of(model):
    """Selected original columns of a fitted pipeline (first selector step), else None."""
    steps = getattr(model, "steps", None)
    if not steps:
        return None
    for _, st in steps:
        if hasattr(st, "get_support") and hasattr(st, "support_"):
            return [int(i) for i in np.where(st.get_support())[0]]
    return None


def _fit_info_of(model):
    last = model.steps[-1][1] if getattr(model, "steps", None) else model
    info = getattr(last, "fit_info_", None)
    if info is None and hasattr(last, "estimator_"):  # SubspaceModel
        info = getattr(last.estimator_, "fit_info_", None)
    return info


class Tuned(ClassifierMixin, BaseEstimator):
    """`estimator`, tuned by an inner CV on the training data only (nested CV).

    grid=None -> plain fit (still logs the selected columns / fit info).
    grid      -> GridSearchCV(scoring=scoring (default 'roc_auc'), cv=RepeatedStratifiedKFold(inner_splits,
                 inner_repeats, inner_seed), refit=True). Ties go to the FIRST grid entry, so
                 grids are written from the most to the least regularised setting.
    """

    def __init__(self, estimator=None, grid=None, inner_splits: int = 5, inner_repeats: int = 2,
                 inner_seed: int = 0, log_dir: str | None = None, n_jobs: int = 1, scoring: str = "roc_auc"):
        self.scoring = scoring
        self.estimator = estimator
        self.grid = grid
        self.inner_splits = inner_splits
        self.inner_repeats = inner_repeats
        self.inner_seed = inner_seed
        self.log_dir = log_dir
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        t0 = time.perf_counter()
        info = {"key": fold_key(X)}
        cm = quiet()
        try:
            if self.grid:
                cv = RepeatedStratifiedKFold(n_splits=self.inner_splits, n_repeats=self.inner_repeats,
                                             random_state=self.inner_seed)
                gs = GridSearchCV(clone(self.estimator), self.grid, scoring=self.scoring, cv=cv, refit=True,
                                  n_jobs=self.n_jobs)
                gs.fit(X, y)
                self.model_ = gs.best_estimator_
                self.best_params_ = gs.best_params_
                info.update(best_params=_jsonable(gs.best_params_), best_inner_auc=float(gs.best_score_),
                            n_candidates=int(len(gs.cv_results_["params"])))
            else:
                self.model_ = clone(self.estimator).fit(X, y)
        finally:
            cm.__exit__(None, None, None)
        info.update(support=_support_of(self.model_), fit_info=_jsonable(_fit_info_of(self.model_)),
                    fit_seconds=time.perf_counter() - t0)
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.log_dir) / f"{info['key']}_{uuid.uuid4().hex[:8]}.json").write_text(
                json.dumps(info), encoding="utf-8")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        cm = quiet()
        try:
            return self.model_.predict_proba(np.asarray(X, dtype=float))
        finally:
            cm.__exit__(None, None, None)

    def decision_function(self, X):
        cm = quiet()
        try:
            return self.model_.decision_function(np.asarray(X, dtype=float))
        finally:
            cm.__exit__(None, None, None)

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def collect_logs(log_dir: str, X, splits) -> pd.DataFrame | None:
    files = list(Path(log_dir).glob("*.json"))
    if not files:
        return None
    Xa = np.asarray(X, dtype=float)
    where = {fold_key(Xa[s.train_idx]): (s.repeat, s.fold) for s in splits}
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        r, k = where.get(d["key"], (-1, -1))
        rows.append({"repeat": r, "fold": k,
                     "best_params": json.dumps(d.get("best_params"), sort_keys=True) if d.get("best_params") is not None else "",
                     "best_inner_auc": d.get("best_inner_auc"), "n_candidates": d.get("n_candidates"),
                     "support": json.dumps(d.get("support")) if d.get("support") is not None else "",
                     "fit_info": json.dumps(d.get("fit_info")) if d.get("fit_info") is not None else "",
                     "fit_seconds": d.get("fit_seconds")})
    return pd.DataFrame(rows).sort_values(["repeat", "fold"]).reset_index(drop=True)


# --------------------------------------------------------------------------- unlabeled pool (train+test)
@lru_cache(maxsize=2)
def _pool_index(source: str) -> dict:
    X_all = np.ascontiguousarray(unlabeled_features(source), dtype=float)
    return {X_all[i].tobytes(): i for i in range(len(X_all))}


def pool_rows(X, source: str = "raw") -> np.ndarray:
    """Index of each row of X in the train+test pool (train rows first, then test), -1 if absent."""
    X = np.ascontiguousarray(X, dtype=float)
    d = _pool_index(source)
    return np.array([d.get(X[i].tobytes(), -1) for i in range(len(X))], dtype=int)


@lru_cache(maxsize=8)
def pool_space(source: str, columns) -> tuple:
    """(cols, mu, sd, Z): label-free column set and train+test standardisation of the pool."""
    X_all = unlabeled_features(source)
    cols = resolve_columns(columns, X_all) if not isinstance(columns, tuple) else np.array(columns)
    B = X_all[:, cols]
    mu, sd = B.mean(0), B.std(0)
    return cols, mu, sd, (B - mu) / sd


_GRAPH_CACHE: dict = {}


class _CachedKnnGraph:
    """Exact optimisation of sklearn's knn kernel: the kNN connectivity graph of the
    (label-free, fixed) 20,000-row pool is built once per process and reused by every fold."""

    def _get_kernel(self, X, y=None):
        if self.kernel == "knn" and y is None:
            key = (hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest(), int(self.n_neighbors))
            if key not in _GRAPH_CACHE:
                nn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs).fit(X)
                _GRAPH_CACHE[key] = (nn, nn.kneighbors_graph(nn._fit_X, self.n_neighbors, mode="connectivity"))
            self.nn_fit, G = _GRAPH_CACHE[key]
            return G.copy()  # LabelPropagation normalises the matrix in place
        return super()._get_kernel(X, y)


class CachedLabelSpreading(_CachedKnnGraph, LabelSpreading):
    pass


class CachedLabelPropagation(_CachedKnnGraph, LabelPropagation):
    pass


class PoolGraphSSL(ClassifierMixin, BaseEstimator):
    """LabelSpreading / LabelPropagation (scikit-learn, knn kernel) on ALL 20,000 rows:
    training-fold rows labelled, every other row -1. Scores of pool rows (validation and
    test rows are in the pool) are their transductive label distributions."""

    def __init__(self, method: str = "spreading", n_neighbors: int = 7, alpha: float = 0.2,
                 max_iter: int = 1000, columns="highvar", source: str = "raw"):
        self.method = method
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.columns = columns
        self.source = source

    def fit(self, X, y):
        cols, mu, sd, Z = pool_space(self.source, self.columns if isinstance(self.columns, str) else tuple(self.columns))
        idx = pool_rows(X, self.source)
        if (idx < 0).any():
            raise ValueError("training rows not found in the train+test pool")
        y_pool = np.full(len(Z), -1, dtype=int)
        y_pool[idx] = np.asarray(y).astype(int)
        if self.method == "spreading":
            m = CachedLabelSpreading(kernel="knn", n_neighbors=self.n_neighbors, alpha=self.alpha, max_iter=self.max_iter)
        elif self.method == "propagation":
            m = CachedLabelPropagation(kernel="knn", n_neighbors=self.n_neighbors, max_iter=self.max_iter)
        else:
            raise ValueError(self.method)
        cm = quiet()
        try:
            m.fit(Z, y_pool)
        finally:
            cm.__exit__(None, None, None)
        j = list(m.classes_).index(1)
        self.dist_ = m.label_distributions_[:, j].copy()
        self.model_, self.cols_, self.mu_, self.sd_ = m, cols, mu, sd
        self.classes_ = np.array([0, 1])
        unl = y_pool < 0
        self.fit_info_ = {"n_iter": int(getattr(m, "n_iter_", -1)),
                          "frac_unlabelled_zero_mass": float((m.label_distributions_[unl].sum(1) == 0).mean())}
        return self

    def predict_proba(self, X):
        idx = pool_rows(X, self.source)
        p = np.empty(len(idx))
        inp = idx >= 0
        p[inp] = self.dist_[idx[inp]]
        if (~inp).any():
            Zx = (np.asarray(X, dtype=float)[~inp][:, self.cols_] - self.mu_) / self.sd_
            p[~inp] = self.model_.predict_proba(Zx)[:, list(self.model_.classes_).index(1)]
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PoolSelfTraining(ClassifierMixin, BaseEstimator):
    """sklearn SelfTrainingClassifier on ALL 20,000 rows (training-fold rows labelled, the
    rest -1), block standardised with train+test statistics. Predictions are inductive (the
    final base classifier applied to the rows)."""

    def __init__(self, estimator=None, threshold: float = 0.75, criterion: str = "threshold",
                 k_best: int = 10, max_iter: int = 10, columns="highvar", source: str = "raw"):
        self.estimator = estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.columns = columns
        self.source = source

    def fit(self, X, y):
        cols, mu, sd, Z = pool_space(self.source, self.columns if isinstance(self.columns, str) else tuple(self.columns))
        idx = pool_rows(X, self.source)
        if (idx < 0).any():
            raise ValueError("training rows not found in the train+test pool")
        y_pool = np.full(len(Z), -1, dtype=int)
        y_pool[idx] = np.asarray(y).astype(int)
        cm = quiet()
        try:
            self.st_ = SelfTrainingClassifier(clone(self.estimator), threshold=self.threshold, criterion=self.criterion,
                                              k_best=self.k_best, max_iter=self.max_iter).fit(Z, y_pool)
        finally:
            cm.__exit__(None, None, None)
        self.cols_, self.mu_, self.sd_ = cols, mu, sd
        self.classes_ = np.array([0, 1])
        li = self.st_.labeled_iter_
        self.fit_info_ = {"n_pseudo_labelled": int((li > 0).sum()), "n_iter": int(self.st_.n_iter_),
                          "termination": str(self.st_.termination_condition_)}
        return self

    def _z(self, X):
        return (np.asarray(X, dtype=float)[:, self.cols_] - self.mu_) / self.sd_

    def predict_proba(self, X):
        cm = quiet()
        try:
            return self.st_.predict_proba(self._z(X))
        finally:
            cm.__exit__(None, None, None)

    def predict(self, X):
        return self.st_.predict(self._z(X))


# --------------------------------------------------------------------------- candidates + runner
@dataclass
class Cand:
    name: str                       # run name suffix: run name = f"{prefix}_{name}"
    make: Callable                  # make(log_dir) -> unfitted estimator
    group: str                      # supervised-linear | supervised-generative | supervised-kernel | supervised-trees | semi-supervised | selection
    features: str                   # all | block(in-fold) | block(train+test) | corr-topk(in-fold) | ...
    uses_unlabeled: bool            # touches test-set features?
    n_configs: int                  # grid size of the nested tuning (1 = no tuning)
    desc: str
    hypothesis: str
    tuning: str = "none"            # description of the nested tuning
    extra: dict = field(default_factory=dict)


def prob_metrics(oof: np.ndarray, y: np.ndarray) -> dict:
    """Pooled out-of-fold log-loss / Brier (mean over repeats), only for scores in [0, 1]."""
    if not np.isfinite(oof).all() or oof.min() < 0 or oof.max() > 1:
        return {"oof_logloss": None, "oof_brier": None}
    eps = 1e-15
    ll, br = [], []
    for row in oof:
        p = np.clip(row, eps, 1 - eps)
        ll.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
        br.append(float(np.mean((row - y) ** 2)))
    return {"oof_logloss": float(np.mean(ll)), "oof_brier": float(np.mean(br))}


def summarise_fold_info(fi: pd.DataFrame | None, tt_block: list[int]) -> dict:
    if fi is None or fi.empty:
        return {}
    out = {"n_fold_logs": int(len(fi))}
    if (fi["best_params"] != "").any():
        vc = fi["best_params"].value_counts()
        out["chosen_params_top"] = {k: int(v) for k, v in vc.head(6).items()}
        out["chosen_params_n_distinct"] = int(len(vc))
        out["best_inner_auc_mean"] = float(pd.to_numeric(fi["best_inner_auc"], errors="coerce").mean())
    if (fi["support"] != "").any():
        sup = fi["support"].map(lambda s: tuple(json.loads(s)) if s else None)
        out["support_size_mean"] = float(sup.map(lambda t: len(t) if t else np.nan).mean())
        out["support_equals_tt_block_frac"] = float(sup.map(lambda t: t == tuple(tt_block)).mean())
        out["support_contains_tt_block_frac"] = float(sup.map(lambda t: bool(t) and set(tt_block) <= set(t)).mean())
    return out


def run_candidates(cands: list[Cand], prefix: str, source: str, n_jobs: int, author: str | None,
                   script: str, only: list[str] | None = None, max_folds: int | None = None) -> pd.DataFrame:
    scheme, ds, splits = setup(source)
    CURRENT["source"] = source
    official = True
    if max_folds:  # smoke tests only (synthetic data, scratch archive): not the official protocol
        if source == "raw":
            raise SystemExit("--max-folds is for smoke tests on --data synthetic only")
        splits, official = splits[:max_folds], False
    tt_block = [int(c) for c in resolve_columns("highvar", unlabeled_features(source))]
    print(f"[sup] {script}: {len(cands)} candidates, n_jobs={n_jobs}, cpus={os.cpu_count()}, "
          f"train+test block = {tt_block}", flush=True)
    rows = []
    for c in cands:
        if only and c.name not in only:
            continue
        name = f"{prefix}_{c.name}"
        log_dir = tempfile.mkdtemp(prefix=f"suplog_{c.name}_")
        cfg = {"name": name, "description": c.desc, "model_code": f"{script} + research/lib_sup.py",
               "group": c.group, "features": c.features, "tuning": c.tuning, "n_configs": c.n_configs,
               "inner_cv": "RepeatedStratifiedKFold on the 200-row training fold (see lib_sup.Tuned)" if c.n_configs > 1 else None,
               "cv": scheme.to_dict(), "test_pred": "fold_mean",
               "uses_unlabeled_test_features": c.uses_unlabeled, **c.extra}
        t0 = time.time()
        try:
            with Run(name, cfg, kind="experiment", source=source, author=author, hypothesis=c.hypothesis) as run:
                model = c.make(log_dir)
                cv = cross_validate(model, ds.X, ds.y, splits, scheme, X_test=ds.X_test, n_jobs=n_jobs)
                s = run.save_cv(cv)
                fi = collect_logs(log_dir, ds.X, splits)
                if fi is not None:
                    run.save_table("fold_fit_info", fi)
                extra = {"official_cv": official, "group": c.group, "features": c.features, "n_configs": c.n_configs,
                         "tuning": c.tuning, "uses_unlabeled_test_features": c.uses_unlabeled,
                         "wall_seconds": round(time.time() - t0, 1), "n_constant_pred_folds": int(cv.fold_scores["constant_pred"].sum()),
                         **prob_metrics(cv.oof, ds.y), **summarise_fold_info(fi, tt_block)}
                s = run.finish(extra_summary=extra)
                rows.append({"candidate": name, "run_id": run.run_id, "group": c.group, "features": c.features,
                             "n_configs": c.n_configs, "uses_unlabeled_test_features": c.uses_unlabeled,
                             "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
                             "oof_auc_mean": s.get("oof_auc_mean"), "oof_logloss": s.get("oof_logloss"),
                             "chosen": json.dumps(s.get("chosen_params_top", {}))[:200], "wall_s": extra["wall_seconds"],
                             "description": c.desc})
                print(f"[sup] {name:34s} auc={s['auc_mean']:.4f} +/- {s['auc_std_repeat']:.4f}  "
                      f"logloss={s.get('oof_logloss')}  {extra['wall_seconds']:.0f}s  run={run.run_id}  "
                      f"chosen={json.dumps(s.get('chosen_params_top', {}))[:160]}", flush=True)
        except Exception as e:  # noqa: BLE001 - keep going with the next candidate; the run is archived as failed
            print(f"[sup] {name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            rows.append({"candidate": name, "run_id": "FAILED", "group": c.group, "description": c.desc})
        finally:
            shutil.rmtree(log_dir, ignore_errors=True)
    res = pd.DataFrame(rows)
    tables = paths.tables_dir(source)
    tables.mkdir(parents=True, exist_ok=True)
    out = tables / f"{prefix}_candidates.csv"
    if only and out.exists():  # partial rerun: keep the other rows
        old = pd.read_csv(out)
        res = pd.concat([old[~old["candidate"].isin(res["candidate"])], res], ignore_index=True)
    res.to_csv(out, index=False)
    if not res.empty and "auc_mean" in res:
        print("\n" + res.sort_values("auc_mean", ascending=False)[["candidate", "auc_mean", "auc_std_repeat", "oof_logloss", "n_configs", "run_id"]].to_string(index=False))
    return res


def common_args(desc: str):
    import argparse
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", default="raw", choices=paths.SOURCES)
    p.add_argument("--author", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--only", nargs="*", default=None, help="subset of candidate names")
    p.add_argument("--max-folds", type=int, default=None, help="smoke test only (synthetic): first N folds")
    return p


# --------------------------------------------------------------------------- figures
LB_PANELS = [
    ("Supervised only: labels of n rows, no unlabeled rows in the model",
     ["sup_lr_tt", "sup_qda_infold_lw", "sup_qda_tt_lw", "sup_qda_tt_r01", "sup_qda_tt_r003", "sup_qda_infold_r003"]),
    ("Using the 20,000 unlabeled rows (labels of n rows)",
     ["ssl_labelspread", "ssl_selftrain_r01", "ssl_selftrain_lw", "trans_gmm2"]),
]
LB_LABELS = {"sup_qda_infold_lw": "QDA-LW, block found in-fold (strict)",
             "sup_qda_infold_r003": "QDA reg 0.003, block found in-fold (strict)",
             "sup_qda_tt_lw": "QDA-LW, train+test block", "sup_qda_tt_r01": "QDA reg 0.1, train+test block",
             "sup_qda_tt_r003": "QDA reg 0.003, train+test block", "sup_lr_tt": "L2 logistic, train+test block",
             "ssl_selftrain_lw": "Self-training QDA-LW", "ssl_selftrain_r01": "Self-training QDA reg 0.1",
             "ssl_labelspread": "Label spreading (kNN graph)", "trans_gmm2": "Transductive GMM K=2"}
# dataviz reference palette (light), categorical slots in fixed order; GMM = primary ink
LB_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]


def plot_label_budget(summ: pd.DataFrame, path, title: str):
    """Two panels of mean validation AUC vs number of labels. Only budgets where every evaluation of a
    model is defined are drawn (svd-QDA needs more rows per class than columns)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = sorted(summ["n_labels"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, (ptitle, ms) in zip(axes, LB_PANELS):
        for i, m in enumerate(ms):
            d = summ[(summ["model"] == m) & (summ["n_failed"] == 0)].sort_values("n_labels")
            if d.empty:
                continue
            c = "#0b0b0b" if m == "trans_gmm2" else LB_SLOTS[i % len(LB_SLOTS)]
            strict = m.startswith("sup_qda_infold_r003")
            ax.plot(d["n_labels"], d["auc_mean"], "--" if strict else "-", color=c, lw=2.6 if strict else 2,
                    marker="o", ms=5, zorder=5 if strict or m == "trans_gmm2" else 3,
                    label=LB_LABELS.get(m, m) + (f" (defined from n={int(d['n_labels'].min())})" if d["n_labels"].min() > budgets[0] and m.startswith("sup_qda") and "lw" not in m else ""))
            ax.fill_between(d["n_labels"], d["auc_mean"] - 2 * d["auc_se"], (d["auc_mean"] + 2 * d["auc_se"]).clip(upper=1.0),
                            color=c, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) if b != 160 else "" for b in budgets])
        ax.minorticks_off()
        ax.set_ylim(0.55, 1.01)
        ax.set_xlabel("labelled training rows n (log scale)")
        ax.set_title(ptitle, fontsize=9.5)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    axes[0].set_ylabel("validation-fold ROC AUC (mean; band = 2 SE)")
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig
