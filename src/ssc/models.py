"""Named model factory: a config dict -> an unfitted sklearn Pipeline.

Pipeline spec (the `pipeline:` block of an experiment config):

    scaler: standard            # standard | robust | none
    selector:                   # optional, re-fitted inside every fold
      name: corr_topk           # corr_topk | stability_topk | fixed_columns
      k: 20
    model:
      name: logreg              # see ESTIMATORS below
      params: {C: 0.1, l1_ratio: 1.0, solver: liblinear}
    random_state: 0             # model seed; held fixed across folds and repeats

Transductive models (TRANSDUCTIVE, from ssc.transductive) read the unlabeled train+test
FEATURES of the run's data source themselves and take the raw 150 columns, so their
pipeline is `scaler: none` with no selector. The final model S003 as a config
(configs/final_s003_gmm2_hv.yaml):

    scaler: none
    model:
      name: transductive_gmm    # TransductiveGMM: 2 full-covariance Gaussians on the high-variance block
      params: {columns: highvar, n_components: 2, covariance_type: full, n_init: 5, alpha: 1.0}

Scores for AUC come from predict_score(): predict_proba[:, 1] if available,
else decision_function, else predict.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import paths
from .features import CorrelationTopK, FixedColumns, StabilityTopK


def _transductive_gmm(params, seed, source=None):
    """ssc.transductive.TransductiveGMM (submission S003). Its mixture is fitted on the train+test
    features of a data source, which must be the run's own: `source` (the run's data source, passed
    by build_pipeline) wins over $SSC_DATA, and a params.source that disagrees with it is an error."""
    from .transductive import TransductiveGMM
    p = dict(params)
    if source is not None and p.get("source", source) != source:
        raise ValueError(f"transductive_gmm: params.source {p['source']!r} differs from the run's data "
                         f"source {source!r} (the mixture must use the features of the data being scored)")
    p["source"] = paths.resolve_source(p.get("source") or source)
    return TransductiveGMM(**p, random_state=seed)


def _lgbm(params, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**{"verbose": -1, "n_jobs": 1, **params}, random_state=seed)


def _xgb(params, seed):
    from xgboost import XGBClassifier
    return XGBClassifier(**{"n_jobs": 1, "eval_metric": "auc", **params}, random_state=seed)


# name -> builder(params, seed). Defaults are deliberately conservative for n=250.
ESTIMATORS: dict[str, Callable[[dict, int], Any]] = {
    "logreg": lambda p, s: LogisticRegression(**{"solver": "liblinear", "max_iter": 5000, **p}, random_state=s),
    # l1_ratios=(0.0,) = L2, set explicitly: sklearn 1.9 warns that the default changes in 1.10
    "logreg_cv": lambda p, s: LogisticRegressionCV(**{"solver": "liblinear", "max_iter": 5000, "scoring": "roc_auc", "use_legacy_attributes": False, "l1_ratios": (0.0,), **p}, random_state=s),
    "lda": lambda p, s: LinearDiscriminantAnalysis(**{"solver": "lsqr", "shrinkage": "auto", **p}),
    "gnb": lambda p, s: GaussianNB(**p),
    "svc": lambda p, s: SVC(**{"kernel": "rbf", **p}, random_state=s),
    "linear_svc": lambda p, s: LinearSVC(**{"max_iter": 20000, **p}, random_state=s),
    "ridge_clf": lambda p, s: RidgeClassifier(**p),
    "lasso_reg": lambda p, s: Lasso(**{"alpha": 0.03, "max_iter": 20000, **p}, random_state=s),
    "knn": lambda p, s: KNeighborsClassifier(**{"n_neighbors": 25, **p}),
    "rf": lambda p, s: RandomForestClassifier(**{"n_estimators": 500, "n_jobs": 1, **p}, random_state=s),
    "extratrees": lambda p, s: ExtraTreesClassifier(**{"n_estimators": 500, "n_jobs": 1, **p}, random_state=s),
    "hgb": lambda p, s: HistGradientBoostingClassifier(**p, random_state=s),
    "lgbm": _lgbm,
    "xgb": _xgb,
    "transductive_gmm": _transductive_gmm,
}
# builder(params, seed, source): models that use the unlabeled train+test features of the data source;
# they need the raw feature columns (scaler: none, no selector), see build_pipeline
TRANSDUCTIVE = ("transductive_gmm",)

SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def build_selector(spec: dict | None):
    if not spec:
        return None
    spec = dict(spec)
    name = spec.pop("name")
    if name == "corr_topk":
        return CorrelationTopK(**spec)
    if name == "stability_topk":
        return StabilityTopK(**spec)
    if name == "fixed_columns":
        return FixedColumns(**spec)
    raise ValueError(f"unknown selector {name!r}")


def build_estimator(spec: dict, random_state: int = 0, source: str | None = None):
    """source: the run's data source; only TRANSDUCTIVE models use it (default: $SSC_DATA, else raw)."""
    name = spec["name"]
    if name not in ESTIMATORS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(ESTIMATORS)}")
    if name in TRANSDUCTIVE:
        return ESTIMATORS[name](dict(spec.get("params") or {}), random_state, source)
    return ESTIMATORS[name](dict(spec.get("params") or {}), random_state)


def build_pipeline(spec: dict, source: str | None = None) -> Pipeline:
    """source: the run's data source, passed on to TRANSDUCTIVE models (see build_estimator)."""
    name = (spec.get("model") or {}).get("name")
    if name in TRANSDUCTIVE and (spec.get("scaler", "standard") not in (None, "none") or spec.get("selector")):
        raise ValueError(f"{name} takes the raw feature columns (it standardises and picks columns itself "
                         "from train+test features): use scaler: none and no selector")
    steps = []
    scaler = spec.get("scaler", "standard")
    if scaler and scaler != "none":
        if scaler not in SCALERS:
            raise ValueError(f"unknown scaler {scaler!r}; available: {sorted(SCALERS)} or 'none'")
        steps.append(("scale", SCALERS[scaler]()))
    selector = build_selector(spec.get("selector"))
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", build_estimator(spec["model"], int(spec.get("random_state", 0)), source)))
    return Pipeline(steps)


def predict_score(model, X) -> np.ndarray:
    """Continuous score for ranking (AUC). Larger = more likely class 1."""
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))[:, 1]
        except AttributeError:  # e.g. SVC(probability=False) exposes the attr but raises
            pass
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X), dtype=float).ravel()


def describe(spec: dict) -> dict[str, str]:
    """Short strings for the registry columns."""
    sel = spec.get("selector") or {}
    model = spec["model"]
    params = ",".join(f"{k}={v}" for k, v in sorted((model.get("params") or {}).items()))
    sel_params = ",".join(f"{k}={v}" for k, v in sorted(sel.items()) if k != "name")
    return {
        "model": f"{model['name']}({params})",
        "selector": f"{sel['name']}({sel_params})" if sel else "none",
        "scaler": str(spec.get("scaler", "standard")),
    }
