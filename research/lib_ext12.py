"""Shared helpers of agent ext12 (extensions E1 validation protocol, E2 learning curve).

`build_pipeline(spec)` extends ssc.models.build_pipeline with the label-free-subspace and
transductive models of ssc.transductive, so the unmodified extension scripts
(extensions/e1_validation_protocol/run.py, extensions/e2_learning_curve/run.py) can study
them through research/r301_run_extension.py (which swaps this builder into the loaded module).

Extra model names (pipeline spec `model.name`), all require `scaler: none` and no selector
because the models choose their columns and standardise internally with train+test
FEATURE statistics (label-free, decision D007):

    subspace_qda     SubspaceModel(QuadraticDiscriminantAnalysis(reg_param), columns)
                     params: reg_param (default 0.1), columns (default "highvar"), pca_dims
    subspace_lda     SubspaceModel(LinearDiscriminantAnalysis(lsqr, shrinkage=auto), columns)
    subspace_logreg  SubspaceModel(LogisticRegression(C, l1_ratio, liblinear), columns)
    transductive_gmm TransductiveGMM(**params)  (GMM on train+test features; labels only map
                     mixture components to classes)

`source` (raw | synthetic) of the unlabeled feature matrix defaults to $SSC_DATA (set by
r301 from --data) and else "raw"; an explicit `source` param wins.
Every other spec is passed to ssc.models.build_pipeline unchanged.
"""

from __future__ import annotations

import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ssc.models import build_pipeline as _ssc_build_pipeline
from ssc.transductive import SubspaceModel, TransductiveGMM

# the 13 label-free high-variance columns found by R001 (std > 1.5 x median std on train+test)
BLOCK = [8, 15, 26, 30, 46, 49, 64, 67, 77, 80, 98, 100, 112]

EXTRA_MODELS = ("subspace_qda", "subspace_lda", "subspace_logreg", "transductive_gmm")


def _source(params: dict) -> str:
    return params.pop("source", None) or os.environ.get("SSC_DATA") or "raw"


def build_estimator(name: str, params: dict | None, random_state: int = 0):
    p = dict(params or {})
    src = _source(p)
    if name == "transductive_gmm":
        if src == "synthetic" and p.get("columns", "highvar") == "highvar":
            p["columns"] = BLOCK  # stand-in columns on the mock data (smoke tests only)
        return TransductiveGMM(**{"random_state": random_state, **p}, source=src)
    columns = p.pop("columns", "highvar")
    pca_dims = p.pop("pca_dims", None)
    if src == "synthetic" and columns == "highvar":
        columns = BLOCK  # the mock data has no high-variance block: stand-in columns, smoke tests only
    if name == "subspace_qda":
        est = QuadraticDiscriminantAnalysis(**{"reg_param": 0.1, **p})
    elif name == "subspace_lda":
        est = LinearDiscriminantAnalysis(**{"solver": "lsqr", "shrinkage": "auto", **p})
    elif name == "subspace_logreg":
        est = LogisticRegression(**{"solver": "liblinear", "max_iter": 5000, **p}, random_state=random_state)
    else:
        raise ValueError(f"unknown extra model {name!r}")
    return SubspaceModel(est, columns=columns, pca_dims=pca_dims, source=src)


def build_pipeline(spec: dict):
    model = spec["model"]
    if model["name"] not in EXTRA_MODELS:
        return _ssc_build_pipeline(spec)
    if spec.get("scaler", "standard") not in (None, "none") or spec.get("selector"):
        raise ValueError(f"{model['name']} standardises and selects columns itself: use scaler: none, no selector")
    return Pipeline([("model", build_estimator(model["name"], model.get("params"),
                                               int(spec.get("random_state", 0))))])
