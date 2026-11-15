"""Tests for ssc.models: every named estimator builds and scores, spec handling,
predict_score routing, describe() strings, and no FutureWarning from our defaults
(sklearn 1.9 deprecations would otherwise turn into errors on the next upgrade)."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from ssc.models import ESTIMATORS, TRANSDUCTIVE, build_estimator, build_pipeline, describe, predict_score

VARIANTS = {
    "standard": {"scaler": "standard"},
    "none+corr_topk": {"scaler": "none", "selector": {"name": "corr_topk", "k": 10}},
    "robust+stability_topk": {"scaler": "robust",
                              "selector": {"name": "stability_topk", "k": 10, "n_boot": 20}},
    "standard+fixed_columns": {"selector": {"name": "fixed_columns", "columns": [0, 5, 9, 33]}},
}

# the generic tests below run every estimator inside scalers/selectors on in-memory data; the
# transductive ones (raw columns only, features of the on-disk source) are covered in test_transductive.py
GENERIC = sorted(n for n in ESTIMATORS if n not in TRANSDUCTIVE)

# estimator name -> reason, for defaults known to emit a FutureWarning (none at the moment;
# logreg_cv was fixed by setting l1_ratios=(0.0,) explicitly for sklearn 1.9)
KNOWN_FUTURE_WARNINGS: dict[str, str] = {}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("name", GENERIC)
def test_every_estimator_builds_and_scores(small_xy, name, variant):
    X, y, Xt = small_xy
    spec = {**VARIANTS[variant], "model": {"name": name}, "random_state": 0}
    pipe = build_pipeline(spec).fit(X, y)
    s = predict_score(pipe, Xt)
    assert s.shape == (len(Xt),) and s.dtype.kind == "f"
    assert np.all(np.isfinite(s))


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=KNOWN_FUTURE_WARNINGS[n]))
    if n in KNOWN_FUTURE_WARNINGS else n
    for n in GENERIC])
def test_default_params_raise_no_future_warning(small_xy, name):
    X, y, Xt = small_xy
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pipe = build_pipeline({"model": {"name": name}}).fit(X, y)
        predict_score(pipe, Xt)


def test_documented_logreg_l1_spec_works_without_warnings(small_xy):
    """The models.py docstring example (sklearn 1.9: l1_ratio replaces penalty)."""
    X, y, _ = small_xy
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        l1 = build_pipeline({"model": {"name": "logreg",
                                       "params": {"C": 0.1, "l1_ratio": 1.0, "solver": "liblinear"}}}).fit(X, y)
        l2 = build_pipeline({"model": {"name": "logreg",
                                       "params": {"C": 0.1, "l1_ratio": 0.0, "solver": "liblinear"}}}).fit(X, y)
    n_zero_l1 = int((l1.named_steps["model"].coef_ == 0).sum())
    assert n_zero_l1 > 100                                 # L1 really is sparse
    assert int((l2.named_steps["model"].coef_ == 0).sum()) == 0


def test_pipeline_steps_follow_spec():
    p = build_pipeline({"model": {"name": "gnb"}})
    assert [n for n, _ in p.steps] == ["scale", "model"] and isinstance(p.named_steps["scale"], StandardScaler)
    p = build_pipeline({"scaler": "robust", "selector": {"name": "corr_topk", "k": 3}, "model": {"name": "gnb"}})
    assert [n for n, _ in p.steps] == ["scale", "select", "model"]
    assert isinstance(p.named_steps["scale"], RobustScaler)
    assert [n for n, _ in build_pipeline({"scaler": "none", "model": {"name": "gnb"}}).steps] == ["model"]
    assert [n for n, _ in build_pipeline({"scaler": None, "model": {"name": "gnb"}}).steps] == ["model"]


def test_random_state_propagates_and_spec_not_mutated():
    spec = {"selector": {"name": "corr_topk", "k": 5},
            "model": {"name": "rf", "params": {"n_estimators": 10}}, "random_state": 7}
    before = copy.deepcopy(spec)
    p = build_pipeline(spec)
    assert p.named_steps["model"].random_state == 7 and p.named_steps["model"].n_estimators == 10
    assert spec == before
    assert build_pipeline({"model": {"name": "rf"}}).named_steps["model"].random_state == 0
    assert build_estimator({"name": "logreg", "params": None}).random_state == 0


def test_unknown_names_raise():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline({"model": {"name": "deep_net"}})
    with pytest.raises(ValueError, match="unknown selector"):
        build_pipeline({"selector": {"name": "mutual_info"}, "model": {"name": "logreg"}})
    # unknown scaler currently surfaces as a bare KeyError (see report); accept either
    with pytest.raises((KeyError, ValueError)):
        build_pipeline({"scaler": "minmax", "model": {"name": "logreg"}})


def test_predict_score_routing(small_xy):
    X, y, Xt = small_xy
    lr = build_pipeline({"model": {"name": "logreg"}}).fit(X, y)
    np.testing.assert_allclose(predict_score(lr, Xt), lr.predict_proba(Xt)[:, 1])
    svc = build_pipeline({"model": {"name": "svc"}}).fit(X, y)  # probability=False -> decision_function
    np.testing.assert_allclose(predict_score(svc, Xt), svc.decision_function(Xt))
    # (SVC(probability=True) is deprecated in sklearn 1.9, so it is deliberately not exercised)
    ridge = build_pipeline({"model": {"name": "ridge_clf"}}).fit(X, y)
    np.testing.assert_allclose(predict_score(ridge, Xt), ridge.decision_function(Xt))
    lasso = build_pipeline({"model": {"name": "lasso_reg"}}).fit(X, y)  # regressor -> predict
    np.testing.assert_allclose(predict_score(lasso, Xt), lasso.predict(Xt))


def test_predict_score_falls_back_when_predict_proba_raises():
    class ProbaRaises:
        def predict_proba(self, X):
            raise AttributeError("predict_proba is not available when probability=False")

        def decision_function(self, X):
            return np.asarray(X)[:, 0] * 2.0

    X = np.array([[1.0], [-3.0], [0.5]])
    np.testing.assert_allclose(predict_score(ProbaRaises(), X), [2.0, -6.0, 1.0])

    class PredictOnly:
        def predict(self, X):
            return [0, 1, 1]

    assert predict_score(PredictOnly(), X).dtype == float


def test_describe_strings():
    spec = {"scaler": "standard", "selector": {"name": "corr_topk", "k": 20},
            "model": {"name": "logreg", "params": {"solver": "liblinear", "C": 0.1, "l1_ratio": 1.0}}}
    assert describe(spec) == {"model": "logreg(C=0.1,l1_ratio=1.0,solver=liblinear)",
                              "selector": "corr_topk(k=20)", "scaler": "standard"}
    assert describe({"model": {"name": "gnb"}}) == {"model": "gnb()", "selector": "none", "scaler": "standard"}
    assert describe({"scaler": "none", "selector": {"name": "stability_topk", "threshold": 0.6, "k": 10},
                     "model": {"name": "lda", "params": {}}}) == {
        "model": "lda()", "selector": "stability_topk(k=10,threshold=0.6)", "scaler": "none"}
