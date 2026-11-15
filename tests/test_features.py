"""Tests for ssc.features: correlation scores, top-k selection, selectors, bootstrap
frequencies, and a demonstration of the selection leakage that extension E3 measures."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conftest import frames_to_xy
from ssc import synthetic
from ssc.features import (CorrelationTopK, FixedColumns, StabilityTopK, abs_correlation,
                          bootstrap_topk_frequency, top_k_indices)
from ssc.models import build_pipeline
from ssc.validation import Scheme, run_cv


@pytest.fixture(scope="module")
def mixed_xy():
    """60 rows, 8 columns with very different scales, some ties, binary y."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=60)
    X = rng.normal(size=(60, 8)) * np.array([1, 10, 0.01, 1e3, 1, 1, 5, 2])
    X[:, 0] += 1.5 * y                    # one clearly informative column
    X[:, 5] = np.round(X[:, 5])           # heavy ties (tests average ranks)
    X[:, 6] = -2.0 * X[:, 0] + 0.1 * X[:, 6]  # negatively correlated copy
    return X, y


# ---------------------------------------------------------------- abs_correlation
def test_abs_correlation_matches_numpy_pearson(mixed_xy):
    X, y = mixed_xy
    expected = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
    np.testing.assert_allclose(abs_correlation(X, y), expected, rtol=1e-10, atol=1e-12)


def test_abs_correlation_matches_scipy_spearman(mixed_xy):
    X, y = mixed_xy
    expected = [abs(spearmanr(X[:, j], y).statistic) for j in range(X.shape[1])]
    np.testing.assert_allclose(abs_correlation(X, y, method="spearman"), expected, rtol=1e-10, atol=1e-12)


def test_abs_correlation_continuous_target_and_sign(mixed_xy):
    X, _ = mixed_xy
    t = X[:, 1] * -3.0 + 7.0  # perfect negative linear relation with column 1
    r = abs_correlation(X, t)
    assert r[1] == pytest.approx(1.0) and np.all((r >= 0) & (r <= 1 + 1e-12))


def test_abs_correlation_constant_column_is_zero(mixed_xy):
    X, y = mixed_xy
    Xc = X.copy()
    Xc[:, 2] = 0.1   # constants whose mean is not exactly representable
    Xc[:, 3] = 3.7
    Xc[:, 4] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning from 0/0
        for method in ("pearson", "spearman"):
            r = abs_correlation(Xc, y, method=method)
            assert np.all(np.isfinite(r))
            np.testing.assert_allclose(r[[2, 3, 4]], 0.0, atol=1e-10)


def test_abs_correlation_rejects_unknown_method(mixed_xy):
    with pytest.raises(ValueError, match="pearson"):
        abs_correlation(*mixed_xy, method="kendall")


# ---------------------------------------------------------------- top_k_indices
def test_top_k_indices_ties_broken_by_lower_index():
    scores = np.array([0.5, 0.9, 0.9, 0.1, 0.9, 0.5])
    assert top_k_indices(scores, 2).tolist() == [1, 2]
    assert top_k_indices(scores, 3).tolist() == [1, 2, 4]
    assert top_k_indices(scores, 4).tolist() == [0, 1, 2, 4]   # 0 beats 5 on the 0.5 tie
    assert top_k_indices(np.zeros(5), 3).tolist() == [0, 1, 2]
    assert top_k_indices(scores, 100).tolist() == list(range(6))  # k > n -> everything
    assert top_k_indices(scores, 0).tolist() == []


def test_top_k_indices_returns_sorted_indices_of_largest():
    rng = np.random.default_rng(1)
    s = rng.random(50)
    got = top_k_indices(s, 7)
    assert np.all(np.diff(got) > 0)
    assert set(got) == set(np.argsort(-s)[:7])


# ---------------------------------------------------------------- CorrelationTopK
@pytest.mark.parametrize("k", [1, 5, 20, 149])
def test_correlation_topk_keeps_exactly_k(small_xy, k):
    X, y, _ = small_xy
    sel = CorrelationTopK(k=k).fit(X, y)
    assert sel.get_support().sum() == k
    assert sel.transform(X).shape == (len(y), k)
    assert sel.get_support(indices=True).tolist() == top_k_indices(abs_correlation(X, y), k).tolist()
    np.testing.assert_allclose(sel.scores_, abs_correlation(X, y))


def test_correlation_topk_all_and_oversized_k(small_xy):
    X, y, _ = small_xy
    assert CorrelationTopK(k="all").fit(X, y).get_support().all()
    assert CorrelationTopK(k=500).fit(X, y).get_support().all()
    sp = CorrelationTopK(k=10, method="spearman").fit(X, y)
    assert sp.get_support(indices=True).tolist() == top_k_indices(abs_correlation(X, y, "spearman"), 10).tolist()


def test_correlation_topk_is_sklearn_clonable():
    sel = CorrelationTopK(k=3, method="spearman")
    assert clone(sel).get_params() == {"k": 3, "method": "spearman"}


def test_correlation_topk_inside_pipeline(small_xy):
    X, y, Xt = small_xy
    pipe = Pipeline([("scale", StandardScaler()), ("select", CorrelationTopK(k=7)),
                     ("model", LogisticRegression(solver="liblinear"))]).fit(X, y)
    support = pipe.named_steps["select"].get_support()
    assert support.sum() == 7 and pipe.named_steps["model"].coef_.shape == (1, 7)
    # |corr| is invariant to standardisation, so the in-pipeline choice equals the raw-data choice
    assert np.flatnonzero(support).tolist() == top_k_indices(abs_correlation(X, y), 7).tolist()
    assert pipe.predict_proba(Xt).shape == (len(Xt), 2)
    # refitting on a subset re-selects from that subset only (what happens inside each CV fold)
    sub = build_pipeline({"selector": {"name": "corr_topk", "k": 7}, "model": {"name": "logreg"}})
    sub.fit(X[:60], y[:60])
    got = sub.named_steps["select"].get_support(indices=True).tolist()
    assert got == top_k_indices(abs_correlation(X[:60], y[:60]), 7).tolist()


def test_selectors_accept_dataframes_without_warning(small_xy):
    """Selectors use sklearn's validate_data, so a DataFrame fit + transform does not warn
    'X has feature names, but CorrelationTopK was fitted without feature names'."""
    X, y, _ = small_xy
    df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for sel in (CorrelationTopK(k=5), StabilityTopK(k=5, n_boot=10), FixedColumns([0, 3])):
            sel.fit(df, y).transform(df)


# ---------------------------------------------------------------- StabilityTopK
def test_stability_topk_never_returns_empty(small_xy):
    X, y, _ = small_xy
    sel = StabilityTopK(k=5, n_boot=30, threshold=1.01).fit(X, y)  # impossible threshold
    assert sel.get_support().sum() == 1
    assert sel.get_support(indices=True)[0] == int(np.argmax(sel.frequency_))
    noise = np.random.default_rng(3).normal(size=(80, 150))
    ynoise = np.random.default_rng(4).integers(0, 2, size=80)
    assert StabilityTopK(k=3, n_boot=30, threshold=0.99).fit(noise, ynoise).get_support().sum() >= 1


def test_stability_topk_frequency_and_determinism(small_xy):
    X, y, _ = small_xy
    a = StabilityTopK(k=10, n_boot=50, threshold=0.5, random_state=1).fit(X, y)
    b = StabilityTopK(k=10, n_boot=50, threshold=0.5, random_state=1).fit(X, y)
    assert a.frequency_.shape == (150,) and np.all((a.frequency_ >= 0) & (a.frequency_ <= 1))
    np.testing.assert_array_equal(a.get_support(), b.get_support())
    np.testing.assert_array_equal(a.get_support(), a.frequency_ >= 0.5)


def test_stability_topk_prefers_informative_features():
    frames = synthetic.make(n_test=10, seed=2026, signal=1.0)
    X, y, _ = frames_to_xy(frames)
    truth = {int(c) for c in frames["truth"]["feature"]}
    chosen = set(StabilityTopK(k=10, n_boot=100, threshold=0.5).fit(X, y).get_support(indices=True).tolist())
    assert len(chosen & truth) >= 0.6 * len(chosen)  # mostly real signal, not noise


# ---------------------------------------------------------------- FixedColumns
def test_fixed_columns(small_xy):
    X, y, _ = small_xy
    sel = FixedColumns(columns=[7, 0, 3]).fit(X)  # y not needed
    assert sel.get_support(indices=True).tolist() == [0, 3, 7]
    np.testing.assert_array_equal(sel.transform(X), X[:, [0, 3, 7]])
    pipe = build_pipeline({"selector": {"name": "fixed_columns", "columns": [1, 2]},
                           "model": {"name": "logreg"}}).fit(X, y)
    assert pipe.named_steps["model"].coef_.shape == (1, 2)


# ---------------------------------------------------------------- bootstrap_topk_frequency
@pytest.mark.parametrize("k", [1, 10, 30])
def test_bootstrap_frequency_range_and_sum(small_xy, k):
    X, y, _ = small_xy
    f = bootstrap_topk_frequency(X, y, k=k, n_boot=40, random_state=0)
    assert f.shape == (X.shape[1],)
    assert np.all((f >= 0) & (f <= 1))
    assert f.sum() == pytest.approx(k)  # every resample contributes exactly k selections


def test_bootstrap_frequency_seed_behaviour(small_xy):
    X, y, _ = small_xy
    a = bootstrap_topk_frequency(X, y, k=10, n_boot=40, random_state=5)
    np.testing.assert_array_equal(a, bootstrap_topk_frequency(X, y, k=10, n_boot=40, random_state=5))
    assert not np.array_equal(a, bootstrap_topk_frequency(X, y, k=10, n_boot=40, random_state=6))
    assert np.all(bootstrap_topk_frequency(X[:, :5], y, k=5, n_boot=10) == 1.0)  # k = all features


def test_bootstrap_frequency_strong_feature_always_selected():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    X = rng.normal(size=(200, 50))
    X[:, 17] += 3.0 * y
    f = bootstrap_topk_frequency(X, y, k=5, n_boot=100, random_state=0)
    assert f[17] == 1.0


# ---------------------------------------------------------------- leakage demonstration (E3)
def test_selection_outside_cv_leaks_on_pure_noise():
    """Labels independent of features, so the true AUC of any model is 0.5.
    (1) choose the top-k columns on ALL rows, then CV on the reduced data:
        the validation rows helped pick the columns -> optimistic AUC;
    (2) selector inside the pipeline (re-fitted on each training fold): ~0.5."""
    k = 10
    sch = Scheme(n_splits=5, n_repeats=3, seed=0)
    model = {"model": {"name": "logreg", "params": {"C": 1.0}}}
    leaky, honest = [], []
    for seed in (0, 1, 2):
        X, y, _ = frames_to_xy(synthetic.make(n_informative=0, n_test=10, seed=seed))
        cols = CorrelationTopK(k=k).fit(X, y).get_support()           # sees every label
        leaky.append(run_cv(build_pipeline(model), X[:, cols], y, sch, test_pred="none").summary()["auc_mean"])
        inside = {**model, "selector": {"name": "corr_topk", "k": k}}
        honest.append(run_cv(build_pipeline(inside), X, y, sch, test_pred="none").summary()["auc_mean"])
    leaky_m, honest_m = float(np.mean(leaky)), float(np.mean(honest))
    assert leaky_m > 0.62, leaky
    assert abs(honest_m - 0.5) < 0.08, honest
    assert leaky_m - honest_m > 0.12
    assert all(l > h for l, h in zip(leaky, honest))
