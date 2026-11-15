"""Tests for ssc.transductive.TransductiveGMM, the model behind submission S003, and its config
builder in ssc.models (configs/final_s003_gmm2_hv.yaml): labels enter only through the training
rows of a fold, fits are deterministic for a fixed random_state, posteriors are probabilities,
and the model works through cross_validate / run_cv, also when built from the config.

The model reads the unlabeled train+test features of a data source from disk, so these tests
write a tiny two-cluster dataset (60 train + 400 test rows, 150 columns, a high-variance block
in columns 0-3) to a temp folder and point the "synthetic" source at it. The label-free caches
of ssc.transductive are cleared around every test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from ssc import data, paths, transductive
from ssc.data import FEATURES
from ssc.models import TRANSDUCTIVE, build_estimator, build_pipeline, describe
from ssc.transductive import TransductiveGMM, resolve_columns, unlabeled_features
from ssc.validation import Scheme, cross_validate, make_splits, run_cv

N_TRAIN, N_TEST, N_HV = 60, 400, 4
SCHEME = Scheme(kind="kfold", n_splits=5, n_repeats=2, seed=2026)


def _clear_caches():
    transductive.unlabeled_features.cache_clear()
    transductive._basis.cache_clear()
    transductive._mixture.cache_clear()


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Dataset of the "synthetic" source, redirected to a tiny two-cluster problem on disk."""
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    y = np.r_[rng.permutation(np.repeat([0, 1], N_TRAIN // 2)), rng.integers(0, 2, N_TEST)]
    X = rng.normal(size=(n, len(FEATURES)))
    X[:, :N_HV] += np.where(y[:, None] == 1, 3.0, -3.0)  # std ~3.2 vs 1: the label-free "highvar" block
    d = tmp_path / "tiny_data"
    d.mkdir()
    feats = pd.DataFrame(X, columns=FEATURES)
    ids = np.arange(n)
    pd.concat([pd.DataFrame({"id": ids[:N_TRAIN], "target": y[:N_TRAIN]}), feats.iloc[:N_TRAIN]], axis=1) \
        .to_csv(d / "train.csv", index=False)
    pd.concat([pd.DataFrame({"id": ids[N_TRAIN:]}), feats.iloc[N_TRAIN:].reset_index(drop=True)], axis=1) \
        .to_csv(d / "test.csv", index=False)
    pd.DataFrame({"id": ids[N_TRAIN:], "target": 0.5}).to_csv(d / "sample_submission.csv", index=False)
    monkeypatch.setattr(paths, "DATA_SYNTHETIC", d)
    _clear_caches()
    yield data.load("synthetic")
    _clear_caches()


def _xy(ds):
    return ds.X.to_numpy(), ds.y, ds.X_test.to_numpy()


# ---------------------------------------------------------------- labels only through the training fold
def test_highvar_block_is_chosen_from_features_alone(tiny):
    assert resolve_columns("highvar", unlabeled_features("synthetic")).tolist() == list(range(N_HV))


def test_mixture_is_label_free_and_labels_only_set_component_rates(tiny):
    X, y, Xt = _xy(tiny)
    tr, rest = np.arange(40), np.arange(40, N_TRAIN)
    m1 = TransductiveGMM(source="synthetic").fit(X[tr], y[tr])
    m2 = TransductiveGMM(source="synthetic").fit(X[tr], 1 - y[tr])
    # the mixture (columns, scaling, Gaussian components) does not depend on the labels
    for attr in ("weights_", "means_", "covariances_"):
        np.testing.assert_array_equal(getattr(m1.gmm_, attr), getattr(m2.gmm_, attr))
    # the labels of the training rows only estimate each component's class-1 rate (Beta smoothing, alpha 1)
    R = m1.gmm_.predict_proba(m1._transform(X[tr]))
    p0 = y[tr].mean()
    np.testing.assert_allclose(m1.component_rate_, (R.T @ y[tr] + 1.0 * p0) / (R.sum(0) + 1.0))
    # ... and they do matter: flipped training labels flip the scores of every other row
    assert not np.allclose(m1.predict_proba(X[rest]), m2.predict_proba(X[rest]))
    assert not np.allclose(m1.predict_proba(Xt), m2.predict_proba(Xt))


def test_validation_labels_never_reach_their_own_fold(tiny):
    X, y, Xt = _xy(tiny)
    splits = make_splits(y, SCHEME)
    base = cross_validate(TransductiveGMM(source="synthetic"), X, y, splits, SCHEME, X_test=Xt)
    s = splits[0]
    y2 = y.copy()
    y2[s.val_idx] = 1 - y2[s.val_idx]  # change only the labels of fold 0's validation rows
    alt = cross_validate(TransductiveGMM(source="synthetic"), X, y2, splits, SCHEME, X_test=Xt)
    np.testing.assert_array_equal(alt.oof[s.repeat, s.val_idx], base.oof[s.repeat, s.val_idx])
    # the other folds of that repeat train on those rows, so their scores (and the test scores) move
    others = np.setdiff1d(np.arange(N_TRAIN), s.val_idx)
    assert not np.array_equal(alt.oof[s.repeat, others], base.oof[s.repeat, others])
    assert not np.array_equal(alt.test_pred, base.test_pred)


# ---------------------------------------------------------------- determinism, output range
def test_fixed_random_state_gives_bit_identical_scores(tiny):
    X, y, Xt = _xy(tiny)
    p1 = TransductiveGMM(source="synthetic", random_state=3).fit(X, y).predict_proba(Xt)
    _clear_caches()  # force a fresh mixture fit instead of the cached one
    p2 = TransductiveGMM(source="synthetic", random_state=3).fit(X, y).predict_proba(Xt)
    np.testing.assert_array_equal(p1, p2)
    a = run_cv(TransductiveGMM(source="synthetic"), X, y, SCHEME, X_test=Xt)
    _clear_caches()
    b = run_cv(TransductiveGMM(source="synthetic"), X, y, SCHEME, X_test=Xt)
    np.testing.assert_array_equal(a.oof, b.oof)
    np.testing.assert_array_equal(a.test_pred, b.test_pred)


@pytest.mark.parametrize("n_components", [2, 3])
def test_posterior_shape_and_range(tiny, n_components):
    X, y, Xt = _xy(tiny)
    m = TransductiveGMM(source="synthetic", n_components=n_components).fit(X, y)
    P = m.predict_proba(Xt)
    assert P.shape == (N_TEST, 2)
    assert np.all((P >= 0) & (P <= 1))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert m.component_rate_.shape == (n_components,)
    assert np.all((m.component_rate_ >= 0) & (m.component_rate_ <= 1))
    assert list(m.classes_) == [0, 1] and m.n_features_in_ == len(FEATURES)
    assert set(np.unique(m.predict(Xt))) <= {0, 1}


def test_cross_validate_on_tiny_dataset(tiny):
    # n_jobs=1: worker processes would not see the monkeypatched data folder
    cv = cross_validate(TransductiveGMM(source="synthetic"), tiny.X, tiny.y, make_splits(tiny.y, SCHEME), SCHEME,
                        X_test=tiny.X_test)
    assert len(cv.fold_scores) == 10 and cv.fold_scores["auc"].min() > 0.95  # two clean clusters
    assert cv.summary()["oof_auc_mean"] > 0.95
    assert cv.test_pred.shape == (N_TEST,) and np.all(np.isfinite(cv.test_pred))
    assert cv.test_pred.min() >= 0 and cv.test_pred.max() <= 1
    assert not np.isnan(cv.oof).any()


# ---------------------------------------------------------------- config builder (ssc.models)
def test_s003_config_builds_the_r003_estimator(tiny):
    cfg = yaml.safe_load((paths.CONFIGS / "final_s003_gmm2_hv.yaml").read_text(encoding="utf-8"))
    assert "cv" not in cfg and cfg["test_pred"] == "fold_mean"  # official CV from project.yaml, fold mean
    pipe = build_pipeline(cfg["pipeline"], source="synthetic")
    assert [n for n, _ in pipe.steps] == ["model"]
    ref = TransductiveGMM("highvar", 2, source="synthetic")  # research/r003 CANDIDATES["gmm2_hv"] (S003)
    assert pipe.named_steps["model"].get_params() == ref.get_params()
    a = run_cv(pipe, tiny.X, tiny.y, SCHEME, X_test=tiny.X_test)
    b = cross_validate(ref, tiny.X, tiny.y, make_splits(tiny.y, SCHEME), SCHEME, X_test=tiny.X_test)
    pd.testing.assert_series_equal(a.fold_scores["auc"], b.fold_scores["auc"])
    np.testing.assert_array_equal(a.oof, b.oof)
    np.testing.assert_array_equal(a.test_pred, b.test_pred)
    assert describe(cfg["pipeline"])["model"].startswith("transductive_gmm(alpha=1.0,columns=highvar,")


def test_builder_takes_the_run_source_and_needs_raw_columns():
    assert "transductive_gmm" in TRANSDUCTIVE
    est = build_estimator({"name": "transductive_gmm"}, random_state=4, source="raw")
    assert isinstance(est, TransductiveGMM) and est.source == "raw" and est.random_state == 4
    # no run source given: params.source, else $SSC_DATA (conftest sets "synthetic")
    assert build_estimator({"name": "transductive_gmm"}).source == "synthetic"
    assert build_estimator({"name": "transductive_gmm", "params": {"source": "raw"}}).source == "raw"
    with pytest.raises(ValueError, match="differs from the run's data source"):
        build_estimator({"name": "transductive_gmm", "params": {"source": "raw"}}, source="synthetic")
    assert build_pipeline({"scaler": "none", "model": {"name": "transductive_gmm"}}, source="raw") \
        .named_steps["model"].source == "raw"
    for bad in ({}, {"scaler": "standard"}, {"scaler": "none", "selector": {"name": "corr_topk", "k": 5}}):
        with pytest.raises(ValueError, match="scaler: none and no selector"):
            build_pipeline({**bad, "model": {"name": "transductive_gmm"}})
