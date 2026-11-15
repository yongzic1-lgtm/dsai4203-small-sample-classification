"""Tests for ssc.validation: split generation, CV bookkeeping, paired comparison, AUC SE.

The split tests pin the conventions the extensions rely on (E1: only the split
seed varies; paired comparison needs identical splits)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ssc import paths
from ssc.models import build_pipeline, predict_score
from ssc.validation import (Scheme, Split, auc_standard_error, cross_validate, make_splits,
                            paired_compare, run_cv, split_signature)

LOGREG = {"model": {"name": "logreg", "params": {"C": 0.5}}}


def _val_sets(splits, repeat):
    return [frozenset(s.val_idx.tolist()) for s in splits if s.repeat == repeat]


# ---------------------------------------------------------------- make_splits: kfold
@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_kfold_partitions_every_row_once_per_repeat(y_250, n_splits):
    n = len(y_250)
    splits = make_splits(y_250, Scheme(kind="kfold", n_splits=n_splits, n_repeats=4, seed=11))
    assert len(splits) == n_splits * 4
    for s in splits:
        assert set(s.train_idx).isdisjoint(s.val_idx)
        assert len(s.train_idx) + len(s.val_idx) == n
        assert np.array_equal(np.sort(np.concatenate([s.train_idx, s.val_idx])), np.arange(n))
    for r in range(4):
        folds = [s for s in splits if s.repeat == r]
        assert [s.fold for s in folds] == list(range(n_splits))
        allval = np.sort(np.concatenate([s.val_idx for s in folds]))
        assert np.array_equal(allval, np.arange(n))  # union = all rows, each exactly once


@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_kfold_is_stratified(y_250, n_splits):
    expected = y_250.sum() / n_splits
    for s in make_splits(y_250, Scheme(n_splits=n_splits, n_repeats=3, seed=5)):
        assert abs(y_250[s.val_idx].sum() - expected) < 1.0 + 1e-9  # within +-1 positive


def test_kfold_unstratified_still_partitions(y_250):
    splits = make_splits(y_250, Scheme(n_splits=5, n_repeats=2, seed=1, stratify=False))
    for r in range(2):
        allval = np.sort(np.concatenate([s.val_idx for s in splits if s.repeat == r]))
        assert np.array_equal(allval, np.arange(len(y_250)))


def test_different_seeds_give_different_splits(y_250):
    a = make_splits(y_250, Scheme(n_splits=5, n_repeats=2, seed=1))
    b = make_splits(y_250, Scheme(n_splits=5, n_repeats=2, seed=100))
    assert _val_sets(a, 0) != _val_sets(b, 0)
    assert _val_sets(a, 0) != _val_sets(a, 1)  # repeats within one scheme differ too
    assert split_signature(a) != split_signature(b)


def test_repeat_r_uses_seed_plus_r(y_250):
    """Documented convention: repeat r <-> random_state seed + r. Consequence for E1:
    schemes with seeds s and s+1 share all but one repeat, so seeds used to compare
    'independent' repetitions must be spaced by at least n_repeats."""
    a = make_splits(y_250, Scheme(n_splits=5, n_repeats=3, seed=40))
    b = make_splits(y_250, Scheme(n_splits=5, n_repeats=3, seed=41))
    assert _val_sets(a, 1) == _val_sets(b, 0)
    assert _val_sets(a, 2) == _val_sets(b, 1)


def test_noshuffle_is_identical_across_repeats_and_seeds(y_250):
    s1 = make_splits(y_250, Scheme(n_splits=5, n_repeats=3, seed=1, shuffle=False))
    s2 = make_splits(y_250, Scheme(n_splits=5, n_repeats=3, seed=999, shuffle=False))
    assert _val_sets(s1, 0) == _val_sets(s1, 1) == _val_sets(s1, 2)
    assert split_signature(s1) == split_signature(s2)  # seed is irrelevant without shuffling
    ref = [frozenset(va.tolist()) for _, va in StratifiedKFold(5).split(np.zeros((250, 1)), y_250)]
    assert _val_sets(s1, 0) == ref  # == sklearn's default, un-shuffled StratifiedKFold


def test_unknown_scheme_kind_raises(y_250):
    with pytest.raises(ValueError, match="unknown scheme kind"):
        make_splits(y_250, Scheme(kind="loo"))


# ---------------------------------------------------------------- make_splits: holdout
@pytest.mark.parametrize("test_size", [0.2, 0.3, 0.5])
def test_holdout_sizes_and_stratification(y_250, test_size):
    n, n_pos = len(y_250), int(y_250.sum())
    splits = make_splits(y_250, Scheme(kind="holdout", test_size=test_size, n_repeats=6, seed=3))
    assert len(splits) == 6
    assert [s.repeat for s in splits] == list(range(6)) and {s.fold for s in splits} == {0}
    n_val = math.ceil(test_size * n)  # sklearn rounds the test part up
    for s in splits:
        assert len(s.val_idx) == n_val and len(s.train_idx) == n - n_val
        assert set(s.train_idx).isdisjoint(s.val_idx)
        assert abs(y_250[s.val_idx].sum() - test_size * n_pos) <= 1.0
    assert len({frozenset(s.val_idx.tolist()) for s in splits}) == 6  # every repeat is a new split


# ---------------------------------------------------------------- split_signature / Scheme
def test_split_signature_stable_and_sensitive(y_250):
    sch = Scheme(n_splits=5, n_repeats=3, seed=2026)
    sig = split_signature(make_splits(y_250, sch))
    assert sig == split_signature(make_splits(y_250, Scheme(n_splits=5, n_repeats=3, seed=2026)))
    assert len(sig) == 12 and all(c in "0123456789abcdef" for c in sig)
    assert sig != split_signature(make_splits(y_250, Scheme(n_splits=5, n_repeats=3, seed=2027)))
    assert sig != split_signature(make_splits(y_250, Scheme(n_splits=3, n_repeats=3, seed=2026)))
    assert sig != split_signature(make_splits(y_250, Scheme(n_splits=5, n_repeats=2, seed=2026)))


def test_split_signature_ignores_index_order_within_a_split():
    rng = np.random.default_rng(0)
    va = np.arange(10, 20)
    a = [Split(0, 0, np.arange(10), va)]
    b = [Split(0, 0, np.arange(10), rng.permutation(va))]
    assert split_signature(a) == split_signature(b)


def test_scheme_dict_roundtrip_and_labels():
    sch = Scheme(kind="kfold", n_splits=10, n_repeats=7, seed=3)
    assert Scheme.from_dict(sch.to_dict()) == sch
    assert Scheme.from_dict({"scheme": "holdout", "test_size": 0.25}).kind == "holdout"
    assert Scheme.from_dict(None) == Scheme()
    assert Scheme(n_splits=5, n_repeats=20).label() == "5foldx20"
    assert Scheme(n_splits=5, n_repeats=3, shuffle=False).label() == "5foldx3_noshuffle"
    assert Scheme(kind="holdout", test_size=0.2, n_repeats=10).label() == "holdout20x10"


def test_official_cv_in_project_yaml_is_a_valid_scheme(y_250):
    cfg = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))
    sch = Scheme.from_dict(cfg["official_cv"])
    assert (sch.kind, sch.n_splits, sch.n_repeats, sch.seed) == ("kfold", 5, 20, 2026)
    assert len(make_splits(y_250, sch)) == 100


# ---------------------------------------------------------------- cross_validate
def test_cv_kfold_shapes_and_full_oof(small_xy):
    X, y, Xt = small_xy
    sch = Scheme(n_splits=5, n_repeats=2, seed=1)
    res = run_cv(build_pipeline(LOGREG), X, y, sch, X_test=Xt)
    fs = res.fold_scores
    assert {"repeat", "fold", "auc", "n_train", "n_val", "fit_seconds"} <= set(fs.columns)
    assert len(fs) == 10 and fs[["repeat", "fold"]].drop_duplicates().shape[0] == 10
    assert res.oof.shape == (2, len(y)) and not np.isnan(res.oof).any()
    assert res.test_pred.shape == (len(Xt),) and np.isfinite(res.test_pred).all()
    assert res.split_signature == split_signature(make_splits(y, sch))
    assert res.scheme == sch.to_dict()
    for row in fs.itertuples():  # fold AUC is computed from the stored OOF scores
        va = make_splits(y, sch)[row.repeat * 5 + row.fold].val_idx
        assert row.auc == pytest.approx(roc_auc_score(y[va], res.oof[row.repeat, va]))
        assert row.n_val == len(va) and row.n_train == len(y) - len(va)


def test_cv_oof_and_test_pred_match_manual_fits(small_xy):
    X, y, Xt = small_xy
    est = build_pipeline(LOGREG)
    splits = make_splits(y, Scheme(n_splits=3, n_repeats=2, seed=4))
    res = cross_validate(est, X, y, splits, X_test=Xt, test_pred="fold_mean")
    preds = []
    for s in splits:
        m = clone(est).fit(X[s.train_idx], y[s.train_idx])
        np.testing.assert_allclose(res.oof[s.repeat, s.val_idx], predict_score(m, X[s.val_idx]))
        preds.append(predict_score(m, Xt))
    np.testing.assert_allclose(res.test_pred, np.mean(preds, axis=0))


def test_cv_test_pred_modes(small_xy):
    X, y, Xt = small_xy
    est = build_pipeline(LOGREG)
    splits = make_splits(y, Scheme(n_splits=3, n_repeats=1, seed=4))
    refit = cross_validate(est, X, y, splits, X_test=Xt, test_pred="refit").test_pred
    np.testing.assert_allclose(refit, predict_score(clone(est).fit(X, y), Xt))
    fold_mean = cross_validate(est, X, y, splits, X_test=Xt, test_pred="fold_mean").test_pred
    assert not np.allclose(refit, fold_mean)  # the two modes really differ
    assert cross_validate(est, X, y, splits, X_test=Xt, test_pred="none").test_pred is None
    assert cross_validate(est, X, y, splits, X_test=None).test_pred is None
    # the fold scores do not depend on the test-prediction mode
    a = cross_validate(est, X, y, splits, X_test=Xt, test_pred="refit").fold_scores["auc"]
    b = cross_validate(est, X, y, splits, test_pred="none").fold_scores["auc"]
    np.testing.assert_allclose(a, b)


def test_cv_holdout_oof_nan_pattern(small_xy):
    X, y, _ = small_xy
    sch = Scheme(kind="holdout", test_size=0.25, n_repeats=3, seed=2)
    res = run_cv(build_pipeline(LOGREG), X, y, sch, test_pred="none")
    splits = make_splits(y, sch)
    assert res.oof.shape == (3, len(y)) and len(res.fold_scores) == 3
    for s in splits:
        filled = ~np.isnan(res.oof[s.repeat])
        assert np.array_equal(np.flatnonzero(filled), np.sort(s.val_idx))


def test_cv_parallel_equals_serial(small_xy):
    X, y, _ = small_xy
    sch = Scheme(n_splits=3, n_repeats=2, seed=8)
    a = run_cv(build_pipeline(LOGREG), X, y, sch, test_pred="none", n_jobs=1)
    b = run_cv(build_pipeline(LOGREG), X, y, sch, test_pred="none", n_jobs=2)
    np.testing.assert_allclose(a.fold_scores["auc"], b.fold_scores["auc"])
    np.testing.assert_allclose(a.oof, b.oof)


BASE_KEYS = {"auc_mean", "auc_std_fold", "auc_std_repeat", "auc_min_repeat", "auc_max_repeat",
             "n_folds_total", "split_signature"}


def test_summary_keys_and_values_kfold(small_xy):
    X, y, _ = small_xy
    res = run_cv(build_pipeline(LOGREG), X, y, Scheme(n_splits=4, n_repeats=3, seed=0), test_pred="none")
    s = res.summary()
    assert set(s) == BASE_KEYS | {"oof_auc_mean", "oof_auc_std"}
    fs = res.fold_scores
    rep = fs.groupby("repeat")["auc"].mean()
    assert s["auc_mean"] == pytest.approx(fs["auc"].mean())
    assert s["auc_std_fold"] == pytest.approx(fs["auc"].std(ddof=1))
    assert s["auc_std_repeat"] == pytest.approx(rep.std(ddof=1))
    assert (s["auc_min_repeat"], s["auc_max_repeat"]) == pytest.approx((rep.min(), rep.max()))
    assert s["n_folds_total"] == 12 and s["split_signature"] == res.split_signature
    assert s["oof_auc_mean"] == pytest.approx(np.mean([roc_auc_score(y, r) for r in res.oof]))


def test_summary_keys_holdout_and_single_repeat(small_xy):
    X, y, _ = small_xy
    s = run_cv(build_pipeline(LOGREG), X, y, Scheme(kind="holdout", n_repeats=2, seed=0),
               test_pred="none").summary()
    assert set(s) == BASE_KEYS  # no pooled OOF AUC: holdout validates only part of the rows
    s1 = run_cv(build_pipeline(LOGREG), X, y, Scheme(n_splits=3, n_repeats=1, seed=0),
                test_pred="none").summary()
    assert math.isnan(s1["auc_std_repeat"]) and math.isnan(s1["oof_auc_std"])


# ---------------------------------------------------------------- paired_compare
def _fold_scores(aucs, n_folds=2, n_train=200, n_val=50):
    """fold_scores frame with repeat-major (repeat, fold) order."""
    return pd.DataFrame({"repeat": [i // n_folds for i in range(len(aucs))],
                         "fold": [i % n_folds for i in range(len(aucs))],
                         "auc": aucs, "n_train": n_train, "n_val": n_val, "fit_seconds": 0.0})


def test_paired_compare_identical_models(small_xy):
    X, y, _ = small_xy
    sch = Scheme(n_splits=5, n_repeats=2, seed=6)
    a = run_cv(build_pipeline(LOGREG), X, y, sch, test_pred="none")
    b = run_cv(build_pipeline(LOGREG), X, y, sch, test_pred="none")
    r = paired_compare(a.fold_scores, b.fold_scores)
    assert r["mean_diff"] == 0.0 and r["std_diff_fold"] == 0.0
    assert r["win_rate_fold"] == 0.5  # ties count 1/2
    assert r["n_folds"] == 10 and r["n_repeats"] == 2 and r["se_split"] == 0.0
    assert r["corrected_t"] == 0.0 and r["corrected_p"] == 1.0  # zero variance, zero mean: no evidence


def test_paired_compare_constant_nonzero_difference():
    b = _fold_scores([0.5, 0.625, 0.75, 0.875])  # dyadic values: every difference is exactly 0.125
    r = paired_compare(b.assign(auc=b["auc"] + 0.125), b)
    assert r["std_diff_fold"] == 0.0 and r["corrected_t"] == math.inf and r["corrected_p"] == 0.0


def test_paired_compare_se_split_is_repeat_std_over_sqrt_r():
    b = _fold_scores([0.70] * 8)
    a = _fold_scores([0.71, 0.71, 0.73, 0.73, 0.72, 0.72, 0.74, 0.74])  # repeat means of d: .01 .03 .02 .04
    r = paired_compare(a, b)
    assert r["n_repeats"] == 4
    assert r["se_split"] == pytest.approx(r["std_diff_repeat"] / 2)


def test_fold_scores_carry_val_id_and_constant_flag(small_xy):
    X, y, _ = small_xy
    res = run_cv(build_pipeline(LOGREG), X, y, Scheme(n_splits=3, n_repeats=2, seed=0), test_pred="none")
    fs = res.fold_scores
    assert fs["val_id"].str.match(r"^v[0-9a-f]{10}$").all() and fs["val_id"].nunique() == 6
    assert fs["constant_pred"].dtype == bool and not fs["constant_pred"].any()
    # a model with every coefficient at 0 predicts one constant score -> flagged
    zero = build_pipeline({"model": {"name": "logreg", "params": {"C": 1e-6, "l1_ratio": 1.0}}})
    fz = run_cv(zero, X, y, Scheme(n_splits=3, n_repeats=1, seed=0), test_pred="none").fold_scores
    assert fz["constant_pred"].all() and (fz["auc"] == 0.5).all()


def test_cross_validate_rejects_unknown_test_pred(small_xy):
    X, y, _ = small_xy
    sch = Scheme(n_splits=3, n_repeats=1, seed=0)
    with pytest.raises(ValueError, match="test_pred"):
        run_cv(build_pipeline(LOGREG), X, y, sch, test_pred="refit ")


def test_paired_compare_mismatched_fold_sets_raise(small_xy):
    X, y, _ = small_xy
    est = build_pipeline(LOGREG)
    five = run_cv(est, X, y, Scheme(n_splits=5, n_repeats=2, seed=6), test_pred="none")
    three = run_cv(est, X, y, Scheme(n_splits=3, n_repeats=2, seed=6), test_pred="none")
    more = run_cv(est, X, y, Scheme(n_splits=5, n_repeats=3, seed=6), test_pred="none")
    with pytest.raises(ValueError, match="same splits"):
        paired_compare(five.fold_scores, three.fold_scores)
    with pytest.raises(ValueError, match="same splits"):
        paired_compare(five.fold_scores, more.fold_scores)


def test_paired_compare_rejects_same_shape_different_seed(small_xy):
    """Same n_splits/n_repeats but another split seed: the per-fold val_id hashes differ."""
    X, y, _ = small_xy
    est = build_pipeline(LOGREG)
    a = run_cv(est, X, y, Scheme(n_splits=5, n_repeats=2, seed=1), test_pred="none")
    b = run_cv(est, X, y, Scheme(n_splits=5, n_repeats=2, seed=500), test_pred="none")
    assert a.split_signature != b.split_signature
    with pytest.raises(ValueError):
        paired_compare(a.fold_scores, b.fold_scores)


def test_paired_compare_sign_convention():
    b = _fold_scores([0.60, 0.62, 0.58, 0.61])
    a = b.assign(auc=b["auc"] + 0.02)
    ab, ba = paired_compare(a, b), paired_compare(b, a)
    assert ab["mean_diff"] == pytest.approx(0.02) and ba["mean_diff"] == pytest.approx(-0.02)
    assert ab["win_rate_fold"] == 1.0 and ba["win_rate_fold"] == 0.0


def test_paired_compare_nadeau_bengio_hand_computation():
    # 2 repeats x 2 folds, n_train=200, n_val=50, differences d = 0.01, 0.03, 0.02, 0.04
    b = _fold_scores([0.70, 0.70, 0.70, 0.70])
    a = _fold_scores([0.71, 0.73, 0.72, 0.74])
    r = paired_compare(a, b)
    # mean 0.025; deviations +-0.015, +-0.005 -> sum sq 5e-4, var(ddof=1) = 5e-4/3
    # corrected var of the mean = (1/J + n_val/n_train) * var = (1/4 + 1/4) * 5e-4/3 = 2.5e-4/3
    # t = 0.025 / sqrt(2.5e-4/3) = sqrt(0.000625 * 3 / 2.5e-4) = sqrt(7.5)
    # (the naive t = 0.025 / sqrt(var/J) would be sqrt(15): the correction divides t by sqrt(2))
    assert r["mean_diff"] == pytest.approx(0.025)
    assert r["std_diff_fold"] == pytest.approx(math.sqrt(5e-4 / 3))
    assert r["std_diff_repeat"] == pytest.approx(math.sqrt(0.5 * 0.01**2))  # repeat means 0.02, 0.03
    assert r["corrected_t"] == pytest.approx(math.sqrt(7.5))
    assert r["corrected_p"] == pytest.approx(2 * stats.t.sf(math.sqrt(7.5), df=3))
    assert r["n_folds"] == 4 and r["win_rate_fold"] == 1.0
    # explicit n_train / n_val override the values stored in the frames: (1/4 + 1) * var
    r2 = paired_compare(a, b, n_train=100, n_val=100)
    assert r2["corrected_t"] == pytest.approx(0.025 / math.sqrt(1.25 * 5e-4 / 3))


# ---------------------------------------------------------------- auc_standard_error
def test_auc_se_at_chance_equals_mann_whitney_null_variance():
    # At AUC = 0.5, Q1 = Q2 = 1/3 and the Hanley-McNeil variance reduces to the
    # exact null variance of the Mann-Whitney AUC: (n1 + n2 + 1) / (12 n1 n2).
    for n1, n2 in [(10, 10), (30, 70), (197, 198)]:
        assert auc_standard_error(0.5, n1, n2) == pytest.approx(math.sqrt((n1 + n2 + 1) / (12 * n1 * n2)))


def test_auc_se_hanley_mcneil_worked_example():
    # Hanley & McNeil (1982, Radiology 143:29-36): CT ratings of 51 abnormal and
    # 58 normal cases, A = 0.893, reported SE = 0.032.
    assert auc_standard_error(0.893, 51, 58) == pytest.approx(0.032, abs=1e-3)


def test_auc_se_monotonic_behaviour():
    ns = [10, 20, 50, 100, 400]
    se_n = [auc_standard_error(0.75, n, n) for n in ns]
    assert all(x > y for x, y in zip(se_n, se_n[1:]))  # more data -> smaller SE
    assert se_n[-1] / se_n[-2] == pytest.approx(0.5, rel=0.05)  # ~ 1/sqrt(n)
    aucs = np.linspace(0.5, 0.99, 25)
    se_a = [auc_standard_error(a, 50, 50) for a in aucs]
    assert all(x > y for x, y in zip(se_a, se_a[1:]))  # SE shrinks towards AUC = 1
    assert auc_standard_error(1.0, 50, 50) == pytest.approx(0.0, abs=1e-12)
    # public LB scale: ~395 rows, roughly balanced -> SE of a few hundredths
    assert 0.015 < auc_standard_error(0.8, 197, 198) < 0.03
