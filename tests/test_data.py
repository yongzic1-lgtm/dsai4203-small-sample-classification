"""Tests for ssc.data (load / verify / fingerprint) and ssc.paths source resolution.

The real Kaggle data is only READ (verify('raw')); corrupted-data cases are built
in a temp dir from ssc.synthetic.make()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ssc import data, paths, synthetic
from ssc.data import FEATURES, data_fingerprint, load, verify


def _write_fake_dir(d, n_test: int = 50, seed: int = 1) -> dict[str, pd.DataFrame]:
    frames = synthetic.make(n_test=n_test, seed=seed)
    d.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "sample_submission"):
        frames[name].to_csv(d / f"{name}.csv", index=False)
    return frames


# ---------------------------------------------------------------- synthetic data on disk
@pytest.mark.synthetic_data
def test_verify_synthetic_is_clean():
    assert verify("synthetic") == []


def test_load_synthetic_shapes_and_dtypes(synthetic_ds):
    ds = synthetic_ds
    assert ds.source == "synthetic" and ds.data_dir == paths.DATA_SYNTHETIC
    assert ds.X.shape == (250, 150) and ds.X_test.shape == (19750, 150)
    assert ds.y.shape == (250,) and ds.ids.shape == (250,) and ds.ids_test.shape == (19750,)
    assert ds.n_train == 250
    assert list(ds.X.columns) == FEATURES and list(ds.X_test.columns) == FEATURES
    assert all(dt == np.float64 for dt in ds.X.dtypes) and all(dt == np.float64 for dt in ds.X_test.dtypes)
    assert ds.y.dtype.kind == "i" and set(np.unique(ds.y)) <= {0, 1}
    sample = data.load_sample_submission("synthetic")
    assert np.array_equal(ds.ids_test, sample["id"].to_numpy())


def test_synthetic_files_match_the_documented_generator(synthetic_ds):
    """data/synthetic was produced by ssc.synthetic.write() with the default seed 2026
    and has not been edited since (checked in memory; nothing is rewritten)."""
    frames = synthetic.make(seed=2026)
    train = frames["train"]
    np.testing.assert_allclose(synthetic_ds.X.to_numpy(), train[FEATURES].to_numpy(), rtol=1e-12)
    np.testing.assert_array_equal(synthetic_ds.y, train["target"].to_numpy())
    truth = pd.read_csv(paths.DATA_SYNTHETIC / "truth.csv")
    assert truth["feature"].astype(str).tolist() == frames["truth"]["feature"].tolist()


def test_summary_of_synthetic(synthetic_ds):
    s = data.summary(synthetic_ds)
    assert s["n_train"] == 250 and s["n_test"] == 19750 and s["n_features"] == 150
    assert s["n_pos"] + s["n_neg"] == 250
    assert s["positive_rate"] == pytest.approx(s["n_pos"] / 250)


@pytest.mark.synthetic_data
def test_data_fingerprint_keys():
    fp = data_fingerprint("synthetic")
    assert set(fp) == {"train.csv", "test.csv"}
    assert all(len(v) == 16 and int(v, 16) >= 0 for v in fp.values())
    assert fp == data_fingerprint("synthetic")
    assert fp["train.csv"] == data.sha256(paths.DATA_SYNTHETIC / "train.csv")[:16]


# ---------------------------------------------------------------- real data (read-only)
@pytest.mark.raw_data
def test_verify_raw_is_clean():
    assert verify("raw") == []


@pytest.mark.raw_data
def test_raw_fingerprint_keys():
    assert set(data_fingerprint("raw")) == {"train.csv", "test.csv"}


# ---------------------------------------------------------------- source resolution
def test_source_resolution_follows_env(monkeypatch):
    assert paths.resolve_source() == "synthetic"          # set by the autouse fixture
    assert paths.data_dir() == paths.DATA_SYNTHETIC
    monkeypatch.setenv(paths.DATA_SOURCE_ENV, "raw")
    assert paths.data_dir() == paths.DATA_RAW
    monkeypatch.delenv(paths.DATA_SOURCE_ENV)
    assert paths.resolve_source() == "raw"                 # default
    with pytest.raises(ValueError, match="unknown data source"):
        paths.resolve_source("kaggle")


# ---------------------------------------------------------------- verify() on corrupted copies
def test_verify_detects_corruption(tmp_path, monkeypatch):
    fake = tmp_path / "fake_synth"
    frames = _write_fake_dir(fake)
    monkeypatch.setattr(paths, "DATA_SYNTHETIC", fake)
    assert verify("synthetic") == []  # sizes are only enforced for raw

    train = frames["train"].copy()
    train.loc[3, "5"] = np.nan
    train.loc[4, "id"] = train.loc[5, "id"]
    train.loc[0, "target"] = 2
    train.to_csv(fake / "train.csv", index=False)
    frames["sample_submission"].iloc[::-1].to_csv(fake / "sample_submission.csv", index=False)
    problems = " | ".join(verify("synthetic"))
    for msg in ("NaN values present", "duplicate ids", "target has values other than 0/1",
                "sample_submission ids differ"):
        assert msg in problems, problems


def test_verify_raw_layout_checks_on_fake_raw(tmp_path, monkeypatch):
    fake = tmp_path / "fake_raw"
    _write_fake_dir(fake)
    monkeypatch.setattr(paths, "DATA_RAW", fake)
    problems = " | ".join(verify("raw"))
    assert "SHA256SUMS.txt missing" in problems
    assert "test.csv: shape" in problems  # 50 test rows instead of 19,750
    # a checksum file that no longer matches is reported as modification
    (fake / "SHA256SUMS.txt").write_text("0" * 64 + "  train.csv\n", encoding="utf-8")
    assert any("checksum mismatch: train.csv" in p for p in verify("raw"))


def test_verify_none_applies_raw_checks_when_env_is_raw(tmp_path, monkeypatch):
    """verify(None) resolves the source from SSC_DATA before applying the raw-only checks."""
    fake = tmp_path / "fake_raw"
    _write_fake_dir(fake)
    monkeypatch.setattr(paths, "DATA_RAW", fake)
    monkeypatch.setenv(paths.DATA_SOURCE_ENV, "raw")
    assert verify(None) == verify("raw")


def test_load_missing_files_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_SYNTHETIC", tmp_path / "nothing_here")
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load("synthetic")
