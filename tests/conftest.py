"""Shared pytest fixtures for the ssc test suite.

Safety rules enforced here for EVERY test (autouse fixture):
  * SSC_ARCHIVE_ROOT -> a fresh pytest tmp_path, so no test can write into the
    real archive (experiments/, registry/, submissions/, reports/) or sandbox/;
  * SSC_DATA = "synthetic", so code that resolves the source from the
    environment never reads the Kaggle data by accident;
  * SSC_AUTHOR = "pytest", so author fields are deterministic.

Tests that need the on-disk mock data are marked `synthetic_data` (or use the
`synthetic_ds` fixture) and are SKIPPED when data/synthetic/ is missing; the
suite never regenerates it. Small in-memory datasets come from
ssc.synthetic.make().
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports pyplot

import numpy as np
import pytest

from ssc import paths, synthetic

SYNTHETIC_FILES = ("train.csv", "test.csv", "sample_submission.csv")
HAS_SYNTHETIC = all((paths.DATA_SYNTHETIC / f).exists() for f in SYNTHETIC_FILES)
HAS_RAW = all((paths.DATA_RAW / f).exists() for f in SYNTHETIC_FILES)
SKIP_SYNTHETIC_REASON = ("data/synthetic/train.csv missing; generate it once with "
                         "scripts/make_synthetic.py (the tests never regenerate it)")


def pytest_configure(config):
    config.addinivalue_line("markers", "synthetic_data: needs data/synthetic/*.csv (skipped if missing)")
    config.addinivalue_line("markers", "raw_data: needs data/raw/*.csv, read-only (skipped if missing)")


def pytest_collection_modifyitems(config, items):
    skip_syn = pytest.mark.skip(reason=SKIP_SYNTHETIC_REASON)
    skip_raw = pytest.mark.skip(reason="data/raw not present")
    for item in items:
        if not HAS_SYNTHETIC and "synthetic_data" in item.keywords:
            item.add_marker(skip_syn)
        if not HAS_RAW and "raw_data" in item.keywords:
            item.add_marker(skip_raw)


@pytest.fixture(autouse=True)
def isolated_archive(tmp_path, monkeypatch):
    """Redirect the whole archive to a per-test temp dir. Returns that root."""
    root = tmp_path / "archive"
    monkeypatch.setenv(paths.ARCHIVE_ROOT_ENV, str(root))
    monkeypatch.setenv(paths.DATA_SOURCE_ENV, "synthetic")
    monkeypatch.setenv("SSC_AUTHOR", "pytest")
    return root


@pytest.fixture(scope="session")
def synthetic_ds():
    """The on-disk synthetic Dataset (loaded once). Do not mutate it."""
    if not HAS_SYNTHETIC:
        pytest.skip(SKIP_SYNTHETIC_REASON)
    from ssc import data
    return data.load("synthetic")


def frames_to_xy(frames: dict):
    """(X, y, X_test) numpy arrays from ssc.synthetic.make() output."""
    train, test = frames["train"], frames["test"]
    X = train.drop(columns=["id", "target"]).to_numpy(dtype=float)
    y = train["target"].to_numpy().astype(int)
    X_test = test.drop(columns=["id"]).to_numpy(dtype=float)
    return X, y, X_test


@pytest.fixture(scope="session")
def small_xy():
    """Small in-memory problem with signal: 120 train rows, 40 test rows, 150 features."""
    return frames_to_xy(synthetic.make(n_train=120, n_test=40, seed=7))


@pytest.fixture(scope="session")
def y_250():
    """Unbalanced 0/1 labels (97 positives of 250) for split tests."""
    rng = np.random.default_rng(0)
    y = np.zeros(250, dtype=int)
    y[rng.choice(250, size=97, replace=False)] = 1
    return y
