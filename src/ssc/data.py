"""Load and verify the competition data.

Kaggle layout (data/raw/):
    train.csv              250 rows:    id, target, 0..149
    test.csv               19,750 rows: id, 0..149
    sample_submission.csv  id, target
    SHA256SUMS.txt         checksums recorded at download time (2026-10-22)

data/synthetic/ has the same layout plus test_labels.csv (hidden labels and the
public/private split), produced by ssc.synthetic for pipeline tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

N_FEATURES = 150
FEATURES = [str(i) for i in range(N_FEATURES)]
ID_COL = "id"
TARGET_COL = "target"
EXPECTED_SHAPES = {"train.csv": (250, N_FEATURES + 2), "test.csv": (19750, N_FEATURES + 1)}


@dataclass
class Dataset:
    X: pd.DataFrame          # train features, columns "0".."149"
    y: np.ndarray            # int 0/1
    ids: np.ndarray          # train ids
    X_test: pd.DataFrame     # test features, same columns as X
    ids_test: np.ndarray     # test ids, in sample_submission order
    source: str              # "raw" or "synthetic"
    data_dir: Path

    @property
    def n_train(self) -> int:
        return len(self.y)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. For the Kaggle data see docs/SETUP.md; "
            "for mock data run: .venv\\Scripts\\python.exe scripts\\make_synthetic.py"
        )
    df = pd.read_csv(path)
    df.columns = [str(c) for c in df.columns]
    return df


def load(source: str | None = None) -> Dataset:
    """Load train + test from data/raw (default) or data/synthetic."""
    d = paths.data_dir(source)
    train = _read_csv(d / "train.csv")
    test = _read_csv(d / "test.csv")
    missing = [c for c in FEATURES if c not in train.columns or c not in test.columns]
    if missing:
        raise ValueError(f"feature columns missing: {missing[:5]}...")
    return Dataset(
        X=train[FEATURES].astype(float),
        y=train[TARGET_COL].to_numpy().astype(int),
        ids=train[ID_COL].to_numpy(),
        X_test=test[FEATURES].astype(float),
        ids_test=test[ID_COL].to_numpy(),
        source=paths.resolve_source(source),
        data_dir=d,
    )


def load_sample_submission(source: str | None = None) -> pd.DataFrame:
    return _read_csv(paths.data_dir(source) / "sample_submission.csv")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def data_fingerprint(source: str | None = None) -> dict[str, str]:
    """Short SHA256 prefixes of train/test, stored with every experiment."""
    d = paths.data_dir(source)
    return {name: sha256(d / name)[:16] for name in ("train.csv", "test.csv") if (d / name).exists()}


def verify(source: str | None = "raw") -> list[str]:
    """Run integrity checks. Returns a list of problems (empty list = all good).

    A missing *.zip listed in SHA256SUMS.txt is not a problem: it is the original Kaggle
    download, and someone who downloaded only the three CSVs can still verify them."""
    source = paths.resolve_source(source)
    d = paths.data_dir(source)
    problems: list[str] = []

    sums = d / "SHA256SUMS.txt"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            f = d / name.strip()
            if not f.exists():
                if not name.strip().lower().endswith(".zip"):
                    problems.append(f"missing file listed in SHA256SUMS: {name}")
            elif sha256(f) != digest:
                problems.append(f"checksum mismatch: {name} (file was modified!)")
    elif source == "raw":
        problems.append("SHA256SUMS.txt missing in data/raw")

    train = _read_csv(d / "train.csv")
    test = _read_csv(d / "test.csv")
    sample = _read_csv(d / "sample_submission.csv")

    if source == "raw":
        for name, df in (("train.csv", train), ("test.csv", test)):
            if df.shape != EXPECTED_SHAPES[name]:
                problems.append(f"{name}: shape {df.shape} != expected {EXPECTED_SHAPES[name]}")
    if list(train.columns[:2]) != [ID_COL, TARGET_COL]:
        problems.append(f"train.csv: first columns {list(train.columns[:2])}, expected ['id', 'target']")
    if list(test.columns) != [ID_COL] + FEATURES:
        problems.append("test.csv: columns are not ['id', '0'..'149']")
    if train.isna().any().any() or test.isna().any().any():
        problems.append("NaN values present")
    if not set(np.unique(train[TARGET_COL])) <= {0, 1}:
        problems.append("target has values other than 0/1")
    if train[ID_COL].duplicated().any() or test[ID_COL].duplicated().any():
        problems.append("duplicate ids")
    if set(train[ID_COL]) & set(test[ID_COL]):
        problems.append("train and test ids overlap")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        problems.append(f"sample_submission columns {list(sample.columns)} != ['id', 'target']")
    if not np.array_equal(sample[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        problems.append("sample_submission ids differ from test ids (order matters)")
    return problems


def summary(ds: Dataset) -> dict:
    """Small descriptive summary for logs and the report's problem formulation."""
    X = ds.X.to_numpy()
    Xt = ds.X_test.to_numpy()
    return {
        "source": ds.source,
        "n_train": int(len(ds.y)),
        "n_test": int(len(ds.ids_test)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(ds.y.mean()),
        "n_pos": int(ds.y.sum()),
        "n_neg": int(len(ds.y) - ds.y.sum()),
        "train_feature_std_range": [float(X.std(0).min()), float(X.std(0).max())],
        "test_feature_std_range": [float(Xt.std(0).min()), float(Xt.std(0).max())],
        "train_feature_mean_abs_max": float(np.abs(X.mean(0)).max()),
    }
