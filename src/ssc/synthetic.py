"""Mock data with the same layout as the Kaggle data, plus the hidden test labels.

Used to (1) test every script end-to-end without touching the real data and
(2) simulate how noisy a 2% public leaderboard is, because here we know the
"private" labels. Nothing learned from synthetic data is evidence about the
real problem; it only checks that the code does what it claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import paths
from .data import FEATURES, N_FEATURES

N_TRAIN = 250
N_TEST = 19750
PUBLIC_FRACTION = 0.02


def make(
    n_informative: int = 10,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    signal: float = 1.0,
    feature_scale: float = 4.0,
    seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Linear-logistic ground truth on a few features; the rest is pure noise."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    informative = rng.choice(N_FEATURES, size=n_informative, replace=False)
    w = np.zeros(N_FEATURES)
    w[informative] = rng.choice([-1.0, 1.0], size=n_informative) * rng.uniform(0.5, 1.5, size=n_informative)
    logit = signal * X @ w
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = X * feature_scale

    ids = np.arange(n)
    cols = {c: X[:, i] for i, c in enumerate(FEATURES)}
    train = pd.DataFrame({"id": ids[:n_train], "target": y[:n_train], **{c: v[:n_train] for c, v in cols.items()}})
    test = pd.DataFrame({"id": ids[n_train:], **{c: v[n_train:] for c, v in cols.items()}})
    sample = pd.DataFrame({"id": ids[n_train:], "target": 0.5})

    public = np.zeros(n_test, dtype=bool)
    public[rng.choice(n_test, size=int(round(PUBLIC_FRACTION * n_test)), replace=False)] = True
    labels = pd.DataFrame({"id": ids[n_train:], "target": y[n_train:], "public": public})
    truth = pd.DataFrame({"feature": [str(i) for i in informative], "weight": w[informative]})
    return {"train": train, "test": test, "sample_submission": sample, "test_labels": labels, "truth": truth}


def write(seed: int = 2026, out: Path | str | None = None, **kwargs) -> Path:
    """Write make(seed, **kwargs) as CSVs to `out` (default data/synthetic). Returns the folder."""
    out = Path(out) if out is not None else paths.DATA_SYNTHETIC
    out.mkdir(parents=True, exist_ok=True)
    frames = make(seed=seed, **kwargs)
    for name, df in frames.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return out


def score_submission(sub: pd.DataFrame, labels_path: Path | str | None = None) -> dict[str, float]:
    """Public/private AUC of a submission against the synthetic hidden labels
    (default data/synthetic/test_labels.csv)."""
    labels = pd.read_csv(Path(labels_path) if labels_path is not None else paths.DATA_SYNTHETIC / "test_labels.csv")
    merged = labels.merge(sub, on="id", suffixes=("", "_pred"), validate="one_to_one")
    pub = merged["public"].to_numpy(bool)
    y, p = merged["target"].to_numpy(), merged["target_pred"].to_numpy()
    return {
        "public_auc": float(roc_auc_score(y[pub], p[pub])),
        "private_auc": float(roc_auc_score(y[~pub], p[~pub])),
        "n_public": int(pub.sum()),
    }
