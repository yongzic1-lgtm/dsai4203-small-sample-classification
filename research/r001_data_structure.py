"""R001 - structure of the feature matrix (no model fitting on labels except |corr|).

Uses train + test features together (20,000 x 150, unlabeled structure is
competition data) to look for: high-variance vs unit-variance features,
linear redundancy (near-zero covariance eigenvalues), feature clusters,
multi-modality / mixture structure, and train-vs-test distribution shift.
Outputs go to the run folder and reports/tables|figures with prefix r001_.

    .venv\\Scripts\\python.exe colab\\remote.py run -- research/r001_data_structure.py
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from ssc import data, paths  # noqa: E402
from ssc.features import abs_correlation  # noqa: E402
from ssc.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
a = p.parse_args()

ds = data.load(a.data)
X_tr, X_te = ds.X.to_numpy(), ds.X_test.to_numpy()
X_all = np.vstack([X_tr, X_te])
tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)

cfg = {"name": "r001_data_structure", "kind": "analysis", "inputs": "train+test features, train labels for |corr| only"}
with Run("r001_data_structure", cfg, kind="analysis", source=a.data, author=a.author,
         hypothesis="Few features carry signal; informative ones may differ in variance/shape from noise and may be "
                    "linearly redundant (make_classification-like generator).") as run:
    # ---- per-feature univariate description
    std = X_all.std(0)
    skew = stats.skew(X_all, axis=0)
    kurt = stats.kurtosis(X_all, axis=0)            # excess kurtosis (0 for a Gaussian)
    bimod = (skew**2 + 1) / (kurt + 3)              # Sarle's bimodality coefficient (>0.555 suggests multimodal)
    ks = np.array([stats.ks_2samp(X_tr[:, j], X_te[:, j]).pvalue for j in range(X_all.shape[1])])
    corr = abs_correlation(X_tr, ds.y)
    feat = pd.DataFrame({"feature": data.FEATURES, "mean": X_all.mean(0), "std": std, "skew": skew,
                         "excess_kurtosis": kurt, "bimodality": bimod, "ks_train_test_p": ks,
                         "abs_corr_target_train": corr}).sort_values("std", ascending=False)
    run.save_table("feature_stats", feat)
    feat.to_csv(tables / "r001_feature_stats.csv", index=False)

    # ---- linear redundancy: eigenvalues of the correlation matrix over 20k rows
    C = np.corrcoef(X_all, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    n_tiny = int((eig < 1e-6).sum())
    off = np.abs(C - np.eye(len(C)))
    strong_pairs = [(data.FEATURES[i], data.FEATURES[j], float(C[i, j]))
                    for i in range(len(C)) for j in range(i + 1, len(C)) if off[i, j] > 0.3]
    pd.DataFrame(strong_pairs, columns=["f1", "f2", "corr"]).to_csv(tables / "r001_strong_feature_pairs.csv", index=False)
    pd.DataFrame({"rank": np.arange(1, len(eig) + 1), "eigenvalue": eig}).to_csv(tables / "r001_corr_eigenvalues.csv", index=False)

    # ---- mixture structure on the high-variance block (if any) else on all features
    hi = std > np.median(std) * 1.5
    block = X_all[:, hi] if hi.sum() >= 2 else X_all
    Z = (block - block.mean(0)) / block.std(0)
    bic = []
    for k in (1, 2, 4, 8, 16):
        gm = GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(Z)
        bic.append({"n_components": k, "bic": gm.bic(Z)})
    bic = pd.DataFrame(bic)
    bic.to_csv(tables / "r001_gmm_bic.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(range(len(std)), np.sort(std)[::-1]); ax[0].set_title("feature std (desc), train+test")
    ax[1].scatter(std, corr, s=10); ax[1].set_xlabel("std"); ax[1].set_ylabel("|corr| with target (train)")
    ax[1].set_title("variance vs target correlation")
    ax[2].semilogy(eig, marker="."); ax[2].set_title(f"corr-matrix eigenvalues ({n_tiny} < 1e-6)")
    fig.tight_layout()
    fig.savefig(figs / "r001_structure.png", dpi=150)
    run.save_figure("structure", fig)

    summary = {
        "n_rows_all": int(X_all.shape[0]),
        "std_min": float(std.min()), "std_median": float(np.median(std)), "std_max": float(std.max()),
        "n_high_variance": int(hi.sum()), "high_variance_features": [data.FEATURES[j] for j in np.where(hi)[0]],
        "n_tiny_eigenvalues": n_tiny, "n_strong_pairs_abs_corr_gt_0.3": len(strong_pairs),
        "n_bimodal_gt_0.555": int((bimod > 0.555).sum()),
        "n_ks_shift_p_lt_0.01": int((ks < 0.01).sum()),
        "top10_corr_features": feat.sort_values("abs_corr_target_train", ascending=False)["feature"].head(10).tolist(),
        "gmm_best_k_by_bic": int(bic.loc[bic["bic"].idxmin(), "n_components"]),
        "positive_rate": float(ds.y.mean()),
    }
    run.finish(extra_summary=summary)
    for k, v in summary.items():
        print(f"{k}: {v}")
