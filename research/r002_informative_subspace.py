"""R002 - is the signal confined to a low-dimensional subspace of the high-variance features?

Follows R001 (13 high-variance features, 8 near-zero correlation eigenvalues). Label-free
steps use train+test features (20,000 rows); labels are used only for descriptive
|corr| / cluster purity and are reported as such.

Checks
  1. rank of the 13-feature block (eigenvalues) and R^2 of each feature regressed on the others
  2. the 137 other features: pairwise correlations, correlation with the block, normality
  3. PCA basis of the block -> d informative dimensions
  4. Gaussian-mixture structure in that basis (BIC over K) and label purity per component

    .venv\\Scripts\\python.exe colab\\remote.py run -- research/r002_informative_subspace.py
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
p.add_argument("--ratio", type=float, default=1.5, help="high-variance rule: std > ratio * median std")
a = p.parse_args()

ds = data.load(a.data)
X_tr, X_te = ds.X.to_numpy(), ds.X_test.to_numpy()
X_all = np.vstack([X_tr, X_te])
n_tr = len(X_tr)
tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)

std = X_all.std(0)
H = np.where(std > a.ratio * np.median(std))[0]
N = np.setdiff1d(np.arange(X_all.shape[1]), H)

cfg = {"name": "r002_informative_subspace", "high_variance_rule": f"std > {a.ratio} * median(std) on train+test",
       "high_variance_features": [data.FEATURES[j] for j in H]}
with Run("r002_informative_subspace", cfg, kind="analysis", source=a.data, author=a.author,
         hypothesis="The high-variance block spans a ~5-dim subspace (informative + redundant linear combos); "
                    "the remaining features are independent Gaussian noise; classes form Gaussian clusters there.") as run:
    out = {"high_variance_features": [data.FEATURES[j] for j in H]}

    # 1. rank / redundancy of the block
    B = X_all[:, H]
    Bc = (B - B.mean(0)) / B.std(0)
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(Bc, rowvar=False)))[::-1]
    d = int((eig > 1e-6 * eig[0]).sum())
    r2 = []
    for i in range(len(H)):
        others = np.delete(Bc, i, axis=1)
        coef, *_ = np.linalg.lstsq(np.c_[others, np.ones(len(Bc))], Bc[:, i], rcond=None)
        resid = Bc[:, i] - np.c_[others, np.ones(len(Bc))] @ coef
        r2.append(1 - resid.var() / Bc[:, i].var())
    block = pd.DataFrame({"feature": [data.FEATURES[j] for j in H], "std": std[H], "r2_on_other_block_features": r2,
                          "abs_corr_target_train": abs_correlation(X_tr[:, H], ds.y)})
    run.save_table("block_redundancy", block)
    block.to_csv(tables / "r002_block_redundancy.csv", index=False)
    out.update({"block_size": int(len(H)), "block_rank_d": d, "block_eigenvalues": [round(float(e), 6) for e in eig]})

    # 2. the other features
    Cn = np.corrcoef(X_all[:, N], rowvar=False)
    off = np.abs(Cn[np.triu_indices(len(N), 1)])
    cross = np.abs(np.corrcoef(np.c_[X_all[:, N], B], rowvar=False)[: len(N), len(N):])
    normal_p = np.array([stats.normaltest(X_all[:, j]).pvalue for j in N])
    out.update({
        "noise_max_abs_pairwise_corr": float(off.max()), "noise_mean_abs_pairwise_corr": float(off.mean()),
        "noise_max_abs_corr_with_block": float(cross.max()),
        "noise_n_nonnormal_p_lt_0.001": int((normal_p < 1e-3).sum()), "noise_n": int(len(N)),
        "noise_std_range": [float(std[N].min()), float(std[N].max())],
        "noise_max_abs_corr_target_train": float(abs_correlation(X_tr[:, N], ds.y).max()),
        "expected_max_abs_corr_under_null_approx": float(np.sqrt(2 * np.log(len(N)) / n_tr)),
    })

    # 3. PCA basis of the block
    U, S, Vt = np.linalg.svd(Bc - Bc.mean(0), full_matrices=False)
    Z = (Bc - Bc.mean(0)) @ Vt[:d].T
    Z = Z / Z.std(0)
    zc = abs_correlation(Z[:n_tr], ds.y)
    out["pc_abs_corr_target_train"] = [round(float(c), 3) for c in zc]
    np.save(run.dir / "Z_all.npy", Z)

    # 4. mixture structure in Z
    rows = []
    for k in range(1, 13):
        gm = GaussianMixture(k, covariance_type="full", random_state=0, n_init=3, max_iter=500).fit(Z)
        rows.append({"K": k, "bic": gm.bic(Z), "aic": gm.aic(Z)})
    bic = pd.DataFrame(rows)
    bic.to_csv(tables / "r002_gmm_bic_subspace.csv", index=False)
    run.save_table("gmm_bic_subspace", bic)
    k_best = int(bic.loc[bic["bic"].idxmin(), "K"])
    gm = GaussianMixture(k_best, covariance_type="full", random_state=0, n_init=5, max_iter=1000).fit(Z)
    lab = gm.predict(Z[:n_tr])
    purity = pd.DataFrame([{"component": c, "weight_all": float(gm.weights_[c]), "n_train": int((lab == c).sum()),
                            "train_pos_rate": float(ds.y[lab == c].mean()) if (lab == c).any() else np.nan}
                           for c in range(k_best)])
    purity.to_csv(tables / "r002_component_purity.csv", index=False)
    run.save_table("component_purity", purity)
    out.update({"gmm_best_K": k_best, "component_train_pos_rates": [round(float(v), 3) for v in purity["train_pos_rate"]]})

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].semilogy(eig, marker="o"); ax[0].set_title(f"block eigenvalues (rank d={d})")
    ax[1].plot(bic["K"], bic["bic"], marker="o"); ax[1].set_title("GMM BIC in PCA basis (20k rows)"); ax[1].set_xlabel("K")
    o = np.argsort(-zc)[:2]
    sc = ax[2].scatter(Z[n_tr:, o[0]], Z[n_tr:, o[1]], s=1, c="lightgrey")
    ax[2].scatter(Z[:n_tr, o[0]], Z[:n_tr, o[1]], s=12, c=ds.y, cmap="coolwarm", edgecolor="k", linewidth=0.2)
    ax[2].set_title(f"PC{o[0]+1} vs PC{o[1]+1} (grey=test, colour=train label)")
    fig.tight_layout()
    fig.savefig(figs / "r002_subspace.png", dpi=150)
    run.save_figure("subspace", fig)

    run.finish(extra_summary=out)
    for k, v in out.items():
        print(f"{k}: {v}")
