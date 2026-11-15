"""R104 - is reg_covar = 1e-6 (S003) already at the unregularised MLE? (label-free)

R101 showed that reg_covar 1e-4 costs +1,724 BIC and re-orders the boundary rows (Kendall 0.77
vs S003). The block's within-component covariances have very small eigenvalues (the 13-dim
block has two pooled eigenvalues of 0.018 and 0.005 and one exact zero), so reg 1e-6 might still
bias the smallest directions. Sweep reg in three bases:
  pca     12-dim non-degenerate PCA basis of the standardised block (unwhitened)
  white   same, whitened (unit pooled variance per PC: reg is relative)
  std13   the 13 standardised block columns (S003's own basis, contains the exact null direction)
For every fit: mean log-likelihood / BIC (comparable within a basis), smallest eigenvalue of each
component covariance, Kendall tau on the S003 boundary rows vs S003, and the expected test AUC
under the reference posterior. Labels only orient components.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rmix --high-mem --no-sync -- research/r104_reg_sweep.py --n-jobs 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

import lib_mix as L
from ssc import data, paths
from ssc.tracking import Run
from ssc.transductive import unlabeled_features

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
a = p.parse_args()
src = a.data
ds = data.load(src)
y = ds.y
X_all = unlabeled_features(src)
n = len(X_all)
s3 = L.s003_pred(src)
b3 = (s3 > 0.1) & (s3 < 0.9)
ref_post, _, _ = L.polished_posterior(L.BLOCK, 2, "full", 5, 0, 1e-6, src)
ref_test = L.oriented(ref_post, y)[250:]

Zb, _, _ = L.standardize(X_all[:, list(L.BLOCK)])
W, w = L.pca_basis(Zb)
BASES = {"pca": (Zb - Zb.mean(0)) @ W, "white": ((Zb - Zb.mean(0)) @ W) / np.sqrt(w[: W.shape[1]]), "std13": Zb}
REGS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12]


def one(bname, reg):
    U = BASES[bname]
    try:
        f = L.gmm_fit_arr(U, 2, "full", 5, 0, reg, max_iter=3000)
    except np.linalg.LinAlgError as e:
        return {"basis": bname, "reg": reg, "error": str(e)[:80]}
    r = L.oriented(f["post"], y)
    t = r[250:]
    mineig = [float(np.linalg.eigvalsh(c).min()) for c in f["covs"]]
    d = U.shape[1]
    return {"basis": bname, "reg": reg, "dim": d, "mean_loglik": f["loglik"],
            "bic": -2 * f["loglik"] * n + L.n_params(2, d, "full") * np.log(n), "n_iter": f["n_iter"],
            "min_eig_comp0": mineig[0], "min_eig_comp1": mineig[1],
            "kendall_s003_boundary": float(kendalltau(t[b3], s3[b3])[0]),
            "n_boundary": int(((t > 0.1) & (t < 0.9)).sum()),
            "expected_auc_under_ref": L.expected_auc(t, ref_test), "post_test": t}


hyp = ("In the PCA/whitened bases the log-likelihood and the boundary ranking stop changing below reg 1e-6 "
       "(Kendall vs S003 > 0.99 for reg <= 1e-6), i.e. S003's reg is already effectively the MLE.")
cfg = {"name": "r104_reg_sweep", "bases": list(BASES), "regs": REGS}
with Run("r104_reg_sweep", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(b, r) for b in BASES for r in REGS)
    posts = {(r["basis"], r["reg"]): r.pop("post_test") for r in res if "post_test" in r}
    df = pd.DataFrame(res)
    # Kendall of each fit vs the smallest-reg fit of the same basis (convergence of the ranking)
    kk = []
    for r in res:
        if "error" in r:
            kk.append(np.nan)
            continue
        regs_ok = [q for q in REGS if (r["basis"], q) in posts]
        base = posts[(r["basis"], min(regs_ok))]
        kk.append(float(kendalltau(posts[(r["basis"], r["reg"])][b3], base[b3])[0]))
    df["kendall_vs_smallest_reg_same_basis"] = kk
    run.save_table("reg_sweep", df)
    tables = paths.tables_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "r104_reg_sweep.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    best = df.dropna(subset=["bic"]).sort_values("bic").groupby("basis").head(1)
    out = {"best_reg_by_bic_per_basis": best[["basis", "reg", "bic", "kendall_s003_boundary"]].to_dict("records")}
    print(out)
    run.finish(extra_summary=out)
