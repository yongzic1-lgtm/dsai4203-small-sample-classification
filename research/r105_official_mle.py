"""R105 - official-CV runs of the unregularised-MLE refinement of S003 (follows R104).

R104: S003's reg_covar 1e-6 is ~1 % of the smallest component eigenvalue; the ranking of the
boundary rows converges for reg <= 1e-8 in the PCA basis (Kendall 0.996 vs S003). Candidates:
  gmm2_block_mle         K=2 full GMM, block PCA basis, reg 1e-9, EM polished (the MLE)
  semisup_eps01_mle      semi-supervised EM (soft label constraints eps 0.01) at reg 1e-9
  gmm2_block_mle_seed10  MLE with 10 restarts averaged (n_init=5 each, seeds 0..9, polished)

    .venv\Scripts\python.exe colab\remote.py run -a pro1 -s rmix --high-mem --no-sync -- research/r105_official_mle.py --n-jobs 8
"""

from __future__ import annotations

import argparse

import pandas as pd

import lib_mix as L
from ssc import paths

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=1)
p.add_argument("--only", nargs="*", default=None)
a = p.parse_args()
src = a.data
B = L.BLOCK
CANDS = {
    "gmm2_block_mle": (lambda: L.MixGMM(B, 2, reg=1e-9, polish=True, source=src),
                       "K=2 full GMM, block PCA basis, reg_covar 1e-9 (unregularised MLE), EM polished",
                       "Removes the ~1 % reg bias of S003 in the smallest-variance direction: boundary Kendall vs S003 "
                       "~0.996, same expected AUC to 1e-6; replicas (R103) decide whether it is better."),
    "semisup_eps01_mle": (lambda: L.SemiSupEM(B, 1, reg=1e-9, eps=0.01, source=src),
                          "semi-supervised EM (eps 0.01) at reg 1e-9",
                          "MLE + 250 soft label constraints; indistinguishable from gmm2_block_mle (Kendall > 0.999)."),
    "gmm2_block_mle_seed10": (lambda: L.MixGMM(B, 2, reg=1e-9, polish=True, seeds=tuple(range(10)), source=src),
                              "MLE, 10 restart seeds averaged (each n_init=5, polished)",
                              "All restarts reach one optimum (R101), so identical to gmm2_block_mle."),
}
rows = []
for name in a.only or list(CANDS):
    make, desc, hyp = CANDS[name]
    r = L.official_run("r105", name, make, desc, hyp, src, a.author, a.n_jobs)
    rows.append(r)
    print({k: r[k] for k in ("candidate", "run_id", "auc_mean", "cv_logloss", "kendall_boundary", "n_boundary_self",
                             "expected_auc_under_ref", "n_side_flips_all")}, flush=True)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(tables / "r105_candidates.csv", index=False)
print(pd.DataFrame(rows)[["candidate", "run_id", "auc_mean", "cv_logloss", "cv_brier", "kendall_boundary",
                          "spearman_boundary", "expected_auc_under_ref"]].to_string(index=False))
