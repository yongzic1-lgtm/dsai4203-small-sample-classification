"""R943 (verifier "ext") - how sharp is E4's "sharp drop" compared with pure noise, and how do its
drop rules behave under permuted labels? (+ one E3 count)

Reproduces the bootstrap selection frequencies of E4 run 20261022-071414_e4_feature_stability with
E4's own functions and seeds (B = 2000, seed 2026; permutations p use default_rng(7 + p) and bootstrap
seed 2027 + p), then
1. checks that m_gap / m_half / m_binom / m_perm and the largest step reproduce e4_drop.csv;
2. gives the null distribution (50 permuted-label curves) of the largest step between consecutive
   sorted frequencies and of its position, i.e. what the "gap" rule finds when there is no signal;
3. applies the perm rule to each null curve against the envelope of the other 49 (leave-one-out)
   -> how many features the perm rule admits under pure noise (false-positive behaviour);
4. E3: counts features with |corr| > 1/sqrt(n) on the real labels and under permutations
   (E3's draft says the accumulated optimism peaks at ~48 features).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vext --high-mem -- research/r943_e4_null_step.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from ssc import data, paths
from ssc.features import abs_correlation
from ssc.tracking import Run

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--n-perm", type=int, default=50)
a = p.parse_args()

spec = importlib.util.spec_from_file_location("e4_run_r943", ROOT / "extensions/e4_feature_stability/run.py")
E4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E4)

KS = [5, 10, 13, 20]
P = 150
HYP = ("The gap rule always returns a largest step; under permuted labels the top of the sorted curve is also steep "
       "(E4: the top noise feature reaches 0.73-0.93 at its 95% quantile), so the null largest step is expected to be "
       "0.2-0.4 for k = 10-20, i.e. the observed 0.30-0.49 is not far above noise for k = 20. The perm rule should "
       "admit 0 features under the null in about 95% of leave-one-out checks. E3: more features than ~48 exceed "
       "|corr| = 1/sqrt(250) on the real labels because 20 features carry signal.")

with Run("r943_e4_null_step", {"n_boot": a.n_boot, "n_perm": a.n_perm, "ks": KS, "seeds": "E4: bootstrap 2026, perm label 7+p, boot 2027+p",
                               "model_code": "research/r943_e4_null_step.py"},
         kind="analysis", source="raw", hypothesis=HYP) as run:
    ds = data.load("raw")
    X, y = ds.X.to_numpy(float), ds.y
    R = E4.ranks_desc(E4.boot_abs_corr(X, y, a.n_boot, 2026, "pearson"))
    freqs = {k: (R <= k).mean(axis=0) for k in KS}
    null_sorted = {k: np.empty((a.n_perm, P)) for k in KS}
    n_above_null = []
    for pp in range(a.n_perm):
        y_p = np.random.default_rng(7 + pp).permutation(y)
        R_p = E4.ranks_desc(E4.boot_abs_corr(X, y_p, a.n_boot, 2026 + 1 + pp, "pearson"))
        for k in KS:
            null_sorted[k][pp] = np.sort((R_p <= k).mean(axis=0))[::-1]
        n_above_null.append(int((abs_correlation(X, y_p, "pearson") > 1 / np.sqrt(len(y))).sum()))
    dcfg = {"threshold": 0.5, "binom_alpha": 0.05, "null_quantile": 0.95, "primary_rule": "perm"}
    rows, nrows = [], []
    for k in KS:
        row, _ = E4.drop_rules(freqs[k], k, a.n_boot, null_sorted[k], dcfg)
        steps_null = null_sorted[k][:, :-1] - null_sorted[k][:, 1:]
        mx, pos = steps_null.max(1), steps_null.argmax(1) + 1
        loo_perm, half_null = [], []
        for pp in range(a.n_perm):
            others = np.delete(null_sorted[k], pp, axis=0)
            q = np.quantile(others, 0.95, axis=0)
            above = null_sorted[k][pp] > q
            loo_perm.append(P if above.all() else int(np.argmin(above)))
            half_null.append(int((null_sorted[k][pp] >= 0.5).sum()))
            nrows.append({"k": k, "perm": pp, "max_step": float(mx[pp]), "pos_max_step": int(pos[pp]),
                          "top_freq": float(null_sorted[k][pp][0]), "m_perm_loo": loo_perm[-1], "m_half": half_null[-1]})
        bound = float(binom.ppf(1 - 0.05 / P, a.n_boot, k / P) / a.n_boot)
        rows.append({**row, "obs_max_step": row["gap_size"],
                     "null_max_step_mean": float(mx.mean()), "null_max_step_q95": float(np.quantile(mx, 0.95)),
                     "null_max_step_max": float(mx.max()), "frac_null_step_ge_obs": float((mx >= row["gap_size"]).mean()),
                     "null_pos_max_step_median": float(np.median(pos)),
                     "null_m_half_mean": float(np.mean(half_null)), "null_frac_m_half_ge1": float(np.mean(np.array(half_null) >= 1)),
                     "null_m_perm_loo_mean": float(np.mean(loo_perm)), "null_frac_m_perm_loo_ge1": float(np.mean(np.array(loo_perm) >= 1)),
                     "null_m_binom_mean": float(np.mean((null_sorted[k] > bound).sum(1))),
                     "null_top_freq_q95": float(np.quantile(null_sorted[k][:, 0], 0.95))})
    tab = pd.DataFrame(rows)
    ntab = pd.DataFrame(nrows)
    run.save_table("drop_vs_null", tab)
    run.save_table("null_curves_summary", ntab)
    ref = pd.read_csv(paths.tables_dir("raw") / "e4_drop.csv") if (paths.tables_dir("raw") / "e4_drop.csv").exists() else None
    repro = None
    if ref is not None:
        cols = ["m_gap", "gap_size", "m_half", "m_binom", "m_perm", "null_q_top1"]
        repro = bool(np.allclose(ref[cols].to_numpy(float), tab[cols].to_numpy(float), atol=1e-9))
    paths.tables_dir("raw").mkdir(parents=True, exist_ok=True)
    tab.to_csv(paths.tables_dir("raw") / "r943_e4_drop_vs_null.csv", index=False)
    r = abs_correlation(X, y, "pearson")
    e3 = {"n_abs_corr_gt_1_over_sqrt_n_real": int((r > 1 / np.sqrt(len(y))).sum()),
          "n_abs_corr_gt_1_over_sqrt_n_perm_mean": float(np.mean(n_above_null)),
          "expected_null_150_x_P(|Z|>1)": float(150 * 2 * (1 - __import__("scipy.stats", fromlist=["norm"]).norm.cdf(1)))}
    run.finish(extra_summary={"drop_vs_null": tab.to_dict(orient="records"), "reproduces_e4_drop_csv": repro, "e3_counts": e3})
    pd.set_option("display.width", 250)
    print(tab.round(4).T.to_string())
    print("reproduces e4_drop.csv:", repro)
    print(e3)
    print("run:", run.run_id)
