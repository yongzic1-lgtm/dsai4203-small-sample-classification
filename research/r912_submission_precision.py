"""R912 - does the submission file format cost AUC? (verifier `repro`, 2026-10-22)

Observation on the real files (local check, research/r911_repro_compare.py): S003's test_pred.npy has only
7,503 distinct values out of 19,750 (float64 saturation of the GMM posterior: 7,907 rows share the minimum
score, 3,434 the maximum), and ssc.submission.create writes `float_format="%.8f"`, after which 2,897 distinct
values remain (8,918 rows tie at the minimum, 6,416 at the maximum; Spearman(csv, test_pred) = 0.969).
Ties count 1/2 in the AUC, so the file Kaggle scores is not the ordering the replica study evaluated.

On make_classification replicas (lib_gen_replica variant b: class_sep 0.71, hamming 4, the generator found by
R150-R154) with known test labels, private-subset AUC of four orderings of the SAME fitted S003 model:
  raw      s = R @ rate (TransductiveGMM score, float64), as stored in test_pred.npy
  csv8     s written with "%.8f" and read back (exactly what make_submission uploads)
  logodds  unsaturated GMM log-odds log w1 p1(x) - log w0 p0(x), oriented by the component rates
           (same ordering as `raw` wherever float64 does not saturate)
  bayes    true Bayes log-odds of the generator (upper bound)
for flip_y in {0, 0.005} (sklearn semantics: ~flip_y/2 labels change) and 30 replica seeds each.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vrepro -- research/r912_submission_precision.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

HYP = ("The %.8f rounding and the float64 saturation only tie rows whose GMM posterior is < 1e-8 from 0 or 1, "
       "i.e. rows far from the decision boundary. A wrongly ordered pair inside such a tie group needs a "
       "label that disagrees with an almost certain posterior: with flip_y = 0 essentially never (Bayes "
       "posterior < 1e-8), with label noise the flipped labels are independent of x, so ordered or tied they "
       "earn 1/2 in expectation. Prediction: mean private AUC(csv8) - AUC(raw) and AUC(logodds) - AUC(raw) "
       "are within +-2e-5 (|mean| < 2 x its SE) at both noise levels; refuted if either difference is "
       "systematically negative by more than 1e-4 (then make_submission should write ranks or %.17g).")


def csv8(s: np.ndarray) -> np.ndarray:
    """Exactly what pandas to_csv(float_format='%.8f') + read_csv returns."""
    return np.array([float("%.8f" % v) for v in s])


def one(seed: int, flip_y: float) -> dict:
    rep = G.sample_make_classification(seed, class_sep=0.71, flip_y=flip_y, hamming=4)
    g = G.UnsupervisedGMM(rep.X_all, "highvar", K=2).fit()
    n_tr = len(rep.y_train)
    rates = g.rates(np.arange(n_tr), rep.y_train)
    s = g.scores(np.arange(n_tr), rep.y_train)[n_tr:]
    Z = (rep.X_all[:, g.cols_] - g.mu_) / g.sd_
    if g.Vt_ is not None:
        Z = Z @ g.Vt_.T
    wlp = g.gmm_._estimate_weighted_log_prob(Z)[n_tr:]
    hi = int(np.argmax(rates))
    lo = 1 - hi
    logodds = wlp[:, hi] - wlp[:, lo]
    s8 = csv8(s)
    m = rep.private_mask
    y = rep.y_test[m]
    out = {"seed": seed, "flip_y": flip_y, "n_flipped": int(rep.params["n_flipped_true"]),
           "block_size": int(len(g.cols_)), "rate_lo": float(rates[lo]), "rate_hi": float(rates[hi]),
           "auc_raw": G.auc_fast(y, s[m]), "auc_csv8": G.auc_fast(y, s8[m]),
           "auc_logodds": G.auc_fast(y, logodds[m]), "auc_bayes": G.auc_fast(y, rep.bayes_test[m]),
           "n_unique_raw": int(len(np.unique(s))), "n_unique_csv8": int(len(np.unique(s8))),
           "n_at_min_raw": int((s == s.min()).sum()), "n_at_max_raw": int((s == s.max()).sum()),
           "n_at_min_csv8": int((s8 == s8.min()).sum()), "n_at_max_csv8": int((s8 == s8.max()).sum()),
           "spearman_csv8_raw": float(stats.spearmanr(s8, s).statistic)}
    for k in ("csv8", "logodds", "bayes"):
        out[f"d_{k}_minus_raw"] = out[f"auc_{k}"] - out["auc_raw"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="raw", choices=paths.SOURCES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.005])
    a = ap.parse_args()
    cfg = {"name": "r912_submission_precision", "description": __doc__.split("\n\n")[0],
           "replica": {"variant": "make_classification", "class_sep": 0.71, "hamming": 4,
                       "flip_y": a.flips, "seeds": list(range(a.seeds))},
           "model": "lib_gen_replica.UnsupervisedGMM(X_all, 'highvar', K=2) = ssc TransductiveGMM defaults",
           "subset": "private (19,355 rows)"}
    with Run("r912_submission_precision", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
        rows = []
        t0 = time.perf_counter()
        for f in a.flips:
            for sd in range(a.seeds):
                rows.append(one(sd, f))
                r = rows[-1]
                print(f"[r912] flip={f} seed={sd} raw={r['auc_raw']:.7f} csv8-raw={r['d_csv8_minus_raw']:+.2e} "
                      f"logodds-raw={r['d_logodds_minus_raw']:+.2e} bayes-raw={r['d_bayes_minus_raw']:+.2e} "
                      f"uniq raw/csv8={r['n_unique_raw']}/{r['n_unique_csv8']} ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
        df = pd.DataFrame(rows)
        run.save_table("replicas", df)
        agg = []
        for f, d in df.groupby("flip_y"):
            row = {"flip_y": f, "n": len(d), "auc_raw_mean": d["auc_raw"].mean(), "auc_bayes_mean": d["auc_bayes"].mean(),
                   "n_unique_raw_mean": d["n_unique_raw"].mean(), "n_unique_csv8_mean": d["n_unique_csv8"].mean(),
                   "n_at_min_csv8_mean": d["n_at_min_csv8"].mean(), "n_at_max_csv8_mean": d["n_at_max_csv8"].mean(),
                   "spearman_csv8_raw_mean": d["spearman_csv8_raw"].mean()}
            for k in ("csv8", "logodds", "bayes"):
                x = d[f"d_{k}_minus_raw"]
                row[f"d_{k}_mean"] = x.mean()
                row[f"d_{k}_se"] = x.std(ddof=1) / np.sqrt(len(x))
                row[f"d_{k}_min"] = x.min()
                row[f"d_{k}_max"] = x.max()
            agg.append(row)
        agg = pd.DataFrame(agg)
        run.save_table("summary_by_flip", agg)
        out = paths.tables_dir(a.data)
        out.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out / "r912_submission_precision.csv", index=False)
        print("\n" + agg.T.to_string(), flush=True)
        run.finish(extra_summary={"by_flip": agg.to_dict("records"), "seconds": round(time.perf_counter() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
