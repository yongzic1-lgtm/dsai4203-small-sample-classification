"""R913 - same question as R912 for the supervised QDA family (S002, r201_qda_block): does the saturation of
predict_proba at 0 / 1 plus the "%.8f" submission format cost AUC? (verifier `repro`, 2026-10-22)

Real files (research/r911_repro_compare.py): S002 (QDA reg 0.1, train+test block) has 760 rows at exactly 0.0
and 718 at 1.0 in the uploaded csv; a submission of r201_qda_block would keep 7,724 distinct values out of
18,332 (largest tie 5,157 rows). On make_classification replicas (class_sep 0.71, hamming 4) with known test
labels, private-subset AUC of the same fitted QDA scored as
  raw      predict_proba[:, 1] (float64)              csv8     raw written with "%.8f" (what is uploaded)
  logodds  decision_function (unsaturated log-odds)   bayes    true Bayes log-odds
Model: lib_sup.HighVarColumns (block derived from the 250 training rows only) + StandardScaler + QDA with
reg_param 0.003 (the value r201_qda_block chose in 49 of 100 folds) and 0.1 (S002's value).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s vrepro -- research/r913_submission_precision_qda.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from lib_sup import HighVarColumns, quiet  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

HYP = ("As for the GMM (R912): saturated or rounded QDA scores only tie rows whose posterior is within 5e-9 of "
       "0 or 1; prediction: mean private AUC(csv8) - AUC(raw) within +-2e-5 and |mean| < 2 SE; "
       "AUC(logodds) - AUC(raw) may be slightly positive if the QDA log-odds order the confident rows "
       "informatively (it is the sharper model), but < 1e-4. Refuted if csv8 - raw < -1e-4 on average.")
REGS = (0.003, 0.1)


def csv8(s: np.ndarray) -> np.ndarray:
    return np.array([float("%.8f" % v) for v in s])


def one(seed: int, flip_y: float) -> list[dict]:
    rep = G.sample_make_classification(seed, class_sep=0.71, flip_y=flip_y, hamming=4)
    m = rep.private_mask
    y = rep.y_test[m]
    out = []
    for reg in REGS:
        pipe = Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", QDA(reg_param=reg))])
        cm = quiet()
        try:
            pipe.fit(rep.X_train, rep.y_train)
            p = pipe.predict_proba(rep.X_test)[:, 1]
            lo = pipe.decision_function(rep.X_test)
        finally:
            cm.__exit__(None, None, None)
        p8 = csv8(p)
        r = {"seed": seed, "flip_y": flip_y, "reg_param": reg, "n_flipped": int(rep.params["n_flipped_true"]),
             "block_size": int(pipe.named_steps["sel"].support_.sum()),
             "auc_raw": G.auc_fast(y, p[m]), "auc_csv8": G.auc_fast(y, p8[m]), "auc_logodds": G.auc_fast(y, lo[m]),
             "auc_bayes": G.auc_fast(y, rep.bayes_test[m]),
             "n_unique_raw": int(len(np.unique(p))), "n_unique_csv8": int(len(np.unique(p8))),
             "n_exact_0_or_1_raw": int(((p == 0) | (p == 1)).sum()),
             "largest_tie_csv8": int(pd.Series(p8).value_counts().iloc[0]),
             "spearman_csv8_raw": float(stats.spearmanr(p8, p).statistic)}
        for k in ("csv8", "logodds", "bayes"):
            r[f"d_{k}_minus_raw"] = r[f"auc_{k}"] - r["auc_raw"]
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="raw", choices=paths.SOURCES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--flips", type=float, nargs="*", default=[0.0, 0.005])
    a = ap.parse_args()
    cfg = {"name": "r913_submission_precision_qda", "description": __doc__.split("\n\n")[0],
           "replica": {"variant": "make_classification", "class_sep": 0.71, "hamming": 4, "flip_y": a.flips,
                       "seeds": list(range(a.seeds))},
           "model": "HighVarColumns(250 train rows) + StandardScaler + QDA(reg_param in %s)" % (list(REGS),),
           "subset": "private (19,355 rows)"}
    with Run("r913_submission_precision_qda", cfg, kind="analysis", source=a.data, author=a.author,
             hypothesis=HYP) as run:
        rows, t0 = [], time.perf_counter()
        for f in a.flips:
            for sd in range(a.seeds):
                for r in one(sd, f):
                    rows.append(r)
                    print(f"[r913] flip={f} seed={sd} reg={r['reg_param']} raw={r['auc_raw']:.6f} "
                          f"csv8-raw={r['d_csv8_minus_raw']:+.2e} logodds-raw={r['d_logodds_minus_raw']:+.2e} "
                          f"exact0/1={r['n_exact_0_or_1_raw']} tie={r['largest_tie_csv8']} "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        run.save_table("replicas", df)
        agg = []
        for (f, reg), d in df.groupby(["flip_y", "reg_param"]):
            row = {"flip_y": f, "reg_param": reg, "n": len(d), "auc_raw_mean": d["auc_raw"].mean(),
                   "auc_bayes_mean": d["auc_bayes"].mean(), "n_unique_csv8_mean": d["n_unique_csv8"].mean(),
                   "largest_tie_csv8_mean": d["largest_tie_csv8"].mean(),
                   "n_exact_0_or_1_raw_mean": d["n_exact_0_or_1_raw"].mean()}
            for k in ("csv8", "logodds", "bayes"):
                x = d[f"d_{k}_minus_raw"]
                row.update({f"d_{k}_mean": x.mean(), f"d_{k}_se": x.std(ddof=1) / np.sqrt(len(x)),
                            f"d_{k}_min": x.min(), f"d_{k}_max": x.max()})
            agg.append(row)
        agg = pd.DataFrame(agg)
        run.save_table("summary_by_flip_reg", agg)
        out = paths.tables_dir(a.data)
        out.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out / "r913_submission_precision_qda.csv", index=False)
        print("\n" + agg.T.to_string(), flush=True)
        run.finish(extra_summary={"by_flip_reg": agg.to_dict("records"), "seconds": round(time.perf_counter() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
