"""R426 (research round 2, agent "r2_noise") - why did the conditional public prediction for r201 move from r936's
0.99884 to r425's 0.99868? Separates Monte-Carlo noise (r936: 125 accepted public subsets) from the change of the
label-noise weights (R156 4-point grid -> R922 flat prior on the extended grid).

Input: the same packed r420 arrays as r425 (research/r424_r420_pooled.npz). ABC on S003's public 0.99923 (+-1.5e-4,
csv scores), predicted public = 0.99923 + (public of X - public of S003) on the same subset and labels.

  per_level    conditional mean public difference X - S003 per flip_y level (n accepted, replica-cluster SE)
  designs      the prediction under: r936's design (4 levels, first 16 subsets, (1 - f/2)^250 prior) = r936;
               the same 4 levels and prior with all 512 subsets; R156 weights on the 4 levels with 512 subsets;
               R922 flat-prior weights truncated to f <= 0.01; R922 flat-prior weights on the full grid (= r425)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2noise --high-mem -- research/r426_abc_attribution.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ssc import paths
from ssc.tracking import Run

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--pack", default=os.path.join(HERE, "r424_r420_pooled.npz"))
p.add_argument("--window", type=float, default=1.5e-4)
p.add_argument("--n-boot", type=int, default=400)
a = p.parse_args()

OBS = 0.99923
W_R156 = {0.0: 0.0773, 0.0025: 0.4069, 0.005: 0.3608, 0.01: 0.1550}
PROCS = ["r201_qda_block", "r209_selftrain_qda_lw"]
blob = Path(a.pack).read_bytes()
Z = np.load(io.BytesIO(blob))
M = json.loads(bytes(Z["manifest_json"]).decode("utf-8"))
flips = np.array(M["flips"], float)
L = len(flips)
seeds = Z["seeds"]
n = len(seeds)
ABC = Z["abc"].astype(float)
ci = {c: i for i, c in enumerate(M["abc_columns"])}
lvl = ABC[:, ci["level"]].astype(int)
draw = ABC[:, ci["draw"]].astype(int)
s3 = ABC[:, ci["pub8_s003_gmm2_hv"]]
pos = {int(s): k for k, s in enumerate(seeds)}
rep = np.array([pos[int(s)] for s in ABC[:, ci["seed"]]])
acc = np.abs(s3 - OBS) <= a.window + 1e-12
D = {k: ABC[:, ci[f"pub8_{k}"]] - s3 for k in PROCS}

HYP = ("Written before running. The conditional public difference r201 - S003 is roughly flat across flip_y levels "
       "(-4e-4 to -6e-4), so the change from r936's 0.99884 to 0.99868 is mostly Monte-Carlo noise of r936's 125 "
       "accepted subsets (MC SE ~8e-5): r936's design with all 512 subsets per replica gives ~0.9987, and changing the "
       "weights from R156 to R922's flat prior moves the prediction by < 5e-5. Same for r209 (shift < 3e-5).")
cfg = {"name": "r426_abc_attribution", "input_pack": Path(a.pack).name,
       "input_pack_sha256": hashlib.sha256(blob).hexdigest(), "window": a.window, "weights_r156": W_R156,
       "n_boot": a.n_boot, "source_runs": [r["run_id"] for r in M["runs"]]}
tables = paths.tables_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)


def binned(lik):
    F = Z["lik_r922"][:, 0]
    near = np.abs(F[:, None] - flips[None, :]).argmin(1)
    w = np.bincount(near, weights=lik, minlength=L)
    return w / w.sum()


with Run("r426_abc_attribution", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    rng = np.random.default_rng(426)
    boots = [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.n_boot)]
    rows = []
    for li, f in enumerate(flips):
        m = acc & (lvl == li)
        r = {"flip_y": f, "n_accepted": int(m.sum()), "n_replicas": int(len(np.unique(rep[m])))}
        for k in PROCS:
            d = D[k][m]
            r[f"{k}_cond_diff_mean"] = float(d.mean()) if m.any() else np.nan
            bs = [float((c[rep[m]] * d).sum() / c[rep[m]].sum()) for c in boots if c[rep[m]].sum() > 0]
            r[f"{k}_cond_diff_cluster_se"] = float(np.std(bs, ddof=1)) if len(bs) > 1 else np.nan
        rows.append(r)
    PL = pd.DataFrame(rows)
    run.save_table("per_level", PL)
    PL.to_csv(tables / "r426_per_level.csv", index=False)

    i4 = [int(np.argmin(np.abs(flips - f))) for f in W_R156]
    prior250 = np.zeros(L)
    prior250[i4] = [(1 - flips[i] / 2) ** 250 for i in i4]
    w156 = np.zeros(L)
    w156[i4] = list(W_R156.values())
    wflat = binned(Z["lik_r922"][:, 2])
    wtrunc = np.where(flips <= 0.01 + 1e-12, wflat, 0.0)
    designs = [  # name, row mask, per-row weight rule ("abc": level prior x acceptance; "level": fixed level weights)
        ("r936 design: 4 levels, 16 subsets, (1-f/2)^250 prior x acceptance", acc & (draw < 16), "abc", prior250),
        ("4 levels, 512 subsets, (1-f/2)^250 prior x acceptance", acc, "abc", prior250),
        ("4 levels, 512 subsets, R156 posterior weights", acc, "level", w156),
        ("R922 flat prior truncated at f <= 0.01, 512 subsets", acc, "level", wtrunc),
        ("R922 flat prior, full grid to 0.03, 512 subsets (= r425)", acc, "level", wflat)]
    out_rows = []
    for name, m0, rule, wl in designs:
        m = m0 & (wl[lvl] > 0)
        if rule == "abc":
            wt = wl[lvl[m]]
        else:
            cnt = np.bincount(lvl[m], minlength=L).astype(float)
            wt = (wl / np.where(cnt > 0, cnt, np.inf))[lvl[m]]
        W = wt / wt.sum()
        r = {"design": name, "n_accepted": int(m.sum()), "n_replicas": int(len(np.unique(rep[m]))),
             "post_mean_f": float((np.bincount(lvl[m], weights=W, minlength=L) * flips).sum())}
        for k in PROCS:
            d = D[k][m]
            r[f"{k}_public_pred"] = float(OBS + (W * d).sum())
            bs = []
            for c in boots:
                cw = c[rep[m]].astype(float)
                if rule == "abc":
                    wb = wt * cw
                else:
                    lw = np.bincount(lvl[m], weights=cw, minlength=L)
                    wb = cw * (wl / np.where(lw > 0, lw, np.inf))[lvl[m]]
                if wb.sum() > 0:
                    bs.append(float((wb * d).sum() / wb.sum()))
            r[f"{k}_mc_se"] = float(np.std(bs, ddof=1))
            r[f"{k}_p_higher"] = float(W[d > 1e-9].sum())
        out_rows.append(r)
    DS = pd.DataFrame(out_rows)
    run.save_table("designs", DS)
    DS.to_csv(tables / "r426_designs.csv", index=False)
    (tables / "r426_designs.md").write_text(DS.to_markdown(index=False, floatfmt=".6g"), encoding="utf-8")
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(PL.to_string(index=False))
        print(DS.to_string(index=False))
    run.finish(extra_summary={"designs": DS.round(7).to_dict(orient="records")})
