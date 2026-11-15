"""R425 (research round 2, agent "r2_noise") - pooled analysis of the r420 replicas (flip_y grid extended to 0.03),
re-weighted with verify_gen's flat-prior likelihood (R922) instead of R156's 4-point grid.

Input: research/r424_r420_pooled.npz (packed by research/r424_pack_r420.py; sha256 of every source array in its
manifest) and research/r935_r933_pooled.npz (cand's packed r933 arrays, for the bit-exact check).

  check         r420 arrays at the 4 original levels / first 16 public subsets vs the archived r933 tn0 arrays
  levels        per flip_y level: S003 private AUC (csv scores) mean / sd / 5-95 %, paired differences of r201, r209
                and Bayes vs S003 (MC SE over replicas), acceptance rates of the ABC windows
  weights       level weights: r936 (R156 grid), R922 flat prior x combined likelihood (random split, exact window),
                R922 flat prior x train likelihood only (pure ABC prior), R421 stratified-split likelihood
  private       posterior predictive of S003's (and r201's, r209's) private AUC under each weighting, P(>= 0.999),
                expected gain of the second final selection E[max(S003, X)] - E[S003]
  public_pred   public AUC of S005 (r201) and S006 (r209) conditional on S003's public 0.99923 (ABC on the replica
                public subsets; two weightings; windows +-5e-5 and +-1.5e-4), compared with B006 / B007; the
                ABC-conditional private AUC of S003; cluster-bootstrap MC SE over replicas
  r936_repro    the same ABC restricted to r936's design (4 levels, 16 subsets, (1 - f/2)^250 prior), which must
                reproduce reports/tables/r936_public_abc.csv

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2noise --high-mem -- research/r425_flip_grid_reweight.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--pack", default=os.path.join(HERE, "r424_r420_pooled.npz"))
p.add_argument("--r935", default=os.path.join(HERE, "r935_r933_pooled.npz"))
p.add_argument("--n-boot", type=int, default=400)
a = p.parse_args()

OBS = 0.99923
W_R156 = {0.0: 0.0773, 0.0025: 0.4069, 0.005: 0.3608, 0.01: 0.1550}
BEL = {"r201_qda_block": {"belief": "B006", "point": 0.9988, "lo": 0.9980, "hi": 0.9997, "bad": 0.9960,
                          "p_higher": "0.20", "pred_sd": 0.00066},
       "r209_selftrain_qda_lw": {"belief": "B007", "point": 0.9991, "lo": 0.9987, "hi": 0.9996, "bad": 0.9975,
                                 "p_higher": "0.25-0.28 (equal 0.04)", "pred_sd": 0.00033}}
WINDOWS = [5e-5, 1.5e-4]

blob = Path(a.pack).read_bytes()
Z = np.load(io.BytesIO(blob))
M = json.loads(bytes(Z["manifest_json"]).decode("utf-8"))
blob935 = Path(a.r935).read_bytes()
Z9 = np.load(io.BytesIO(blob935))
M9 = json.loads(bytes(Z9["manifest_json"]).decode("utf-8"))

flips = np.array(M["flips"], float)
procs, sel, cols = M["procs"], M["sel_procs"], M["abc_columns"]
seeds = Z["seeds"]
n, L = len(seeds), len(flips)
n_abc = {r["n_abc"] for r in M["runs"]}
assert len(n_abc) == 1
n_abc = n_abc.pop()
priv8, pub8 = Z["priv8"].astype(float), Z["pub8"].astype(float)       # (n, L, 16, 4) in sel order
ABC = Z["abc"].astype(float)
ci = {c: i for i, c in enumerate(cols)}
J = {k: sel.index(k) for k in sel}

HYP = ("Written before running (after reading only the r420 quick summaries of the runs, not these pooled numbers). "
       "(1) Bit-exact agreement with the archived r933 tn0 arrays at the 4 shared levels. (2) Under R922's flat-prior "
       "weights S003's private predictive has mean ~0.9968 and 90 % interval ~[0.992, 0.9996] (R922 on real features: "
       "0.9968, [0.9923, 0.9996]), i.e. ~1e-3 lower and twice as wide as r936's 0.9978 (R156 weights). (3) The paired "
       "private differences to S003 are flat in flip_y, so the second-selection gains (r209 +1.5e-5, r201 +4e-6) and "
       "D010's ranking do not change. (4) Conditional on S003's public 0.99923 the predicted public AUC is ~0.9988 for "
       "r201 and ~0.9991 for r209 under both weightings (within 1.5e-4 of B006 / B007), P(r201 > S003) ~0.2, "
       "P(r209 > S003) ~0.25-0.3. (5) The r936-design restriction reproduces r936_public_abc exactly.")

tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r425_flip_grid_reweight", "input_pack": Path(a.pack).name,
       "input_pack_sha256": hashlib.sha256(blob).hexdigest(), "input_r935_sha256": hashlib.sha256(blob935).hexdigest(),
       "source_runs": [r["run_id"] for r in M["runs"]], "likelihood_sources": {k: M.get(k) for k in ("lik_r922", "lik_r421")},
       "flips": M["flips"], "n_abc": n_abc, "windows": WINDOWS, "weights_r156": W_R156, "n_boot": a.n_boot,
       "beliefs": BEL}


def wquant(x, w, q):
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(cw, q), len(x) - 1)])


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


with Run("r425_flip_grid_reweight", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    out = {"n_replicas": int(n), "n_levels": int(L), "n_abc_rows_le_1.5e-4": int(len(ABC))}

    # ---- 1. bit-exact check against the archived r933 tn0 arrays
    li_orig = [int(np.argmin(np.abs(flips - f))) for f in M["orig_levels"]]
    ref = {}
    for e in M9["runs"]:
        if e["train_noise"] or e.get("round_decimals") != 8:
            continue
        i = e["index"]
        for s_i, s in enumerate(Z9[f"{i}__seeds"]):
            ref[int(s)] = (Z9[f"{i}__pub_auc_r8"][s_i], Z9[f"{i}__priv_auc_r8"][s_i], e["run_id"])
    chk = []
    for nm, mine in (("pub_auc_r8", Z["chk_pub8"]), ("priv_auc_r8", Z["chk_priv8"])):
        both = [k for k, s in enumerate(seeds) if int(s) in ref]
        A_ = np.stack([mine[k] for k in both])
        B_ = np.stack([ref[int(seeds[k])][0 if nm == "pub_auc_r8" else 1] for k in both])
        chk.append({"array": nm, "n_replicas_compared": len(both), "n_values": int(A_.size),
                    "n_identical": int((A_ == B_).sum()), "max_abs_diff": float(np.abs(A_ - B_).max()),
                    "reference_runs": sorted({ref[int(seeds[k])][2] for k in both})})
    CHK = pd.DataFrame(chk)
    run.save_table("check", CHK)
    CHK.to_csv(tables / "r425_check.csv", index=False)
    print(CHK.to_string(index=False), flush=True)
    out["bit_exact"] = bool((CHK.n_identical == CHK.n_values).all())

    # ---- 2. level weights
    LK = Z["lik_r922"]                       # flip_y, Ltrain_rel, Lcomb_main  (R922, uniform prior on [0, 0.1])
    F = LK[:, 0]
    near = np.abs(F[:, None] - flips[None, :]).argmin(1)

    def bin_w(lik):
        w = np.bincount(near, weights=lik, minlength=L)
        return w / w.sum()
    WTS = {"r936_R156": np.array([W_R156.get(round(f, 4), 0.0) for f in flips]),
           "R922_flat_combined": bin_w(LK[:, 2]),
           "R922_flat_train_only": bin_w(LK[:, 1])}
    WTS["r936_R156"] = WTS["r936_R156"] / WTS["r936_R156"].sum()
    if "lik_r421" in Z.files:
        L4 = Z["lik_r421"]
        assert np.allclose(L4[:, 0], F)
        WTS["R421_flat_random"] = bin_w(L4[:, 2])
        WTS["R421_flat_stratified_mix"] = bin_w(L4[:, 5])
    beyond = float(LK[F > flips.max() + 1e-12, 2].sum() / LK[:, 2].sum())
    out["R922_flat_mass_beyond_grid"] = beyond
    WT = pd.DataFrame({"flip_y": flips, **WTS})
    run.save_table("weights", WT)
    WT.to_csv(tables / "r425_weights.csv", index=False)
    post_mean_f = {k: float((w * flips).sum()) for k, w in WTS.items()}
    out["posterior_mean_f_binned"] = post_mean_f

    # ---- 3. per-level table
    j0, j1, j2, jb = J["s003_gmm2_hv"], J["r201_qda_block"], J["r209_selftrain_qda_lw"], J["bayes"]
    lev = []
    for li, f in enumerate(flips):
        P = priv8[:, li]                                   # (n, 16, 4)
        r = {"flip_y": f, "s003_private_mean": float(P[..., j0].mean()), "s003_private_sd": float(P[:, 0, j0].std(ddof=1)),
             "s003_private_q05": float(np.quantile(P[..., j0], .05)), "s003_private_q95": float(np.quantile(P[..., j0], .95)),
             "approx_1_minus_f_over_2": 1 - f / 2, "s003_public_mean": float(pub8[:, li, :, j0].mean())}
        for nm, j in (("r201", j1), ("r209", j2), ("bayes", jb)):
            d = (P[..., j] - P[..., j0]).mean(1)
            r[f"diff_{nm}_mean"] = float(d.mean())
            r[f"diff_{nm}_mc_se"] = se(d)
            r[f"p_{nm}_beats_s003"] = float((P[..., j] > P[..., j0]).mean())
        lv = ABC[:, ci["level"]] == li
        for w in WINDOWS + [5e-6]:
            acc = lv & (np.abs(ABC[:, ci["pub8_s003_gmm2_hv"]] - OBS) <= w + 1e-12)
            if w == 5e-6:   # exact 5-decimal rounding of the displayed public score
                v = ABC[:, ci["pub8_s003_gmm2_hv"]]
                acc = lv & (v >= OBS - 5e-6 - 1e-12) & (v < OBS + 5e-6 - 1e-12)
            r[f"accept_{'exact' if w == 5e-6 else f'pm{w:g}'}"] = float(acc.sum() / (n * n_abc))
        r["n_abc_rows_pm1.5e-4"] = int((lv & (np.abs(ABC[:, ci["pub8_s003_gmm2_hv"]] - OBS) <= 1.5e-4 + 1e-12)).sum())
        lev.append(r)
    LEV = pd.DataFrame(lev)
    for k, w in WTS.items():
        LEV[f"w_{k}"] = w
    run.save_table("levels", LEV)
    LEV.to_csv(tables / "r425_levels.csv", index=False)
    (tables / "r425_levels.md").write_text(LEV.to_markdown(index=False, floatfmt=".6g"), encoding="utf-8")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(LEV.to_string(index=False), flush=True)

    # ---- 4. posterior predictive of the private AUC (unconditional on the replica public; levels weighted)
    prv = []
    for k, w in WTS.items():
        vals = priv8.transpose(1, 0, 2, 3).reshape(L, -1, 4)          # (L, n*16, 4)
        ww = np.repeat(w[:, None], vals.shape[1], 1).ravel()
        V = vals.reshape(-1, 4)
        keep = ww > 0
        V, ww = V[keep], ww[keep]
        r = {"weights": k, "post_mean_f": post_mean_f[k]}
        for nm, j in (("s003", j0), ("r201", j1), ("r209", j2), ("bayes", jb)):
            x = V[:, j]
            m_ = float((ww * x).sum() / ww.sum())
            r[f"{nm}_mean"] = m_
            if nm == "s003":
                r["s003_sd"] = float(np.sqrt((ww * (x - m_) ** 2).sum() / ww.sum()))
                for q in (.05, .5, .95):
                    r[f"s003_q{int(q * 100):02d}"] = wquant(x, ww, q)
                r["P_s003_ge_0.999"] = float(ww[x >= 0.999 - 5e-6].sum() / ww.sum())
                r["P_abs_pub_minus_priv_le_0.001"] = float(ww[np.abs(x - OBS) <= 0.001].sum() / ww.sum())
        for nm, j in (("r201", j1), ("r209", j2)):
            g = np.maximum(V[:, j], V[:, j0]) - V[:, j0]
            r[f"gain_second_{nm}"] = float((ww * g).sum() / ww.sum())
            r[f"diff_{nm}"] = float((ww * (V[:, j] - V[:, j0])).sum() / ww.sum())
        prv.append(r)
    PRV = pd.DataFrame(prv)
    run.save_table("private", PRV)
    PRV.to_csv(tables / "r425_private.csv", index=False)
    (tables / "r425_private.md").write_text(PRV.to_markdown(index=False, floatfmt=".6g"), encoding="utf-8")
    print(PRV.to_string(index=False), flush=True)

    # ---- 5. ABC: public of r201 / r209 given S003's public, and S003's conditional private
    lvl = ABC[:, ci["level"]].astype(int)
    sid = ABC[:, ci["seed"]].astype(int)
    s3p = ABC[:, ci["pub8_s003_gmm2_hv"]]
    uniq_seeds, sidx = np.unique(sid, return_inverse=True)
    rng = np.random.default_rng(425)
    boot_seeds = [rng.integers(0, n, n) for _ in range(a.n_boot)]
    seed_pos = {int(s): k for k, s in enumerate(seeds)}
    rep_index = np.array([seed_pos[int(s)] for s in sid])

    def abc_weights(method, acc):
        if method == "R922_flat_combined":         # R922 posterior per level, equal weight within level
            cnt = np.bincount(lvl[acc], minlength=L).astype(float)
            w_l = WTS["R922_flat_combined"] / np.where(cnt > 0, cnt, np.inf)
            dropped = float(WTS["R922_flat_combined"][cnt == 0].sum())
            return w_l[lvl[acc]], dropped
        if method == "pure_ABC_R922_train":        # flat prior x train likelihood; public via acceptance
            return WTS["R922_flat_train_only"][lvl[acc]], 0.0
        if method == "R421_stratified_mix":
            cnt = np.bincount(lvl[acc], minlength=L).astype(float)
            w_l = WTS["R421_flat_stratified_mix"] / np.where(cnt > 0, cnt, np.inf)
            return w_l[lvl[acc]], float(WTS["R421_flat_stratified_mix"][cnt == 0].sum())
        raise KeyError(method)

    methods = ["R922_flat_combined", "pure_ABC_R922_train"] + (["R421_stratified_mix"] if "R421_flat_stratified_mix" in WTS else [])
    pp = []
    for method in methods:
        for w in WINDOWS:
            acc = np.abs(s3p - OBS) <= w + 1e-12
            wt, dropped = abc_weights(method, acc)
            W_ = wt / wt.sum()
            lev_post = np.bincount(lvl[acc], weights=W_, minlength=L)
            base = {"method": method, "window": w, "n_rows": int(acc.sum()), "n_replicas": int(len(np.unique(sid[acc]))),
                    "eff_n": float(1 / (W_ ** 2).sum()), "dropped_level_mass": dropped,
                    "implied_post_mean_f": float((lev_post * flips).sum()),
                    "s003_public_mean_in_window": float((W_ * s3p[acc]).sum())}
            s3v = ABC[acc, ci["priv8_s003_gmm2_hv"]]
            base["s003_cond_private_mean"] = float((W_ * s3v).sum())
            base["s003_cond_private_q05"] = wquant(s3v, W_, .05)
            base["s003_cond_private_q95"] = wquant(s3v, W_, .95)
            base["s003_cond_P_private_ge_0.999"] = float(W_[s3v >= 0.999 - 5e-6].sum())
            # cluster bootstrap over replicas: resample seeds, keep the method's weights
            ri = rep_index[acc]
            boots = {"r201_qda_block": [], "r209_selftrain_qda_lw": [], "s003_priv": []}
            dmat = {k: ABC[acc, ci[f"pub8_{k}"]] - s3p[acc] for k in ("r201_qda_block", "r209_selftrain_qda_lw")}
            for bs in boot_seeds:
                cnt_rep = np.bincount(bs, minlength=n)[ri].astype(float)
                if method == "pure_ABC_R922_train":
                    wb = wt * cnt_rep
                else:
                    lw = np.bincount(lvl[acc], weights=cnt_rep, minlength=L)
                    base_w = WTS["R922_flat_combined" if method == "R922_flat_combined" else "R421_flat_stratified_mix"]
                    wb = cnt_rep * (base_w / np.where(lw > 0, lw, np.inf))[lvl[acc]]
                if wb.sum() <= 0:
                    continue
                wb = wb / wb.sum()
                for k in dmat:
                    boots[k].append(float((wb * dmat[k]).sum()))
                boots["s003_priv"].append(float((wb * s3v).sum()))
            base["s003_cond_private_mean_mc_se"] = float(np.std(boots["s003_priv"], ddof=1))
            for k in ("r201_qda_block", "r209_selftrain_qda_lw", "r210_qda_block_nll", "r255_bqda_eb_hv",
                      "r105_gmm2_block_mle", "bayes"):
                pubk = ABC[acc, ci[f"pub8_{k}"]]
                d = pubk - s3p[acc]
                pred = OBS + d
                r = dict(base, proc=k)
                r["public_raw_mean"] = float((W_ * pubk).sum())
                r["public_pred_mean"] = float((W_ * pred).sum())
                r["public_pred_mc_se"] = float(np.std(boots[k], ddof=1)) if k in boots else np.nan
                r["public_pred_sd"] = float(np.sqrt((W_ * (pred - r["public_pred_mean"]) ** 2).sum()))
                for q in (.10, .50, .90):
                    r[f"public_pred_q{int(q * 100):02d}"] = wquant(pred, W_, q)
                r["P_higher_than_s003"] = float(W_[d > 1e-9].sum())
                r["P_equal_s003"] = float(W_[np.abs(d) <= 1e-9].sum())
                privk = ABC[acc, ci[f"priv8_{k}"]]
                r["cond_private_mean"] = float((W_ * privk).sum())
                r["cond_private_diff_vs_s003"] = float((W_ * (privk - s3v)).sum())
                if k in BEL:
                    b = BEL[k]
                    r["belief"] = b["belief"]
                    r["belief_point"] = b["point"]
                    r["P_in_belief_80pct"] = float(W_[(pred >= b["lo"] - 5e-6) & (pred <= b["hi"] + 5e-6)].sum())
                    r["P_below_bad_threshold"] = float(W_[pred < b["bad"]].sum())
                pp.append(r)
    PP = pd.DataFrame(pp)
    run.save_table("public_pred", PP)
    PP.to_csv(tables / "r425_public_pred.csv", index=False)
    show = ["method", "window", "proc", "n_rows", "n_replicas", "eff_n", "implied_post_mean_f", "public_pred_mean",
            "public_pred_mc_se", "public_pred_sd", "public_pred_q10", "public_pred_q90", "P_higher_than_s003",
            "P_equal_s003", "P_in_belief_80pct", "s003_cond_private_mean", "s003_cond_private_q05",
            "s003_cond_private_q95"]
    (tables / "r425_public_pred.md").write_text(PP[show].to_markdown(index=False, floatfmt=".5g"), encoding="utf-8")
    with pd.option_context("display.width", 280, "display.max_columns", 40):
        print(PP[show].to_string(index=False), flush=True)

    # ---- 6. r936-design reproduction (4 levels, first 16 subsets, prior (1 - f/2)^250, window 1.5e-4)
    rep = []
    li4 = np.array(li_orig)
    pri = np.array([(1 - flips[li] / 2) ** 250 for li in li4])
    pri = pri / pri.sum()
    accr = np.isin(lvl, li4) & (ABC[:, ci["draw"]] < 16) & (np.abs(s3p - OBS) <= 1.5e-4 + 1e-12)
    wr = np.zeros(len(ABC))
    for li, pv in zip(li4, pri):
        wr[lvl == li] = pv
    Wr = wr[accr] / wr[accr].sum()
    for k in ("r201_qda_block", "r209_selftrain_qda_lw"):
        rep.append({"proc": k, "n_accepted": int(accr.sum()), "n_replicas": int(len(np.unique(sid[accr]))),
                    "public_mean": float((Wr * ABC[accr, ci[f"pub8_{k}"]]).sum()),
                    "public_diff_vs_s003": float((Wr * (ABC[accr, ci[f"pub8_{k}"]] - s3p[accr])).sum()),
                    "p_public_higher": float(Wr[ABC[accr, ci[f"pub8_{k}"]] > s3p[accr] + 1e-9].sum())})
    REP = pd.DataFrame(rep)
    run.save_table("r936_repro", REP)
    REP.to_csv(tables / "r425_r936_repro.csv", index=False)
    print(REP.to_string(index=False), flush=True)

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].fill_between(flips, LEV.s003_private_q05, LEV.s003_private_q95, alpha=.25, color="#2a78d6")
    ax[0].plot(flips, LEV.s003_private_mean, color="#2a78d6", label="S003 private (replicas, csv scores)")
    ax[0].plot(flips, 1 - flips / 2, ":", color="#0b0b0b", label="1 - f/2")
    ax[0].axhline(0.999, color="#eb6834", ls="--", lw=1)
    ax[0].set_xlabel("flip_y")
    ax[0].set_ylabel("private AUC (mean, 5-95 %)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("S003 private AUC by label-noise level")
    wdt = 0.0004
    ax[1].bar(flips - wdt / 2, WTS["r936_R156"], width=wdt, label="r936 (R156 grid)", color="#888")
    ax[1].bar(flips + wdt / 2, WTS["R922_flat_combined"], width=wdt, label="R922 flat prior", color="#2a78d6")
    ax[1].set_xlabel("flip_y level")
    ax[1].set_ylabel("weight")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Level weights")
    sub = PP[(PP.window == 1.5e-4) & PP.proc.isin(list(BEL))]
    yl, yt = [], []
    for i, (k, g) in enumerate(sub.groupby("proc", sort=False)):
        for jj, (_, r) in enumerate(g.iterrows()):
            yv = i + 0.18 * jj
            ax[2].plot([r.public_pred_q10, r.public_pred_q90], [yv, yv], color="#2a78d6")
            ax[2].plot(r.public_pred_mean, yv, "o", color="#eb6834")
            ax[2].text(r.public_pred_q90 + 2e-5, yv, r.method, fontsize=6, va="center")
        b = BEL[k]
        ax[2].plot([b["lo"], b["hi"]], [i - 0.2] * 2, color="#0b0b0b", lw=3)
        yt.append(i)
        yl.append(f"{b['belief']} {k.split('_')[0]}")
    ax[2].axvline(OBS, color="#888", ls=":")
    ax[2].set_yticks(yt)
    ax[2].set_yticklabels(yl)
    ax[2].set_xlabel("public AUC given S003 = 0.99923 (10-90 %); black = belief 80 %")
    for x_ in ax:
        for sp in ("top", "right"):
            x_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "r425_flip_grid.png", dpi=150)
    run.save_figure("r425_flip_grid", fig)
    plt.close(fig)

    out["private"] = PRV.set_index("weights")[["s003_mean", "s003_q05", "s003_q50", "s003_q95", "P_s003_ge_0.999",
                                               "gain_second_r201", "gain_second_r209"]].round(7).to_dict(orient="index")
    out["public_pred_pm1.5e-4"] = PP[(PP.window == 1.5e-4) & PP.proc.isin(list(BEL))][
        ["method", "proc", "public_pred_mean", "public_pred_mc_se", "public_pred_q10", "public_pred_q90",
         "P_higher_than_s003", "P_equal_s003", "P_in_belief_80pct"]].round(6).to_dict(orient="records")
    out["r936_repro"] = REP.round(7).to_dict(orient="records")
    run.finish(extra_summary=out)
    print(json.dumps(out, indent=1, default=str))
