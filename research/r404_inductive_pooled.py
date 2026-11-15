"""R404 - pooled analysis of the r402 replicas (research round 2, "r2_inductive").

Input: research/r403_r402_pooled.npz (packed copy of the archived r402 runs, sha256 of every source file in its
manifest) and research/r935_r933_pooled.npz (r933, for the cross-check of the shared procedures).

(a) Strictly inductive final selection (D010 switch 3). Design = r936's tn0 group: 222 parametric replicas, clean
    training labels, test-label noise only, scores formatted like the submission CSV ("%.8f"), 16 public/private
    splits per replica. Two weightings of the label-noise levels:
      R156       0 / 0.0025 / 0.005 / 0.01 with 0.0773 / 0.4069 / 0.3608 / 0.1550 (run 20261022-081314_r156_flip_posterior)
      flat_R922  posterior of R922 under a flat prior on flip_y in [0, 0.1] (combined train + public likelihood,
                 reports/tables/r922_likelihood_fine.csv), binned to the nearest simulated level
                 0 / 0.0025 / 0.005 / 0.0075 / 0.01 / 0.015 / 0.02 / 0.03
    Per candidate: expected private AUC, paired difference vs r201_qda_block (mean, Monte-Carlo SE over replicas,
    5-95 % over replica-draws), P(beat / tie r201), the same with the variance-reduced expected AUC (labels integrated
    out under the noisy Bayes posterior), difference vs S003 and regret vs Bayes; per noise level; every pair of
    inductive candidates as the two final selections (Kaggle scores the better one): E[max(A, B)] - E[r201] and
    E[max(A, B)] - the better member alone; public-LB predictive of each candidate given S003's real public 0.99923
    (ABC as in r936).
(b) D009 evidence (2): boundary-row ranking on the 222 replicas. Rows whose TRUE Bayes posterior is in (0.1, 0.9)
    (all 19,750 test rows, ~200 per replica): AUC against the clean labels, expected AUC under the Bayes posterior,
    Kendall tau with the posterior; and r254's definition (rows where S003's own score is in (0.1, 0.9)). Paired
    differences vs S003 with Monte-Carlo SE and the share of replicas in which S003 ranks better.
(c) Checks: r402 reproduces r933's CSV-score private and public AUCs for the shared procedures on the shared
    seeds; the real-data pass reproduces the archived test_pred.npy of every reference run.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s r2ind -- research/r404_inductive_pooled.py
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
p.add_argument("--npz", default=os.path.join(HERE, "r403_r402_pooled.npz"))
p.add_argument("--r933-npz", default=os.path.join(HERE, "r935_r933_pooled.npz"))
p.add_argument("--s003-public", type=float, default=0.99923)
p.add_argument("--abc-window", type=float, default=0.00015)
a = p.parse_args()

REF, BASE = "s003_gmm2_hv", "r201_qda_block"
INDUCTIVE = ["r201_qda_block", "r210_qda_block_nll", "x_qda_block_nll_ext", "r255_bqda_eb_trainonly",
             "x_bqda_trainonly_wk", "r208_qda_block_lw", "x_ens_r201_bqda_to"]
W_R156 = {0.0: 0.0773, 0.0025: 0.4069, 0.005: 0.3608, 0.01: 0.1550}

blob = Path(a.npz).read_bytes()
Z = np.load(io.BytesIO(blob))
man = json.loads(bytes(Z["manifest_json"]).decode("utf-8"))
runs = man["runs"]
key = {(tuple(e["flips"]), tuple(e["procedures"]), e["n_draws"]) for e in runs}
if len(key) != 1:
    raise SystemExit(f"inconsistent r402 configs: {key}")
flips, procs, n_draws = list(runs[0]["flips"]), list(runs[0]["procedures"]), runs[0]["n_draws"]
cat = lambda nm: np.concatenate([Z[f"{e['index']}__{nm}"] for e in runs])  # noqa: E731
PRIV, PUB, EXP, BND, SEEDS = cat("priv_auc_r8"), cat("pub_auc_r8"), cat("exp_auc"), cat("bnd_stats"), cat("seeds")
if len(np.unique(SEEDS)) != len(SEEDS):
    raise SystemExit("duplicate seeds")
metas = sum([json.loads(bytes(Z[f"{e['index']}__meta_json"]).decode("utf-8")) for e in runs], [])
n = len(SEEDS)
J = {k: procs.index(k) for k in procs}
j0, jb, jr = J[REF], J["bayes"], J[BASE]

# ---- weights
fl = np.array(flips)
w156 = np.array([W_R156.get(f, 0.0) for f in flips])
w156 = w156 / w156.sum()
fg, Lc = Z["r922__flip_y"], Z["r922__Lcomb_main"]
post = Lc / Lc.sum()                                  # flat prior on the R922 grid [0, 0.1]
near = np.abs(fg[:, None] - fl[None, :]).argmin(1)
wflat = np.bincount(near, weights=post, minlength=len(fl))
WEIGHTS = {"R156": w156, "flat_R922": wflat}
wtab = pd.DataFrame({"flip_y": flips, "w_R156": w156, "w_flat_R922": wflat})

HYP = ("Written before running (after the r402 job logs printed per-run quick summaries for flip 0/0.005/0.01 only). "
       "(a) Strictly inductive candidates vs r201 (CSV scores): ext-grid QDA +1.5e-4 to +2.2e-4, r210 +6e-5 to "
       "+1.2e-4, BQDA-trainonly and its wide-kappa variant below r201, QDA-LW far below (-3e-3 or worse), the "
       "r201+BQDA-trainonly average within +-1.5e-4 of r201; the ranking is the same under the R156 and the flat-R922 "
       "weights, and every difference is flat across flip_y. Best inductive pair: ext-grid QDA with r210 or r201; the "
       "pair adds < 2e-5 over the ext-grid QDA alone. (b) Bayes-boundary rows: S003 within 0.005 of the Bayes "
       "posterior's AUC; S003 ranks better than every supervised procedure in > 90% of replicas; the r254 gap "
       "(0.825 vs 0.702) is of the right sign but its size differs on the neutral definition. (c) r402 = r933 exactly "
       "for the shared procedures and seeds; the real-data pass reproduces the archived test predictions "
       "bit-identically (r255_bqda_eb_hv to < 1e-13).")

tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def wl(A, w):
    """weighted over the level axis (axis 1)."""
    return np.tensordot(A, w, axes=([1], [0]))


cfg = {"name": "r404_inductive_pooled", "input_npz": Path(a.npz).name, "input_sha256": hashlib.sha256(blob).hexdigest(),
       "source_runs": [e["run_id"] for e in runs], "n_replicas": int(n), "flips": flips, "procedures": procs,
       "weights": {"R156": dict(zip(map(str, flips), w156.round(6).tolist())),
                   "flat_R922": dict(zip(map(str, flips), wflat.round(6).tolist()))},
       "r922_source": man["r922"], "s003_public": a.s003_public, "abc_window": a.abc_window,
       "r933_npz": Path(a.r933_npz).name}
with Run("r404_inductive_pooled", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    out = {}
    run.save_table("weights", wtab)
    wtab.to_csv(tables / "r404_weights.csv", index=False)

    # ---------------------------------------------------------------- (c) checks
    chk = []
    b933 = Path(a.r933_npz).read_bytes()
    Z3 = np.load(io.BytesIO(b933))
    m3 = json.loads(bytes(Z3["manifest_json"]).decode("utf-8"))
    for e in m3["runs"]:
        if e["train_noise"] or e.get("round_decimals") != 8:
            continue
        i = e["index"]
        s3 = Z3[f"{i}__seeds"]
        P3, U3 = Z3[f"{i}__priv_auc_r8"], Z3[f"{i}__pub_auc_r8"]
        for k in procs:
            if k not in e["procedures"]:
                continue
            j3 = e["procedures"].index(k)
            for l3, f in enumerate(e["flips"]):
                if f not in flips:
                    continue
                l4 = flips.index(f)
                common = np.intersect1d(s3, SEEDS)
                i3 = np.searchsorted(s3, common) if np.all(np.diff(s3) > 0) else np.array([list(s3).index(s) for s in common])
                i4 = np.array([list(SEEDS).index(s) for s in common])
                d1 = PRIV[i4, l4, :, J[k]] - P3[i3, l3, :, j3]
                d2 = PUB[i4, l4, :, J[k]] - U3[i3, l3, :, j3]
                chk.append({"r933_run": e["run_id"], "proc": k, "flip_y": f, "n_seeds": int(len(common)),
                            "max_abs_diff_private": float(np.abs(d1).max()), "max_abs_diff_public": float(np.abs(d2).max()),
                            "share_identical": float(((d1 == 0) & (d2 == 0)).mean())})
    chk = pd.DataFrame(chk)
    run.save_table("check_vs_r933", chk)
    chk.to_csv(tables / "r404_check_vs_r933.csv", index=False)
    out["check_vs_r933"] = {"rows": int(len(chk)), "max_abs_diff": float(chk[["max_abs_diff_private", "max_abs_diff_public"]].to_numpy().max()) if len(chk) else None,
                            "all_identical": bool(len(chk) and (chk.share_identical == 1).all())}
    real = []
    for e in runs:
        if not e.get("real_check") or f"{e['index']}__real_test_pred" not in Z.files:
            continue
        RT = Z[f"{e['index']}__real_test_pred"]
        pl = [k for k in procs if k != "bayes"]
        for jj, k in enumerate(pl):
            r = {"proc": k, "run": e["run_id"]}
            if f"ref__{k}" in Z.files:
                ref = Z[f"ref__{k}"]
                r.update(archived_run=man["references"][k]["run_id"], max_abs_diff=float(np.abs(RT[jj] - ref).max()),
                         bit_identical=bool(np.array_equal(RT[jj], ref)),
                         csv8_values_differing=int(sum("%.8f" % u != "%.8f" % v for u, v in zip(RT[jj], ref))))
            real.append(r)
        if "x_ens_r201_bqda_to" in pl and f"ref__r201_qda_block" in Z.files:
            ens = RT[pl.index("x_ens_r201_bqda_to")]
            avg = 0.5 * (Z["ref__r201_qda_block"] + Z["ref__r255_bqda_eb_trainonly"])
            real.append({"proc": "x_ens_r201_bqda_to vs 0.5*(archived r201 + archived r255_trainonly)", "run": e["run_id"],
                         "max_abs_diff": float(np.abs(ens - avg).max()), "bit_identical": bool(np.array_equal(ens, avg))})
        out["real_eb_choice"] = e.get("real_eb_choice")
    real = pd.DataFrame(real)
    run.save_table("real_check", real)
    real.to_csv(tables / "r404_real_check.csv", index=False)
    print(chk.groupby("proc")[["max_abs_diff_private", "max_abs_diff_public", "share_identical"]].agg(["max", "min"]).to_string(), flush=True)
    print(real.to_string(index=False), flush=True)

    # ---------------------------------------------------------------- (a) private AUC
    rows, lvl = [], []
    for wn, w in WEIGHTS.items():
        for k in procs:
            j = J[k]
            v = wl(PRIV[..., j], w)                                   # (n, D)
            dr = wl(PRIV[..., j] - PRIV[..., jr], w)
            ds = wl(PRIV[..., j] - PRIV[..., j0], w)
            beat = wl((PRIV[..., j] > PRIV[..., jr]).astype(float), w)
            tie = wl((PRIV[..., j] == PRIV[..., jr]).astype(float), w)
            beat0 = wl((PRIV[..., j] > PRIV[..., j0]).astype(float), w)
            ex = wl(EXP[:, :, j, 1], w)                                # variance-reduced expected AUC, CSV scores
            exd = wl(EXP[:, :, j, 1] - EXP[:, :, jr, 1], w)
            reg = wl(PRIV[:, :, 0, j] - PRIV[:, :, 0, jb], w)
            rows.append({"weighting": wn, "proc": k, "inductive": k in INDUCTIVE, "n_replicas": n,
                         "private_auc_mean": float(v.mean()), "private_auc_sd_draw0": float(v[:, 0].std(ddof=1)),
                         "expected_auc_mean": float(ex.mean()),
                         "diff_vs_r201_mean": float(dr.mean()), "diff_vs_r201_mc_se": se(dr.mean(1)),
                         "diff_vs_r201_q05": float(np.quantile(dr, 0.05)), "diff_vs_r201_q95": float(np.quantile(dr, 0.95)),
                         "expected_diff_vs_r201": float(exd.mean()), "expected_diff_vs_r201_mc_se": se(exd),
                         "p_beats_r201": float(beat.mean()), "p_beats_r201_mc_se": se(beat.mean(1)),
                         "p_ties_r201": float(tie.mean()), "p_loses_r201": float(1 - beat.mean() - tie.mean()),
                         "share_replicas_expected_above_r201": float((exd > 0).mean()),
                         "diff_vs_s003_mean": float(ds.mean()), "diff_vs_s003_mc_se": se(ds.mean(1)),
                         "p_beats_s003": float(beat0.mean()), "regret_vs_bayes": float(reg.mean()), "regret_mc_se": se(reg)})
    for li, f in enumerate(flips):
        for k in procs:
            j = J[k]
            dr = PRIV[:, li, :, j] - PRIV[:, li, :, jr]
            lvl.append({"flip_y": f, "proc": k, "private_auc_mean": float(PRIV[:, li, :, j].mean()),
                        "private_auc_sd_draw0": float(PRIV[:, li, 0, j].std(ddof=1)),
                        "diff_vs_r201_mean": float(dr.mean()), "diff_vs_r201_mc_se": se(dr.mean(1)),
                        "expected_diff_vs_r201": float((EXP[:, li, j, 1] - EXP[:, li, jr, 1]).mean()),
                        "p_beats_r201": float((dr > 0).mean()), "p_ties_r201": float((dr == 0).mean()),
                        "diff_vs_s003_mean": float((PRIV[:, li, :, j] - PRIV[:, li, :, j0]).mean())})
    S, LV = pd.DataFrame(rows), pd.DataFrame(lvl)
    for nm, df in (("private", S), ("by_level", LV)):
        run.save_table(nm, df)
        df.to_csv(tables / f"r404_{nm}.csv", index=False)
    show = ["weighting", "proc", "private_auc_mean", "diff_vs_r201_mean", "diff_vs_r201_mc_se", "expected_diff_vs_r201",
            "expected_diff_vs_r201_mc_se", "p_beats_r201", "p_ties_r201", "diff_vs_s003_mean", "regret_vs_bayes"]
    (tables / "r404_private.md").write_text(S[show].to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # pairs (both final selections inductive; Kaggle scores the better one)
    pr = []
    for wn, w in WEIGHTS.items():
        means = {k: float(wl(PRIV[..., J[k]], w).mean()) for k in INDUCTIVE}
        for ia, ka in enumerate(INDUCTIVE):
            for kb in INDUCTIVE[ia:]:
                A_, B_ = PRIV[..., J[ka]], PRIV[..., J[kb]]
                mx = np.maximum(A_, B_)
                g = wl(mx - PRIV[..., jr], w).mean(1)
                best = ka if means[ka] >= means[kb] else kb
                gb = wl(mx - PRIV[..., J[best]], w).mean(1)
                pb = wl((mx > PRIV[..., jr]).astype(float), w).mean(1)
                pr.append({"weighting": wn, "a": ka, "b": kb, "e_max": float(wl(mx, w).mean()),
                           "gain_vs_r201_alone": float(g.mean()), "gain_vs_r201_mc_se": se(g),
                           "better_member": best, "gain_vs_better_member": float(gb.mean()), "gain_vs_better_mc_se": se(gb),
                           "p_pair_beats_r201_alone": float(pb.mean())})
    PR = pd.DataFrame(pr).sort_values(["weighting", "gain_vs_r201_alone"], ascending=[True, False])
    run.save_table("pairs", PR)
    PR.to_csv(tables / "r404_pairs.csv", index=False)
    (tables / "r404_pairs.md").write_text(PR.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # public predictive given S003's real public score (ABC, r936 method: 4 R156 levels, train-only flip prior)
    lv4 = [flips.index(f) for f in W_R156]
    prior = np.array([(1 - flips[l] / 2) ** 250 for l in lv4])
    prior = prior / prior.sum()
    PB, PV = PUB[:, lv4], PRIV[:, lv4]
    nearp = np.abs(PB[..., j0] - a.s003_public) <= a.abc_window              # (n, 4, D)
    wt = np.broadcast_to(prior[None, :, None], nearp.shape)[nearp]
    W = wt / wt.sum()
    ab = []
    for k in procs:
        j = J[k]
        pb_, db_ = PB[..., j][nearp], (PB[..., j] - PB[..., j0])[nearp]
        pv_ = PV[..., j][nearp]
        m = float((W * pb_).sum())
        ab.append({"proc": k, "n_accepted": int(nearp.sum()), "n_replicas_accepted": int(nearp.any(axis=(1, 2)).sum()),
                   "public_mean": m, "public_pred_sd": float(np.sqrt((W * (pb_ - m) ** 2).sum())),
                   "public_q10": float(np.quantile(pb_, 0.1)), "public_q90": float(np.quantile(pb_, 0.9)),
                   "public_diff_vs_s003": float((W * db_).sum()), "p_public_higher_than_s003": float((W * (db_ > 0)).sum()),
                   "p_public_equal_s003": float((W * (db_ == 0)).sum()), "private_mean_given_public": float((W * pv_).sum())})
    AB = pd.DataFrame(ab)
    run.save_table("public_abc", AB)
    AB.to_csv(tables / "r404_public_abc.csv", index=False)

    # ---------------------------------------------------------------- (b) boundary rows
    bd = []
    for k in procs:
        j = J[k]
        r = {"proc": k, "inductive": k in INDUCTIVE, "n_replicas": int(np.isfinite(BND[:, j, 0]).sum()),
             "n_bnd_rows_mean": float(BND[:, j, 4].mean()), "n_s003bnd_rows_mean": float(BND[:, j, 5].mean())}
        for c, nm in ((0, "bnd_auc_clean"), (1, "bnd_expected_auc"), (2, "bnd_kendall_tau"), (3, "s003bnd_auc_clean")):
            x = BND[:, j, c]
            ok = np.isfinite(x) & np.isfinite(BND[:, j0, c])
            r[f"{nm}_mean"] = float(np.nanmean(x))
            r[f"{nm}_mc_se"] = se(x[np.isfinite(x)])
            d = BND[ok, j0, c] - x[ok]
            r[f"{nm}_s003_minus_proc"] = float(d.mean())
            r[f"{nm}_s003_minus_proc_mc_se"] = se(d)
            r[f"{nm}_share_s003_better"] = float((d > 0).mean())
            r[f"{nm}_share_s003_equal"] = float((d == 0).mean())
        bd.append(r)
    BD = pd.DataFrame(bd)
    run.save_table("boundary", BD)
    BD.to_csv(tables / "r404_boundary.csv", index=False)
    bshow = ["proc", "n_replicas", "n_bnd_rows_mean", "bnd_auc_clean_mean", "bnd_auc_clean_mc_se", "bnd_auc_clean_s003_minus_proc",
             "bnd_auc_clean_s003_minus_proc_mc_se", "bnd_auc_clean_share_s003_better", "bnd_expected_auc_mean",
             "bnd_expected_auc_s003_minus_proc", "bnd_expected_auc_s003_minus_proc_mc_se", "bnd_kendall_tau_mean",
             "s003bnd_auc_clean_mean", "s003bnd_auc_clean_s003_minus_proc", "s003bnd_auc_clean_share_s003_better"]
    (tables / "r404_boundary.md").write_text(BD[bshow].to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # EB / reg choices on the replicas
    from collections import Counter
    eb = {kk: sum((Counter(m["eb_choice"][kk]) for m in metas), Counter()) for kk in ("eb_trainonly", "eb_trainonly_wk")}
    ebrows = [{"variant": kk, "kappa0|nu0_extra|prior": c, "n_folds": v, "share": v / sum(eb[kk].values())}
              for kk in eb for c, v in eb[kk].most_common()]
    EB = pd.DataFrame(ebrows)
    kap = []
    for kk in eb:
        tot = Counter()
        for c, v in eb[kk].items():
            tot[float(c.split("|")[0])] += v
        s_ = sum(tot.values())
        kap += [{"variant": kk, "kappa0": kv, "n_folds": v, "share": v / s_} for kv, v in sorted(tot.items())]
    KAP = pd.DataFrame(kap)
    regs = {kk: sum((Counter(m["chosen_reg"][kk]) for m in metas), Counter())
            for kk in ("r201_qda_block", "r210_qda_block_nll", "x_qda_block_nll_ext")}
    REGT = pd.DataFrame([{"proc": kk, "reg": float(r_), "n_folds": v, "share": v / sum(regs[kk].values())}
                         for kk in regs for r_, v in sorted(regs[kk].items(), key=lambda t: float(t[0]))])
    for nm, df in (("eb_choice", EB), ("eb_kappa", KAP), ("reg_choice", REGT)):
        run.save_table(nm, df)
        df.to_csv(tables / f"r404_{nm}.csv", index=False)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    showp = [k for k in INDUCTIVE if k != BASE] + ["s003_gmm2_hv"]
    for i, k in enumerate(showp):
        dd = wl(PRIV[..., J[k]] - PRIV[..., jr], w156).mean(1)
        jit = (np.random.default_rng(i).random(len(dd)) - 0.5) * 0.5
        ax.scatter(np.full(len(dd), i) + jit, dd, s=6, alpha=0.35, color="#2a78d6")
        ax.plot([i - 0.3, i + 0.3], [dd.mean()] * 2, color="#eb6834", lw=2)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(range(len(showp)))
    ax.set_xticklabels(showp, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_ylabel("private AUC minus r201 (per replica)")
    ax.set_title(f"(a) strictly inductive candidates vs r201 ({n} replicas, CSV scores, R156)", fontsize=9)
    ax = axes[1]
    order = ["bayes", "s003_gmm2_hv", "r209_selftrain_qda_lw", "r255_bqda_eb_hv", "x_qda_block_nll_ext",
             "r210_qda_block_nll", "r201_qda_block", "r255_bqda_eb_trainonly", "r208_qda_block_lw"]
    vals = [BND[:, J[k], 0] for k in order]
    ax.boxplot(vals, showfliers=False)
    ax.plot(range(1, len(order) + 1), [np.nanmean(v) for v in vals], "o", color="#eb6834", ms=4)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC on Bayes-boundary rows (clean labels)")
    ax.set_title(f"(b) rows with Bayes posterior in (0.1, 0.9), {n} replicas; orange = mean", fontsize=9)
    for axx in axes:
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "r404_inductive_boundary.png", dpi=150)
    run.save_figure("inductive_boundary", fig)
    plt.close(fig)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.6g}".format):
        print(wtab.to_string(index=False))
        print(S[show].to_string(index=False))
        print(PR.groupby("weighting").head(8).to_string(index=False))
        print(AB.to_string(index=False))
        print(BD[bshow].to_string(index=False))
        print(KAP.to_string(index=False))
    best = {wn: PR[PR.weighting == wn].iloc[0][["a", "b", "gain_vs_r201_alone", "gain_vs_r201_mc_se",
                                                   "gain_vs_better_member", "gain_vs_better_mc_se"]].to_dict()
            for wn in WEIGHTS}
    out.update({"n_replicas": int(n), "seed_range": [int(SEEDS.min()), int(SEEDS.max())], "best_pair": best,
                "weights_flat_R922": dict(zip(map(str, flips), wflat.round(6).tolist()))})
    run.finish(extra_summary=out)
