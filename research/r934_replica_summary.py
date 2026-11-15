"""R934 - pooled analysis of the r933 replica runs (verifier "cand"): joint distribution of the PRIVATE AUC of the
final candidates, paired differences vs S003, hedge value of a second final selection, public-LB expectations.

Inputs: every *_r933_replicas_* run folder found in this Colab session's earlier job archives
(/content/jobs/<job>/archive/experiments, i.e. the parent of this job's SSC_ARCHIVE_ROOT), in the local archive,
and in --search folders. Arrays per run: priv_auc / pub_auc (seeds x levels x draws x procs), extra
(seeds x levels x procs x [clean-label private AUC, expected private AUC under the noisy Bayes posterior,
Kendall with Bayes on Bayes-boundary rows, Kendall with S003 on Bayes-boundary rows]).

Outputs (per variant; levels = flip_y grid; "weighted" = R156 posterior over flip_y 0/0.0025/0.005/0.01 =
0.0773/0.4069/0.3608/0.1550, run 20261022-081314_r156_flip_posterior):
  r934_private_summary   mean / sd of the private AUC, paired diff vs S003 (mean, MC SE, sd), P(beat / tie / lose),
                         expected-AUC diff (variance reduced), clean-label diff, regret vs Bayes
  r934_hedge_pairs       E[max(A, B)] - E[A_S003] for every pair (Kaggle scores the better of the 2 selections),
                         with MC SE
  r934_corr              correlation of private AUCs and of regrets (AUC - Bayes AUC) across replicas
  r934_public            public-LB paired difference vs S003 over all draws (mean, sd, P(>0/=0/<0)), and the
                         ABC posterior predictive given S003's real public 0.99923 (uniform prior over the levels)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vcand --high-mem -- research/r934_replica_summary.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--search", nargs="*", default=[])
p.add_argument("--s003-public", type=float, default=0.99923)
p.add_argument("--abc-window", type=float, default=0.00015)
a = p.parse_args()

W_R156 = {0.0: 0.0773, 0.0025: 0.4069, 0.005: 0.3608, 0.01: 0.1550}
REF = "s003_gmm2_hv"

roots = [Path(s) for s in a.search]
if os.environ.get("SSC_ARCHIVE_ROOT"):
    roots += sorted(Path(os.environ["SSC_ARCHIVE_ROOT"]).resolve().parents[1].glob("*/archive/experiments"))
roots.append(paths.experiments_dir(a.data))
found = {}
for r in roots:
    for d in sorted(r.glob("*_r933_replicas_*")):
        if (d / "priv_auc.npy").exists() and (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            if s.get("status") == "done":
                found.setdefault(d.name, d)
print(f"found {len(found)} r933 runs: {sorted(found)}", flush=True)

groups = {}
for name, d in sorted(found.items()):
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    key = (cfg["variant"], tuple(cfg["flips"]), tuple(cfg["procedures"]), int(cfg["n_draws"]))
    g = groups.setdefault(key, {"runs": [], "seeds": [], "priv": [], "pub": [], "extra": []})
    seeds = np.load(d / "seeds.npy")
    g["runs"].append(name)
    g["seeds"].append(seeds)
    g["priv"].append(np.load(d / "priv_auc.npy"))
    g["pub"].append(np.load(d / "pub_auc.npy"))
    g["extra"].append(np.load(d / "extra.npy"))

HYP = ("Pooled over >= 200 parametric replicas: (1) the GMM family (S003, r105, r155, r102) ties S003 on the private LB "
       "in most replicas and differs by < 1e-6 on average; (2) BQDA / self-training lose 1e-5 to 1e-4, supervised "
       "nested QDA 1e-4 to 1e-3; (3) private AUCs are correlated > 0.9 across procedures (shared label noise) but the "
       "regrets of the GMM family are nearly uncorrelated with the supervised ones; (4) no pair raises E[max] above "
       "S003 alone by more than ~1e-5, so the second selection is insurance against a hidden failure mode, not a "
       "score gain; (5) given S003's public 0.99923 the expected public of a supervised QDA is ~0.999 with sd ~0.001.")
tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r934_replica_summary", "inputs": sorted(found), "weights_r156": W_R156, "s003_public": a.s003_public,
       "abc_window": a.abc_window}


def md(df, path):
    path.write_text(df.to_markdown(index=False, floatfmt=".6g"), encoding="utf-8")


with Run("r934_replica_summary", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    all_summ, all_hedge, all_corr, all_pub, all_abc, meta = [], [], [], [], [], {}
    for (variant, flips, procs, n_draws), g in groups.items():
        seeds = np.concatenate(g["seeds"])
        if len(np.unique(seeds)) != len(seeds):
            raise SystemExit(f"duplicate seeds in variant {variant}: {g['runs']}")
        PRIV = np.concatenate(g["priv"])     # (n, L, D, P)
        PUB = np.concatenate(g["pub"])
        EX = np.concatenate(g["extra"])      # (n, L, P, 4)
        n, nL, nD, nP = PRIV.shape
        procs = list(procs)
        j0, jb = procs.index(REF), procs.index("bayes")
        w = np.array([W_R156.get(f, np.nan) for f in flips])
        has_w = np.isfinite(w).all()
        w = w / w.sum() if has_w else w
        meta[variant] = {"n_replicas": int(n), "runs": g["runs"], "flips": list(flips), "n_draws": int(nD)}
        levels = [(f"{f:g}", np.eye(nL)[li]) for li, f in enumerate(flips)] + ([("weighted_r156", w)] if has_w else [])

        # ---------------- private summary (draw 0 = the replica's own public/private split; all draws for MC means)
        for lname, wl in levels:
            pv0 = np.tensordot(PRIV[:, :, 0, :], wl, axes=([1], [0]))       # (n, P) weighted over levels
            pvD = np.tensordot(PRIV.mean(2), wl, axes=([1], [0]))            # mean over draws
            exp_ = np.tensordot(EX[:, :, :, 1], wl, axes=([1], [0]))
            cln = np.tensordot(EX[:, :, :, 0], wl, axes=([1], [0]))
            # per-replica indicator probabilities, level-weighted
            for j, k in enumerate(procs):
                d0 = PRIV[:, :, 0, j] - PRIV[:, :, 0, j0]                     # (n, L)
                dall = PRIV[:, :, :, j] - PRIV[:, :, :, j0]                   # (n, L, D)
                beat = np.tensordot((dall > 0).mean(2), wl, axes=([1], [0]))
                tie = np.tensordot((dall == 0).mean(2), wl, axes=([1], [0]))
                dw = np.tensordot(dall.mean(2), wl, axes=([1], [0]))
                d0w = np.tensordot(d0, wl, axes=([1], [0]))
                de = exp_[:, j] - exp_[:, j0]
                dc = cln[:, j] - cln[:, j0]
                reg = pv0[:, j] - pv0[:, jb]
                kb = np.tensordot(EX[:, :, j, 2], wl, axes=([1], [0]))
                ks = np.tensordot(EX[:, :, j, 3], wl, axes=([1], [0]))
                all_summ.append({
                    "variant": variant, "flip_y": lname, "proc": k, "n_replicas": n,
                    "private_auc_mean": float(pvD[:, j].mean()), "private_auc_sd_draw0": float(pv0[:, j].std(ddof=1)),
                    "diff_vs_s003_mean": float(dw.mean()), "diff_vs_s003_mc_se": float(dw.std(ddof=1) / np.sqrt(n)),
                    "diff_vs_s003_sd_draw0": float(d0w.std(ddof=1)),
                    "diff_q05_draw0": float(np.quantile(d0w, 0.05)), "diff_q95_draw0": float(np.quantile(d0w, 0.95)),
                    "p_beats_s003": float(beat.mean()), "p_ties_s003": float(tie.mean()),
                    "p_loses_s003": float(1 - beat.mean() - tie.mean()),
                    "p_beats_mc_se": float(beat.std(ddof=1) / np.sqrt(n)),
                    "exp_auc_diff_vs_s003": float(de.mean()), "exp_auc_diff_mc_se": float(de.std(ddof=1) / np.sqrt(n)),
                    "clean_label_diff_vs_s003": float(dc.mean()), "clean_label_diff_mc_se": float(dc.std(ddof=1) / np.sqrt(n)),
                    "regret_vs_bayes_mean": float(reg.mean()), "kendall_bayes_boundary": float(np.nanmean(kb)),
                    "kendall_s003_boundary": float(np.nanmean(ks))})
            # ---------------- hedge: E[max(A, B)] - E[S003] (draws averaged within replica)
            for i, ka in enumerate(procs):
                if ka == "bayes":
                    continue
                for jj, kb_ in enumerate(procs):
                    if jj < i or kb_ == "bayes":
                        continue
                    mx = np.maximum(PRIV[:, :, :, i], PRIV[:, :, :, jj]) - PRIV[:, :, :, j0]
                    g_ = np.tensordot(mx.mean(2), wl, axes=([1], [0]))
                    all_hedge.append({"variant": variant, "flip_y": lname, "a": ka, "b": kb_,
                                      "gain_vs_s003_alone": float(g_.mean()), "gain_mc_se": float(g_.std(ddof=1) / np.sqrt(n)),
                                      "p_pair_beats_s003_alone": float(np.tensordot((mx > 0).mean(2), wl, axes=([1], [0])).mean())})
            # ---------------- correlations across replicas (draw 0)
            C1 = np.corrcoef(pv0.T)
            Rg = pv0 - pv0[:, [jb]]
            with np.errstate(invalid="ignore", divide="ignore"):
                C2 = np.corrcoef(Rg.T)
            for i, ka in enumerate(procs):
                for jj, kb_ in enumerate(procs):
                    if jj <= i:
                        continue
                    all_corr.append({"variant": variant, "flip_y": lname, "a": ka, "b": kb_,
                                     "corr_private_auc": float(C1[i, jj]), "corr_regret_vs_bayes": float(C2[i, jj]),
                                     "sd_diff_draw0": float((pv0[:, i] - pv0[:, jj]).std(ddof=1))})
            # ---------------- public paired differences over all draws
            for j, k in enumerate(procs):
                dp = PUB[:, :, :, j] - PUB[:, :, :, j0]
                mean_ = np.tensordot(dp.mean(2), wl, axes=([1], [0]))
                # pooled sd over replicas x draws, level-weighted second moment
                m2 = np.tensordot((dp ** 2).mean(2), wl, axes=([1], [0])).mean()
                mu = mean_.mean()
                all_pub.append({"variant": variant, "flip_y": lname, "proc": k, "public_diff_mean": float(mu),
                                "public_diff_mc_se": float(mean_.std(ddof=1) / np.sqrt(n)),
                                "public_diff_sd": float(np.sqrt(max(m2 - mu ** 2, 0))),
                                "p_public_higher": float(np.tensordot((dp > 0).mean(2), wl, axes=([1], [0])).mean()),
                                "p_public_equal": float(np.tensordot((dp == 0).mean(2), wl, axes=([1], [0])).mean()),
                                "public_mean": float(np.tensordot(PUB[:, :, :, j].mean(2), wl, axes=([1], [0])).mean()),
                                "public_sd": float(np.sqrt(np.tensordot(PUB[:, :, :, j].var(2), wl, axes=([1], [0])).mean()))})
        # ---------------- ABC: condition on S003's real public score (uniform prior over the levels, all draws)
        near = np.abs(PUB[:, :, :, j0] - a.s003_public) <= a.abc_window          # (n, L, D)
        acc_lvl = near.mean(axis=(0, 2))
        meta[variant]["abc_accept_rate_by_level"] = dict(zip(map(str, flips), acc_lvl.round(5).tolist()))
        meta[variant]["abc_n_accepted"] = int(near.sum())
        post_lvl = acc_lvl / acc_lvl.sum() if acc_lvl.sum() else acc_lvl
        meta[variant]["abc_posterior_levels_public_only"] = dict(zip(map(str, flips), post_lvl.round(4).tolist()))
        for j, k in enumerate(procs):
            pv, pb = PRIV[:, :, :, j][near], PUB[:, :, :, j][near]
            dv = (PRIV[:, :, :, j] - PRIV[:, :, :, j0])[near]
            db = (PUB[:, :, :, j] - PUB[:, :, :, j0])[near]
            # cluster-robust SE: replicas are the independent units
            rep_ids = np.broadcast_to(np.arange(n)[:, None, None], near.shape)[near]
            dfr = pd.DataFrame({"r": rep_ids, "dv": dv, "pb": pb})
            per = dfr.groupby("r").mean()
            nr = len(per)
            all_abc.append({"variant": variant, "proc": k, "n_accepted": int(near.sum()), "n_replicas_accepted": int(nr),
                            "private_mean_given_s003_public": float(pv.mean()), "private_sd": float(pv.std(ddof=1)),
                            "private_diff_vs_s003_mean": float(dv.mean()),
                            "private_diff_se_cluster": float(per["dv"].std(ddof=1) / np.sqrt(nr)) if nr > 1 else np.nan,
                            "p_private_beats_s003": float((dv > 0).mean()), "p_private_ties_s003": float((dv == 0).mean()),
                            "public_mean_given_s003_public": float(pb.mean()), "public_sd_given": float(pb.std(ddof=1)),
                            "public_diff_mean": float(db.mean()), "public_diff_sd": float(db.std(ddof=1)),
                            "public_se_cluster": float(per["pb"].std(ddof=1) / np.sqrt(nr)) if nr > 1 else np.nan})

    S = pd.DataFrame(all_summ)
    H = pd.DataFrame(all_hedge).sort_values(["variant", "flip_y", "gain_vs_s003_alone"], ascending=[True, True, False])
    Cc = pd.DataFrame(all_corr)
    P = pd.DataFrame(all_pub)
    A = pd.DataFrame(all_abc)
    for nm, df in (("private_summary", S), ("hedge_pairs", H), ("corr", Cc), ("public", P), ("abc", A)):
        run.save_table(nm, df)
        df.to_csv(tables / f"r934_{nm}.csv", index=False)
    md(S[S.flip_y.isin(["weighted_r156", "0.01"])], tables / "r934_private_summary.md")
    md(H[H.flip_y == "weighted_r156"].head(40), tables / "r934_hedge_pairs.md")
    md(A, tables / "r934_abc.md")
    md(P[P.flip_y == "weighted_r156"], tables / "r934_public.md")

    # ---------------- figure: paired private-AUC difference vs S003 per replica (weighted levels, draw 0)
    for (variant, flips, procs, n_draws), g in groups.items():
        PRIV = np.concatenate(g["priv"])
        procs = list(procs)
        j0 = procs.index(REF)
        w = np.array([W_R156.get(f, np.nan) for f in flips])
        if not np.isfinite(w).all():
            continue
        w = w / w.sum()
        show = [k for k in procs if k not in (REF,)]
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        data_ = [np.tensordot(PRIV[:, :, 0, procs.index(k)] - PRIV[:, :, 0, j0], w, axes=([1], [0])) for k in show]
        ax.axhline(0, color="#888", lw=0.8)
        for i, dd in enumerate(data_):
            jit = (np.random.default_rng(i).random(len(dd)) - 0.5) * 0.5
            ax.scatter(np.full(len(dd), i) + jit, dd, s=6, alpha=0.35, color="#2a78d6" if show[i] != "bayes" else "#0b0b0b")
            ax.plot([i - 0.3, i + 0.3], [dd.mean()] * 2, color="#eb6834", lw=2)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels(show, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("private AUC minus S003 (per replica)")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_title(f"r934: paired private-AUC difference vs S003, {variant} replicas (n={len(data_[0])}), "
                     "label noise weighted by the R156 posterior; orange = mean", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / f"r934_private_diff_{variant}.png", dpi=150)
        run.save_figure(f"private_diff_{variant}", fig)
        plt.close(fig)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S[S.flip_y == "weighted_r156"].to_string(index=False))
        print(H[H.flip_y == "weighted_r156"].head(15).to_string(index=False))
        print(A.to_string(index=False))
        print(P[P.flip_y == "weighted_r156"].to_string(index=False))
    run.finish(extra_summary={"variants": meta})
