"""R936 - final-selection analysis of the 8 candidates on the pooled r933 replicas (verifier "cand").

Input: research/r935_r933_pooled.npz (packed copy of every finished r933 run; sha256 of each source array in its
manifest). Two groups of replicas (parametric bootstrap of the real data, research/lib_gen_replica.py):
  tn1  train AND test labels flipped (sklearn flip_y semantics), raw scores                  (r933 default)
  tn0  test labels only (train labels clean, as on the real data: GMM CV 1.0 in 100/100 folds),
       raw scores AND scores formatted like the submission CSV ("%.8f")                       (r933 --train-noise 0 --round-decimals 8)
Label-noise levels flip_y 0 / 0.0025 / 0.005 / 0.01 (~0 / 0.125 / 0.25 / 0.5 % of labels changed), combined with the
R156 posterior weights 0.0773 / 0.4069 / 0.3608 / 0.1550 (run 20261022-081314_r156_flip_posterior, which already uses
both the train-label evidence and S003's public 0.99923).

Outputs (tables r936_*.csv in reports/tables/raw and in the run folder):
  private      per group / score type / level: private-AUC mean and sd, paired diff vs S003 (mean, MC SE over replicas,
               sd, 5-95 %), P(beat / tie / lose S003) with MC SE, hedge value E[max(0, X - S003)] with MC SE, regret vs
               Bayes; for tn1 also the train-clean subset (replica-levels with no flipped train label)
  pairs        E[max(A, B)] - E[S003] for every pair of candidates (Kaggle scores the better of the 2 final selections)
  corr         correlation of private AUCs across replicas x draws within a level, correlation of their differences
               to the Bayes AUC (regret), and sd of the pairwise difference
  public_abc   public-LB predictive given S003's real public 0.99923 (tn0, CSV-formatted scores; ABC window +-w, levels
               weighted by the train-only flip posterior of R156 = (1 - f/2)^250 normalised): E[public], predictive
               sd, cluster MC SE, P(public > S003), and the matching private expectations
  choice       per candidate as the 2nd final selection next to S003: expected gain, P(gain > 0), MC SE
  cv_calib     (tn0 runs made with --cv-stats) official-CV fold AUCs of every procedure on the replicas' clean training
               labels vs the REAL CV of the archived runs (folds < 1, sum of 1 - AUC), and the real CV recomputed on the
               VM by r933 (must equal the archived fold_scores.csv)
  cv_power     how often the official-CV paired test (Nadeau-Bengio corrected t, 5x20, n_val/n_train = 50/200)
               would call each candidate "indistinguishable" from S003 on replicas where its true private AUC is known
               to be lower: share of replicas with p >= 0.05, CV paired mean vs the true clean-label gap
  cv_cond      private-AUC differences vs S003 (CSV scores, R156-weighted) on the replicas whose CV looks like the
               real CV: every GMM-family / BQDA / self-training candidate perfect in 100/100 folds, r210 <= 1 fold below
               1.0, r201 <= 4 folds below 1.0 (ABC on the CV evidence)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vcand --high-mem -- research/r936_final_selection.py
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
from scipy import stats  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--npz", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "r935_r933_pooled.npz"))
p.add_argument("--s003-public", type=float, default=0.99923)
p.add_argument("--abc-windows", type=float, nargs="*", default=[0.00015, 0.0003])
a = p.parse_args()

REF = "s003_gmm2_hv"
W_R156 = {0.0: 0.0773, 0.0025: 0.4069, 0.005: 0.3608, 0.01: 0.1550}
CANDS = ["s003_gmm2_hv", "r105_gmm2_block_mle", "r155_gmm2_inf_tight", "r102_semisup_eps01", "r255_bqda_eb_hv",
         "r209_selftrain_qda_lw", "r210_qda_block_nll", "r201_qda_block"]
POSTHOC = ["x_qda_block_nll_ext"]

blob = Path(a.npz).read_bytes()
Z = np.load(io.BytesIO(blob))
manifest = json.loads(bytes(Z["manifest_json"]).decode("utf-8"))


def load_group(train_noise: bool):
    out = {"runs": [], "priv": [], "pub": [], "extra": [], "seeds": [], "nft": [], "nfte": [], "priv8": [], "pub8": [],
           "cv": [], "real_cv": []}
    key = None
    for e in manifest["runs"]:
        if e["train_noise"] != train_noise:
            continue
        k = (e["variant"], tuple(e["flips"]), tuple(e["procedures"]), e["n_draws"])
        if key is None:
            key = k
        elif k != key:
            raise SystemExit(f"inconsistent r933 configs in group train_noise={train_noise}: {k} vs {key}")
        i = e["index"]
        out["runs"].append(e["run_id"])
        for nm, arr in (("priv", "priv_auc"), ("pub", "pub_auc"), ("extra", "extra"), ("seeds", "seeds"),
                        ("nft", "n_flip_train"), ("nfte", "n_flip_test"), ("priv8", "priv_auc_r8"), ("pub8", "pub_auc_r8"),
                        ("cv", "cv_auc"), ("real_cv", "real_cv_auc")):
            if f"{i}__{arr}" in Z.files:
                out[nm].append(Z[f"{i}__{arr}"])
    if key is None:
        return None
    for nm in list(out):
        if nm == "runs":
            continue
        if nm in ("priv8", "pub8", "cv") and len(out[nm]) != len(out["runs"]):
            out[nm] = None
            continue
        if nm == "real_cv":
            out[nm] = np.stack(out[nm]) if out[nm] else None      # (n_runs, procs, folds): one real pass per run
            continue
        out[nm] = np.concatenate(out[nm]) if out[nm] else None
    if len(np.unique(out["seeds"])) != len(out["seeds"]):
        raise SystemExit(f"duplicate seeds in group train_noise={train_noise}")
    out["variant"], out["flips"], out["procs"], out["n_draws"] = key[0], list(key[1]), list(key[2]), key[3]
    return out


groups = {"tn1": load_group(True), "tn0": load_group(False)}
groups = {k: v for k, v in groups.items() if v is not None}
print({k: (len(v["seeds"]), v["runs"]) for k, v in groups.items()}, flush=True)

HYP = ("Pooled replicas, test-label noise only (tn0) and CSV-formatted scores, levels weighted by R156: (1) no candidate "
       "has a positive expected private-AUC difference to S003; the GMM family (r105, r155, r102) is within 1e-6 and ties "
       "S003 exactly in most replicas, so its P(beat) and hedge value E[max(0, X - S003)] are < 0.3 and < 1e-5; "
       "(2) BQDA r255 and self-training r209 lose 5e-5 to 2e-4 and beat S003 in < 15 % of replica-draws; supervised nested "
       "QDA r201 / r210 lose 2e-4 to 4e-4 and beat S003 in < 10 %; (3) private AUCs are correlated > 0.9 within a noise level "
       "(shared test labels) but the sd of the difference to S003 is ~1e-4 for the non-GMM candidates; (4) the best "
       "second selection raises E[max] by < 2e-5; (5) with train-label noise (tn1) the supervised QDAs lose 2-4x more, "
       "which the train-clean subset of tn1 removes; (6) given S003's public 0.99923 the expected public of r201 is "
       "~0.9989 with predictive sd ~0.0008.")

tables = paths.tables_dir(a.data)
figs = paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
cfg = {"name": "r936_final_selection", "input_npz": Path(a.npz).name,
       "input_sha256": hashlib.sha256(blob).hexdigest(), "source_runs": [e["run_id"] for e in manifest["runs"]],
       "weights_r156": W_R156, "s003_public": a.s003_public, "abc_windows": a.abc_windows, "candidates": CANDS,
       "posthoc": POSTHOC}


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def wsum(A, w):
    """Level-weighted combination: A has the level axis at position 1."""
    return np.tensordot(A, w, axes=([1], [0]))


def private_rows(g, PRIV, stype, tag, levels, procs, mask=None):
    """mask: (n, L) bool of usable replica-levels (None = all)."""
    rows = []
    j0, jb = procs.index(REF), procs.index("bayes")
    n = PRIV.shape[0]
    for lname, wl in levels:
        if mask is not None and np.count_nonzero(wl) > 1:
            continue  # weighted combination for masked data is done below from the per-level rows
        li = int(np.argmax(wl)) if np.count_nonzero(wl) == 1 else None
        sel = np.ones(n, bool) if (mask is None or li is None) else mask[:, li]
        P = PRIV[sel]
        for j, k in enumerate(procs):
            if li is None:
                dall = wsum(P[..., j] - P[..., j0], wl)                       # (m, D) weighted over levels
                av = wsum(P[..., j], wl)
                beat = wsum((P[..., j] > P[..., j0]).astype(float), wl)
                tie = wsum((P[..., j] == P[..., j0]).astype(float), wl)
                reg = wsum(P[:, :, 0, j] - P[:, :, 0, jb], wl)
            else:
                dall = P[:, li, :, j] - P[:, li, :, j0]
                av = P[:, li, :, j]
                beat = (dall > 0).astype(float)
                tie = (dall == 0).astype(float)
                reg = P[:, li, 0, j] - P[:, li, 0, jb]
            drep = dall.mean(1)
            hedge = np.maximum(dall, 0).mean(1) if li is not None else wsum(np.maximum(P[..., j] - P[..., j0], 0), wl).mean(1)
            rows.append({"group": tag, "scores": stype, "flip_y": lname, "proc": k, "n_replicas": int(sel.sum()),
                         "private_auc_mean": float(av.mean()), "private_auc_sd": float(av[:, 0].std(ddof=1)),
                         "diff_vs_s003_mean": float(drep.mean()), "diff_mc_se": se(drep),
                         "diff_sd_draw0": float(dall[:, 0].std(ddof=1)),
                         "diff_q05": float(np.quantile(dall, 0.05)), "diff_q95": float(np.quantile(dall, 0.95)),
                         "p_beats_s003": float(beat.mean()), "p_beats_mc_se": se(beat.mean(1)),
                         "p_ties_s003": float(tie.mean()), "p_loses_s003": float(1 - beat.mean() - tie.mean()),
                         "hedge_gain_with_s003": float(hedge.mean()), "hedge_mc_se": se(hedge),
                         "regret_vs_bayes": float(reg.mean()), "regret_mc_se": se(reg)})
    if mask is not None:   # conservative weighted combination of the per-level rows (levels share replicas)
        df = pd.DataFrame(rows)
        w = dict(zip([f"{f:g}" for f in g["flips"]], levels[-1][1]))
        for k in procs:
            sub = df[(df.proc == k) & (df.flip_y.isin(w))].set_index("flip_y")
            ww = np.array([w[f] for f in sub.index])
            r = {"group": tag, "scores": stype, "flip_y": "weighted_r156", "proc": k,
                 "n_replicas": int(sub.n_replicas.min())}
            for c in ("private_auc_mean", "diff_vs_s003_mean", "p_beats_s003", "p_ties_s003", "p_loses_s003",
                      "hedge_gain_with_s003", "regret_vs_bayes"):
                r[c] = float((sub[c].to_numpy() * ww).sum())
            for c in ("diff_mc_se", "p_beats_mc_se", "hedge_mc_se", "regret_mc_se"):
                r[c] = float((sub[c].to_numpy() * ww).sum())    # upper bound (perfect correlation)
            rows.append(r)
    return rows


def pair_rows(PRIV, stype, tag, levels, procs):
    rows = []
    j0 = procs.index(REF)
    cand = [k for k in procs if k != "bayes"]
    for lname, wl in levels:
        for i, ka in enumerate(cand):
            for kb in cand[i:]:
                A_, B_ = PRIV[..., procs.index(ka)], PRIV[..., procs.index(kb)]
                g_ = wsum(np.maximum(A_, B_) - PRIV[..., j0], wl).mean(1)
                pb = wsum((np.maximum(A_, B_) > PRIV[..., j0]).astype(float), wl).mean(1)
                rows.append({"group": tag, "scores": stype, "flip_y": lname, "a": ka, "b": kb,
                             "gain_vs_s003_alone": float(g_.mean()), "gain_mc_se": se(g_),
                             "p_pair_beats_s003_alone": float(pb.mean())})
    return rows


def corr_rows(PRIV, stype, tag, flips, procs):
    rows = []
    jb = procs.index("bayes")
    for li, f in enumerate(flips):
        X = PRIV[:, li].reshape(-1, len(procs))             # replicas x draws pooled
        Rg = X - X[:, [jb]]
        C1 = np.corrcoef(X.T)
        with np.errstate(invalid="ignore", divide="ignore"):
            C2 = np.corrcoef(Rg.T)
        for i, ka in enumerate(procs):
            for j, kb in enumerate(procs):
                if j <= i:
                    continue
                d = X[:, i] - X[:, j]
                rows.append({"group": tag, "scores": stype, "flip_y": f"{f:g}", "a": ka, "b": kb,
                             "corr_private_auc": float(C1[i, j]), "corr_regret_vs_bayes": float(C2[i, j]),
                             "sd_private_auc_a": float(X[:, i].std()), "sd_diff": float(d.std()),
                             "p_a_gt_b": float((d > 0).mean()), "p_equal": float((d == 0).mean())})
    return rows


def abc_rows(g, PUB, PRIV, stype, tag, window):
    procs, flips = g["procs"], g["flips"]
    j0 = procs.index(REF)
    prior = np.array([(1 - f / 2) ** 250 for f in flips])
    prior = prior / prior.sum()                                  # = R156 train-only posterior 0.39/0.29/0.21/0.11
    near = np.abs(PUB[..., j0] - a.s003_public) <= window       # (n, L, D)
    acc = near.mean(axis=(0, 2))
    post = prior * acc
    post = post / post.sum() if post.sum() > 0 else post
    # every level has the same number of simulated (replica, draw) pairs, so weighting each accepted pair by its
    # level's prior gives the posterior prior x acceptance rate
    wt = np.broadcast_to(prior[None, :, None], near.shape)[near]
    n = PUB.shape[0]
    rid = np.broadcast_to(np.arange(n)[:, None, None], near.shape)[near]
    rows = []
    for j, k in enumerate(procs):
        pb, pv = PUB[..., j][near], PRIV[..., j][near]
        db = (PUB[..., j] - PUB[..., j0])[near]
        dv = (PRIV[..., j] - PRIV[..., j0])[near]
        W = wt / wt.sum()

        def wm(x):
            return float((W * x).sum())

        def wsd(x):
            m = wm(x)
            return float(np.sqrt(max((W * (x - m) ** 2).sum(), 0)))
        df = pd.DataFrame({"r": rid, "w": wt, "pb": pb * wt, "db": db * wt, "dv": dv * wt})
        per = df.groupby("r").sum()
        # cluster (replica) SE of a weighted ratio estimator
        tot = per["w"].sum()
        def cse(col):
            m = per[col].sum() / tot
            resid = per[col] - m * per["w"]
            return float(np.sqrt((resid ** 2).sum()) / tot)
        rows.append({"group": tag, "scores": stype, "window": window, "proc": k, "n_accepted": int(near.sum()),
                     "n_replicas_accepted": int(len(per)),
                     "posterior_levels": json.dumps(dict(zip([f"{f:g}" for f in flips], np.round(post, 4).tolist()))),
                     "public_mean": wm(pb), "public_pred_sd": wsd(pb), "public_mc_se": cse("pb"),
                     "public_diff_vs_s003": wm(db), "public_diff_sd": wsd(db), "public_diff_mc_se": cse("db"),
                     "p_public_higher": wm((db > 0).astype(float)), "p_public_equal": wm((db == 0).astype(float)),
                     "private_mean": wm(pv), "private_pred_sd": wsd(pv),
                     "private_diff_vs_s003": wm(dv), "private_diff_mc_se": cse("dv"),
                     "p_private_beats_s003": wm((dv > 0).astype(float)), "p_private_ties_s003": wm((dv == 0).astype(float))})
    return rows


# real CV of the archived candidate runs (fold_scores.csv, official 5x20): folds below 1.0 and sum of (1 - AUC)
REAL_CV = {"s003_gmm2_hv": (0, 0.0), "r105_gmm2_block_mle": (0, 0.0), "r155_gmm2_inf_tight": (0, 0.0),
           "r102_semisup_eps01": (0, 0.0), "r255_bqda_eb_hv": (0, 0.0), "r209_selftrain_qda_lw": (0, 0.0),
           "r210_qda_block_nll": (1, 0.00162), "r201_qda_block": (4, 0.00971)}
HYP += (" (7) CV calibration: on replicas the supervised QDAs show MORE folds below 1.0 than on the real data (real: "
        "r210 1, r201 4), i.e. the parametric replica is somewhat harder for them; conditioning on the real CV pattern "
        "shrinks their private loss vs S003 but leaves it negative.")


def cv_tables(g, PRIV, procs, flips):
    cvA = g["cv"]                                   # (n, P, F)
    nb = (cvA < 1).sum(2)                           # folds below 1
    disc = (1 - cvA).sum(2)
    rows = []
    for j, k in enumerate(procs):
        if k == "bayes":
            continue
        rn, rd = REAL_CV.get(k, (np.nan, np.nan))
        r = {"proc": k, "replica_cv_mean": float(np.nanmean(cvA[:, j])), "replica_folds_below1_mean": float(nb[:, j].mean()),
             "replica_folds_below1_median": float(np.median(nb[:, j])), "replica_sum_1mauc_mean": float(disc[:, j].mean()),
             "real_folds_below1": rn, "real_sum_1mauc": rd,
             "p_replica_folds_le_real": float((nb[:, j] <= rn).mean()) if np.isfinite(rn) else np.nan}
        if g["real_cv"] is not None:
            rc = g["real_cv"][:, j]                       # (n_runs, F)
            r["real_cv_recomputed_folds_below1"] = int((rc[0] < 1).sum())
            r["real_cv_recomputed_sum_1mauc"] = float((1 - rc[0]).sum())
            r["real_cv_recomputed_runs_identical"] = bool(np.all(rc == rc[0]))
        rows.append(r)
    j = {k: procs.index(k) for k in procs}
    perfect = np.all([nb[:, j[k]] == 0 for k in ("s003_gmm2_hv", "r105_gmm2_block_mle", "r155_gmm2_inf_tight",
                                                  "r102_semisup_eps01", "r255_bqda_eb_hv", "r209_selftrain_qda_lw")], axis=0)
    conds = {"all": np.ones(len(nb), bool), "transductive_all_perfect": perfect,
             "r210_le1": nb[:, j["r210_qda_block_nll"]] <= 1, "r201_le4": nb[:, j["r201_qda_block"]] <= 4}
    conds["real_pattern"] = conds["transductive_all_perfect"] & conds["r210_le1"] & conds["r201_le4"]
    w = np.array([W_R156[f] for f in flips])
    w = w / w.sum()
    j0 = procs.index(REF)
    crow = []
    for cn, m in conds.items():
        for jj, k in enumerate(procs):
            if m.sum() < 2:
                crow.append({"condition": cn, "n_replicas": int(m.sum()), "proc": k})
                continue
            P = PRIV[m]
            d = wsum(P[..., jj] - P[..., j0], w)                    # (m, D)
            beat = wsum((P[..., jj] > P[..., j0]).astype(float), w)
            hed = wsum(np.maximum(P[..., jj] - P[..., j0], 0), w).mean(1)
            crow.append({"condition": cn, "n_replicas": int(m.sum()), "share": float(m.mean()), "proc": k,
                         "diff_vs_s003_mean": float(d.mean()), "diff_mc_se": se(d.mean(1)),
                         "p_beats_s003": float(beat.mean()), "hedge_gain_with_s003": float(hed.mean()), "hedge_mc_se": se(hed),
                         "clean_label_diff": float((g["extra"][m][:, 0, jj, 0] - g["extra"][m][:, 0, j0, 0]).mean()),
                         "clean_label_diff_mc_se": se(g["extra"][m][:, 0, jj, 0] - g["extra"][m][:, 0, j0, 0])})
    # power of the official-CV comparison: corrected resampled t-test of the fold-paired AUC difference vs S003
    prow = []
    nf = cvA.shape[2]
    corr = 1.0 / nf + 50.0 / 200.0
    for jj, k in enumerate(procs):
        if k in ("bayes", REF):
            continue
        d = cvA[:, jj] - cvA[:, j0]                                # (n, F)
        mu = d.mean(1)
        sd = d.std(1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sd > 0, mu / np.sqrt(corr * sd ** 2), 0.0)
        pval = 2 * stats.t.sf(np.abs(t), df=nf - 1)
        true_gap = g["extra"][:, 0, jj, 0] - g["extra"][:, 0, j0, 0]     # clean-label private AUC difference
        prow.append({"proc": k, "n_replicas": int(len(d)), "cv_paired_mean": float(mu.mean()),
                     "true_clean_gap_mean": float(true_gap.mean()), "share_true_gap_negative": float((true_gap < 0).mean()),
                     "share_cv_not_significant": float((pval >= 0.05).mean()),
                     "share_cv_mean_exactly_0": float((mu == 0).mean()),
                     "share_cv_never_better_in_any_fold": float((d.max(1) <= 0).mean()),
                     "share_not_significant_given_true_gap_negative":
                         float((pval[true_gap < 0] >= 0.05).mean()) if (true_gap < 0).any() else np.nan})
    return rows, crow, prow


with Run("r936_final_selection", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    CAL, CND, POW = [], [], []
    PR, PA, CO, AB = [], [], [], []
    meta = {"groups": {}}
    for tag, g in groups.items():
        procs, flips = g["procs"], g["flips"]
        w = np.array([W_R156[f] for f in flips])
        w = w / w.sum()
        levels = [(f"{f:g}", np.eye(len(flips))[li]) for li, f in enumerate(flips)] + [("weighted_r156", w)]
        meta["groups"][tag] = {"n_replicas": int(len(g["seeds"])), "runs": g["runs"],
                               "seed_range": [int(g["seeds"].min()), int(g["seeds"].max())],
                               "mean_n_flip_test": dict(zip(map(str, flips), g["nfte"].mean(0).round(2).tolist())),
                               "share_train_clean": dict(zip(map(str, flips), (g["nft"] == 0).mean(0).round(3).tolist()))}
        score_sets = [("raw", g["priv"], g["pub"])] + ([("csv8", g["priv8"], g["pub8"])] if g["priv8"] is not None else [])
        for stype, PRIV, PUB in score_sets:
            PR += private_rows(g, PRIV, stype, tag, levels, procs)
            PA += pair_rows(PRIV, stype, tag, levels, procs)
            CO += corr_rows(PRIV, stype, tag, flips, procs)
            if tag == "tn1":
                PR += private_rows(g, PRIV, stype, "tn1_trainclean_subset", levels, procs, mask=(g["nft"] == 0))
            if tag == "tn0":
                for wdw in a.abc_windows:
                    AB += abc_rows(g, PUB, PRIV, stype, tag, wdw)
                if g["cv"] is not None and stype == ("csv8" if g["priv8"] is not None else "raw"):
                    cal, cnd, pw = cv_tables(g, PRIV, procs, flips)
                    CAL += cal
                    POW += pw
                    CND += [dict(r, scores=stype) for r in cnd]

    S, H, Cc, A = pd.DataFrame(PR), pd.DataFrame(PA), pd.DataFrame(CO), pd.DataFrame(AB)
    H = H.sort_values(["group", "scores", "flip_y", "gain_vs_s003_alone"], ascending=[True, True, True, False])
    # second-selection choice table (primary: tn0, CSV scores if present, weighted levels)
    prim_scores = "csv8" if (S.group.eq("tn0") & S.scores.eq("csv8")).any() else "raw"
    prim_group = "tn0" if S.group.eq("tn0").any() else "tn1"
    ch = H[(H.group == prim_group) & (H.scores == prim_scores) & (H.flip_y == "weighted_r156") & (H.a == REF)].copy()
    ch = ch.rename(columns={"b": "second_selection"})[["second_selection", "gain_vs_s003_alone", "gain_mc_se",
                                                       "p_pair_beats_s003_alone"]]
    meta["primary"] = {"group": prim_group, "scores": prim_scores}
    CAL, CND, POW = pd.DataFrame(CAL), pd.DataFrame(CND), pd.DataFrame(POW)
    for nm, df in (("cv_calib", CAL), ("cv_cond", CND), ("cv_power", POW)):
        if len(df):
            run.save_table(nm, df)
            df.to_csv(tables / f"r936_{nm}.csv", index=False)
            print(df.to_string(index=False), flush=True)
    for nm, df in (("private", S), ("pairs", H), ("corr", Cc), ("public_abc", A), ("choice", ch)):
        run.save_table(nm, df)
        df.to_csv(tables / f"r936_{nm}.csv", index=False)
    (tables / "r936_choice.md").write_text(ch.to_markdown(index=False, floatfmt=".3g"), encoding="utf-8")
    keep = S[(S.flip_y == "weighted_r156")]
    (tables / "r936_private_weighted.md").write_text(keep.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")
    if len(A):
        (tables / "r936_public_abc.md").write_text(A.to_markdown(index=False, floatfmt=".5g"), encoding="utf-8")

    # figure: per-replica paired private difference vs S003 (primary group, weighted levels, all draws averaged)
    g = groups[prim_group]
    procs = g["procs"]
    PRIV = g["priv8"] if prim_scores == "csv8" else g["priv"]
    w = np.array([W_R156[f] for f in g["flips"]])
    w = w / w.sum()
    j0 = procs.index(REF)
    show = [k for k in procs if k != REF]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(0, color="#888", lw=0.8)
    for i, k in enumerate(show):
        dd = wsum(PRIV[..., procs.index(k)] - PRIV[..., j0], w).mean(1)
        jit = (np.random.default_rng(i).random(len(dd)) - 0.5) * 0.5
        ax.scatter(np.full(len(dd), i) + jit, dd, s=6, alpha=0.35, color="#0b0b0b" if k == "bayes" else "#2a78d6")
        ax.plot([i - 0.3, i + 0.3], [dd.mean()] * 2, color="#eb6834", lw=2)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(show, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_ylabel("private AUC minus S003 (per replica)")
    ax.set_title(f"r936: paired private-AUC difference vs S003 ({prim_group}, {prim_scores} scores, n={PRIV.shape[0]} "
                 "replicas, R156-weighted label noise); orange = mean", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "r936_private_diff.png", dpi=150)
    run.save_figure("private_diff", fig)
    plt.close(fig)

    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.6g}".format):
        print(S[S.flip_y == "weighted_r156"].to_string(index=False))
        print(ch.to_string(index=False))
        print(H[(H.flip_y == "weighted_r156")].groupby(["group", "scores"]).head(8).to_string(index=False))
        if len(A):
            print(A.to_string(index=False))
    meta["manifest_runs"] = [{k: e[k] for k in ("run_id", "n_replicas", "train_noise", "round_decimals")}
                             for e in manifest["runs"]]
    run.finish(extra_summary=meta)
