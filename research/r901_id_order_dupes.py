"""R901 - ordering / id / duplicate / file-format leakage probes (verifier `leak`).

Questions (each answered with a number, no model selection):
  1. Does the id column or the row order of train.csv carry label information?
     AUC of id (and of the row position) as a score for y; Wald-Wolfowitz runs test and lag-1
     autocorrelation of y in id order; positive rate per id quintile (chi-square).
  2. Does the id / row order of test.csv carry structure in the S003 predictions?
     S003 = TransductiveGMM("highvar", 2), official CV, fold_mean (recomputed here with the R003 code;
     saved as s003_test_pred.npy for an exact comparison with the archived test_pred.npy).
     Spearman(id, score); predicted-positive rate per id decile; runs test along the file order.
     Plus id vs every feature and vs the label-free mixture posterior on all 20,000 rows.
  3. Exact and near duplicates between train and test (and within train): all 150 columns, the
     13-column high-variance block, the 12 label-free informative columns. Near = nearest-neighbour
     distance (train+test standardisation) compared with the test-to-test nearest-neighbour distance.
  4. Is train a random sample of the 20,000 rows? Adversarial validation (train vs test classifier),
     KS test of the mixture posterior, predicted positive rate on test vs train positive rate.
  5. File-format channel: do the printed decimals / string lengths of the CSV values differ by class?
  6. Label-freeness of the high-variance block: std ratios around the 1.5 x median threshold on
     all / train-only / test-only rows, and where the block columns rank by |corr| with y.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s vleak --high-mem -- research/r901_id_order_dupes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.data import FEATURES  # noqa: E402
from ssc.features import abs_correlation  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import TransductiveGMM, _basis, _mixture, resolve_columns, unlabeled_features  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-ref", type=int, default=3000, help="test rows used for the test-to-test NN reference")
p.add_argument("--n-perm", type=int, default=5000)
a = p.parse_args()
src = a.data


def runs_test(b):
    b = np.asarray(b).astype(int)
    n1, n0 = int(b.sum()), int(len(b) - b.sum())
    n = n1 + n0
    r = 1 + int((b[1:] != b[:-1]).sum())
    mu = 2 * n1 * n0 / n + 1
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1))
    z = (r - mu) / np.sqrt(var)
    return {"runs": r, "expected": float(mu), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def lag1(b):
    b = np.asarray(b, float)
    return float(np.corrcoef(b[1:], b[:-1])[0, 1])


def lag1_perm_p(b, n_perm, seed=0):
    obs = lag1(b)
    rng = np.random.default_rng(seed)
    null = np.array([lag1(rng.permutation(b)) for _ in range(n_perm)])
    return obs, float((1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n_perm))


def bin_rate_test(order_key, b, n_bins):
    q = pd.qcut(pd.Series(order_key).rank(method="first"), n_bins, labels=False)
    tab = pd.crosstab(q, np.asarray(b).astype(int))
    chi2, pv, *_ = stats.chi2_contingency(tab.to_numpy())
    rates = (tab[1] / tab.sum(axis=1)).round(4).tolist() if 1 in tab.columns else []
    return {"rates_by_bin": rates, "chi2": float(chi2), "p": float(pv)}


def auc_and_p(y, s):
    au = roc_auc_score(y, s)
    u = stats.mannwhitneyu(np.asarray(s)[np.asarray(y) == 1], np.asarray(s)[np.asarray(y) == 0])
    return {"auc": float(au), "mw_p": float(u.pvalue)}


def nn_dist(A, B, self_idx=None, chunk=500):
    """Euclidean nearest-neighbour distance (and index) of every row of A among the rows of B.
    self_idx[k] = position of row k of A inside B (excluded), when A is a subset of B."""
    out_d, out_i = np.empty(len(A)), np.empty(len(A), int)
    bb = (B ** 2).sum(1)
    for s in range(0, len(A), chunk):
        Ac = A[s:s + chunk]
        d2 = (Ac ** 2).sum(1)[:, None] + bb[None, :] - 2 * Ac @ B.T
        if self_idx is not None:
            d2[np.arange(len(Ac)), np.asarray(self_idx)[s:s + chunk]] = np.inf
        j = np.argmin(d2, 1)
        out_d[s:s + chunk] = np.sqrt(np.maximum(d2[np.arange(len(Ac)), j], 0))
        out_i[s:s + chunk] = j
    return out_d, out_i


def dup_count(A, B=None):
    """Exact duplicate rows: within A (B None) or rows of A that also occur in B."""
    ka = [r.tobytes() for r in np.ascontiguousarray(A)]
    if B is None:
        return int(len(ka) - len(set(ka)))
    kb = set(r.tobytes() for r in np.ascontiguousarray(B))
    return int(sum(k in kb for k in ka))


hyp = ("No leakage channel outside the features: (1) AUC(id, y) and AUC(row position, y) within 0.5 +/- 0.07 "
       "(2 SE for 113/137 rows), runs test p > 0.01; (2) S003 test scores unrelated to test id / order "
       "(|Spearman| < 0.03, decile chi-square p > 0.01); (3) no exact duplicate rows between train and test and no "
       "train row closer to a test row than the 1% quantile of test-to-test NN distances in any space; (4) adversarial "
       "validation AUC ~0.5 (train is a random subset), predicted test positive rate within 2 SE of 45.2%; (5) number "
       "format independent of y; (6) the block is found by the variance rule on train-only and on test-only rows alike, "
       "with a wide gap in std ratio around the 1.5 threshold.")
cfg = {"name": "r901_id_order_dupes", "kind": "analysis", "verifier": "leak", "n_ref": a.n_ref, "n_perm": a.n_perm,
       "inputs": "train.csv/test.csv ids, features, train labels; S003 recomputed (R003 code)"}

with Run("r901_id_order_dupes", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    t0 = time.time()
    ddir = paths.data_dir(src)
    ds = data.load(src)
    X, y, Xt = ds.X.to_numpy(float), ds.y.astype(int), ds.X_test.to_numpy(float)
    ids_tr, ids_te = np.asarray(ds.ids), np.asarray(ds.ids_test)
    X_all = unlabeled_features(src)
    n_tr = len(y)
    out: dict = {}

    # ---------------------------------------------------------------- 1. train id / order
    all_ids = np.r_[ids_tr, ids_te]
    out["ids"] = {"train_min": int(ids_tr.min()), "train_max": int(ids_tr.max()), "test_min": int(ids_te.min()),
                  "test_max": int(ids_te.max()), "train_sorted": bool((np.diff(ids_tr) > 0).all()),
                  "test_sorted": bool((np.diff(ids_te) > 0).all()),
                  "union_is_0_to_N-1": bool(np.array_equal(np.sort(all_ids), np.arange(len(all_ids)))),
                  "overlap": int(len(set(ids_tr) & set(ids_te))),
                  "train_ids_first_10": [int(v) for v in ids_tr[:10]], "test_ids_first_10": [int(v) for v in ids_te[:10]]}
    lo, hi = all_ids.min(), all_ids.max()
    out["ids"]["ks_train_ids_vs_uniform_p"] = float(stats.kstest((ids_tr - lo) / (hi - lo), "uniform").pvalue)
    out["ids"]["ks_train_vs_test_ids_p"] = float(stats.ks_2samp(ids_tr, ids_te).pvalue)
    order_id = np.argsort(ids_tr, kind="stable")
    obs_l1, p_l1 = lag1_perm_p(y[order_id], a.n_perm)
    out["train_order"] = {
        "auc_id_vs_y": auc_and_p(y, ids_tr), "auc_rowpos_vs_y": auc_and_p(y, np.arange(n_tr)),
        "spearman_id_y": float(stats.spearmanr(ids_tr, y).statistic),
        "runs_test_id_order": runs_test(y[order_id]), "runs_test_file_order": runs_test(y),
        "lag1_autocorr_id_order": obs_l1, "lag1_perm_p": p_l1,
        "quintile_rates": bin_rate_test(ids_tr, y, 5),
        "auc_id_mod2_vs_y": auc_and_p(y, ids_tr % 2), "auc_id_mod10_vs_y": auc_and_p(y, ids_tr % 10),
    }

    # ---------------------------------------------------------------- 2. S003 test predictions vs test id / order
    scheme = Scheme.from_dict(yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))["official_cv"])
    splits = make_splits(y, scheme)
    cv = cross_validate(TransductiveGMM("highvar", 2, source=src), X, y, splits, scheme, X_test=Xt, n_jobs=1)
    tp = cv.test_pred
    run.save_array("s003_test_pred", tp)
    lab = (tp > 0.5).astype(int)
    out["s003_recomputed_cv_auc"] = float(cv.fold_scores["auc"].mean())
    order_te = np.argsort(ids_te, kind="stable")
    obs_l1t, p_l1t = lag1_perm_p(lab[order_te], min(a.n_perm, 2000), seed=1)
    out["test_order"] = {
        "spearman_id_score": float(stats.spearmanr(ids_te, tp).statistic),
        "spearman_rowpos_score": float(stats.spearmanr(np.arange(len(tp)), tp).statistic),
        "auc_id_as_score_for_pred_label": auc_and_p(lab, ids_te),
        "pred_pos_rate": float(lab.mean()), "n_boundary_0.01_0.99": int(((tp > 0.01) & (tp < 0.99)).sum()),
        "decile_rates_pred_label": bin_rate_test(ids_te, lab, 10),
        "runs_test_pred_label_file_order": runs_test(lab), "runs_test_pred_label_id_order": runs_test(lab[order_te]),
        "lag1_pred_label_id_order": obs_l1t, "lag1_perm_p": p_l1t,
    }
    # id vs features and vs the label-free posterior on all 20,000 rows
    hv = resolve_columns("highvar", X_all)
    gmm = _mixture(src, tuple(int(c) for c in hv), None, 2, "full", 5, 0, 1e-6)
    *_, Zhv = _basis(src, tuple(int(c) for c in hv), None)
    post = gmm.predict_proba(Zhv)[:, 1]
    rho = np.array([stats.spearmanr(all_ids, X_all[:, j]).statistic for j in range(X_all.shape[1])])
    thr = 4 / np.sqrt(len(all_ids))
    out["id_vs_features"] = {"max_abs_spearman": float(np.abs(rho).max()), "argmax_feature": int(np.abs(rho).argmax()),
                             "n_abs_gt_4se": int((np.abs(rho) > thr).sum()), "threshold_4se": float(thr),
                             "spearman_id_vs_mixture_posterior_all20k": float(stats.spearmanr(all_ids, post).statistic)}

    # ---------------------------------------------------------------- 3. duplicates
    mu, sd = X_all.mean(0), X_all.std(0)
    Z_all = (X_all - mu) / sd
    inf_cols = np.asarray(G.identify_structure(X_all)["informative"])
    spaces = {"all150": np.arange(X_all.shape[1]), "block13": hv, "informative12": inf_cols}
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(len(Xt), size=min(a.n_ref, len(Xt)), replace=False)
    dup_rows, nn_rows, close_pairs = [], [], []
    for sname, cols in spaces.items():
        Ztr, Zte = Z_all[:n_tr][:, cols], Z_all[n_tr:][:, cols]
        dup_rows.append({"space": sname, "n_cols": int(len(cols)), "dup_within_train": dup_count(X[:, cols]),
                         "dup_within_test": dup_count(Xt[:, cols]), "train_rows_in_test": dup_count(X[:, cols], Xt[:, cols])})
        d_trte, j_trte = nn_dist(Ztr, Zte)
        d_trtr, _ = nn_dist(Ztr, Ztr, self_idx=np.arange(n_tr))
        d_ref, _ = nn_dist(Zte[ref_idx], Zte, self_idx=ref_idx)
        q01, q50 = np.quantile(d_ref, 0.01), np.quantile(d_ref, 0.5)
        nn_rows.append({"space": sname, "ref_q001": float(np.quantile(d_ref, 0.001)), "ref_q01": float(q01),
                        "ref_median": float(q50), "ref_min": float(d_ref.min()),
                        "train_to_test_min": float(d_trte.min()), "train_to_test_q01": float(np.quantile(d_trte, 0.01)),
                        "train_to_test_median": float(np.median(d_trte)),
                        "n_train_closer_than_ref_q01": int((d_trte < q01).sum()),
                        "n_train_closer_than_ref_min": int((d_trte < d_ref.min()).sum()),
                        "train_to_train_min": float(d_trtr.min()), "train_to_train_median": float(np.median(d_trtr)),
                        "n_train_pairs_closer_than_ref_q01": int((d_trtr < q01).sum()),
                        "ks_trte_vs_ref_p": float(stats.ks_2samp(d_trte, d_ref).pvalue)})
        for i in np.argsort(d_trte)[:5]:
            close_pairs.append({"space": sname, "train_row": int(i), "train_id": int(ids_tr[i]), "train_y": int(y[i]),
                                "test_row": int(j_trte[i]), "test_id": int(ids_te[j_trte[i]]), "dist": float(d_trte[i]),
                                "dist_over_ref_median": float(d_trte[i] / q50), "s003_test_score": float(tp[j_trte[i]])})
    dup_df, nn_df, pairs_df = pd.DataFrame(dup_rows), pd.DataFrame(nn_rows), pd.DataFrame(close_pairs)
    run.save_table("duplicates_exact", dup_df)
    run.save_table("near_duplicates", nn_df)
    run.save_table("closest_train_test_pairs", pairs_df)
    out["closest_pairs_label_agreement"] = float(np.mean([(r["s003_test_score"] > 0.5) == bool(r["train_y"])
                                                           for r in close_pairs]))

    # ---------------------------------------------------------------- 4. is train a random subset?
    is_tr = np.r_[np.ones(n_tr, int), np.zeros(len(Xt), int)]
    cvs = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    adv = {}
    for sname, cols in spaces.items():
        aucs = []
        for r, (tr_i, va_i) in enumerate(cvs.split(X_all, is_tr)):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
            m.fit(X_all[tr_i][:, cols], is_tr[tr_i])
            aucs.append(roc_auc_score(is_tr[va_i], m.decision_function(X_all[va_i][:, cols])))
        adv[sname] = {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1))}
    post_tr, post_te = post[:n_tr], post[n_tr:]
    n_te = len(Xt)
    exp_rate = float(y.mean())
    out["random_subset"] = {
        "adversarial_auc": adv,
        "ks_mixture_posterior_train_vs_test_p": float(stats.ks_2samp(post_tr, post_te).pvalue),
        "train_pos_rate": exp_rate, "test_pred_pos_rate_s003": float(lab.mean()),
        "binom_p_train_rate_vs_test_pred_rate": float(stats.binomtest(int(y.sum()), n_tr, float(lab.mean())).pvalue),
        "mixture_weight_component1_all20k": float(post.mean()),
    }

    # ---------------------------------------------------------------- 5. file-format channel
    trs = pd.read_csv(ddir / "train.csv", dtype=str)
    tes = pd.read_csv(ddir / "test.csv", dtype=str)
    S = trs[FEATURES]
    dec = S.apply(lambda c: c.str.split(".").str[1].fillna("").str.len())
    slen = S.apply(lambda c: c.str.len())
    neg = S.apply(lambda c: c.str.startswith("-").astype(int))
    expo = S.apply(lambda c: c.str.contains("e", case=False).astype(int))
    col_auc_dec = np.array([roc_auc_score(y, dec[c]) if dec[c].nunique() > 1 else 0.5 for c in FEATURES])
    dec_te = tes[FEATURES].apply(lambda c: c.str.split(".").str[1].fillna("").str.len())
    out["format"] = {
        "auc_row_mean_decimals": auc_and_p(y, dec.mean(axis=1).to_numpy()),
        "auc_row_mean_strlen": auc_and_p(y, slen.mean(axis=1).to_numpy()),
        "auc_row_n_exponent_notation": auc_and_p(y, expo.sum(axis=1).to_numpy()) if expo.to_numpy().sum() else "none",
        "max_abs_col_auc_decimals_minus_half": float(np.abs(col_auc_dec - 0.5).max()),
        "argmax_col_decimals": FEATURES[int(np.abs(col_auc_dec - 0.5).argmax())],
        "decimals_values_train": sorted(int(v) for v in np.unique(dec.to_numpy())),
        "decimals_values_test": sorted(int(v) for v in np.unique(dec_te.to_numpy())),
        "mean_decimals_train": float(dec.to_numpy().mean()), "mean_decimals_test": float(dec_te.to_numpy().mean()),
        "n_exponent_values_train": int(expo.to_numpy().sum()),
        "min_nunique_per_column_all20k": int(pd.DataFrame(X_all).nunique().min()),
        "train_columns_order": list(trs.columns[:3]),
    }

    # ---------------------------------------------------------------- 6. label-freeness of the block
    def ratio_profile(M):
        s = M.std(0)
        r = s / np.median(s)
        o = np.argsort(-r)
        return r, o
    blk = {}
    for nm, M in (("all20k", X_all), ("train_only", X), ("test_only", Xt)):
        r, o = ratio_profile(M)
        sel = np.where(r > 1.5)[0]
        blk[nm] = {"n_cols_ratio_gt_1.5": int(len(sel)), "cols": [int(c) for c in sel],
                   "same_as_block13": bool(set(sel) == set(int(c) for c in hv)),
                   "top16_ratios": [round(float(r[c]), 3) for c in o[:16]], "top16_cols": [int(c) for c in o[:16]]}
    cors = abs_correlation(X, y)
    rank = (-cors).argsort().argsort() + 1
    out["block"] = {"ratio_profile": blk, "block13": [int(c) for c in hv],
                    "block_corr_ranks_on_250_labels": {int(c): int(rank[c]) for c in hv},
                    "overlap_block13_with_top13_corr": int(len(set(int(c) for c in hv) & set(int(c) for c in np.argsort(-cors)[:13]))),
                    "informative12": [int(c) for c in inf_cols],
                    "informative12_corr_ranks_on_250_labels": {int(c): int(rank[c]) for c in inf_cols},
                    "informative12_abs_corr": {int(c): round(float(cors[c]), 4) for c in inf_cols},
                    "overlap_informative12_with_top12_corr": int(len(set(int(c) for c in inf_cols) & set(int(c) for c in np.argsort(-cors)[:12])))}
    out["seconds"] = round(time.time() - t0, 1)

    tables = paths.tables_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    dup_df.to_csv(tables / "r901_duplicates_exact.csv", index=False)
    nn_df.to_csv(tables / "r901_near_duplicates.csv", index=False)
    (tables / "r901_id_order_probes.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    run.finish(extra_summary=out)
    print(json.dumps(out, indent=1, default=float)[:12000])
    print(dup_df.to_string(index=False))
    print(nn_df.to_string(index=False))
    print(pairs_df.to_string(index=False))
