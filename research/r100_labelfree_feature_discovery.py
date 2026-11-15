"""R100 - label-free discovery of class-dependent features beyond the 13-feature block.

All steps below use only the FEATURES of the 20,000 train+test rows, except the clearly
marked descriptive comparisons with the training labels (sections 7, 8).

  1. exact linear redundancies of the 150 features (null space of the correlation matrix)
  2. dependency graph: which features are correlated with anything at all (n = 20,000, SE(r) ~ 0.007)
  3. GMM (K=2 full) on the block in its non-degenerate PCA basis -> soft labels r
  4. per-feature class-dependence given r: marginal LRT (df 2) and LRT conditional on the
     current set S (df dim(S)+2), BIC threshold df*log(n); exact redundancies flagged
  5. iterate: S <- S + {features with dBIC < 0}, refit the GMM, until nothing is added;
     backward check: is every member of S class-dependent given the rest of S?
  6. Gaussianity of the fitted components (Mahalanobis distances vs chi2_d, skew/kurtosis)
  7. (labels, descriptive) supervised in-fold selection frequency over the 100 official folds
  8. (labels, descriptive) training rows whose label disagrees with the mixture

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rmix --high-mem -- research/r100_labelfree_feature_discovery.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import lib_mix as L
from ssc import data, paths
from ssc.tracking import Run
from ssc.transductive import unlabeled_features
from ssc.validation import Scheme, make_splits

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--max-iter", type=int, default=6)
a = p.parse_args()
src = a.data

ds = data.load(src)
X_all = unlabeled_features(src)
n, P = X_all.shape
logn = np.log(n)
Z_all, mu, sd = L.standardize(X_all)
tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)

cfg = {"name": "r100_labelfree_feature_discovery", "rows": int(n), "block": list(L.BLOCK),
       "rule": "add feature j to S if conditional LRT given PCA(S) > (dim(S)+2)*log(n), not an exact linear combination of S"}
hyp = ("Label-free conditional likelihood-ratio tests (soft labels from the block GMM) find a few features outside the "
       "13-feature block that are class-dependent given the block; all other features are independent noise "
       "(no correlation with anything). Refitting on the enlarged set changes the ranking of the ~200 boundary rows.")
with Run("r100_labelfree_feature_discovery", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
    out = {}

    # 1. null space of the correlation matrix
    C = np.corrcoef(Z_all, rowvar=False)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)
    null_rows = []
    for rank, k in enumerate(o[:12]):
        vec = V[:, k]
        big = np.argsort(-np.abs(vec))
        members = [int(j) for j in big if abs(vec[j]) > 0.02]
        null_rows.append({"eig_rank_from_bottom": rank, "eigenvalue": float(w[k]),
                          "members": " ".join(map(str, members)),
                          "loadings": " ".join(f"{vec[j]:+.3f}" for j in members)})
    null_df = pd.DataFrame(null_rows)
    run.save_table("null_vectors", null_df)
    null_df.to_csv(tables / "r100_null_vectors.csv", index=False)
    exact = [r for r in null_rows if r["eigenvalue"] < 1e-8]
    redundant_members = sorted({int(m) for r in exact for m in r["members"].split()})
    out["n_exact_null"] = len(exact)
    out["features_in_exact_relations"] = redundant_members
    print("smallest eigenvalues:", np.round(np.sort(w)[:12], 10))
    print("features in exact relations:", redundant_members)

    # 2. dependency graph (|corr| > 0.05 ~ 7 SE)
    A = (np.abs(C) > 0.05) & ~np.eye(P, dtype=bool)
    deg = A.sum(1)
    connected = np.where(deg > 0)[0]
    # connected components
    comp = -np.ones(P, int)
    cid = 0
    for s in range(P):
        if comp[s] >= 0 or deg[s] == 0:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            u = stack.pop()
            for v in np.where(A[u])[0]:
                if comp[v] < 0:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    off = np.abs(C[np.triu_indices(P, 1)])
    out["n_features_with_any_corr_gt_0.05"] = int(len(connected))
    out["connected_features"] = connected.tolist()
    out["graph_components"] = [np.where(comp == c)[0].tolist() for c in range(cid)]
    out["max_abs_corr_isolated_features"] = float(np.abs(C[np.ix_(deg == 0, np.arange(P))] - np.eye(P)[deg == 0]).max())
    print("connected features:", connected.tolist())
    print("components:", out["graph_components"])

    # 3-5. iterative discovery
    S = list(L.BLOCK)
    history, iter_tables = [], []
    for it in range(a.max_iter):
        cols = tuple(S)
        *_, W, U = L.basis(cols, src)
        post, ll, fit = L.polished_posterior(cols, 2, "full", 5, 0, 1e-6, src)
        r = post[:, 0] if post[:, 0].mean() < 0.5 else post[:, 1]
        cand = [j for j in range(P) if j not in S]
        res_c = L.class_dependence(Z_all[:, cand], r, U)
        res_m = L.class_dependence(Z_all[:, cand], r, None)
        tab = pd.DataFrame({"iter": it, "feature": cand, "cond_lrt": res_c["lrt"], "cond_df": res_c["df"],
                            "cond_dbic": res_c["dbic"], "r2_on_S": res_c["r2"],
                            "cond_resvar_ratio": res_c["resvar_ratio"],
                            "marg_lrt": res_m["lrt"], "marg_dbic": res_m["dbic"],
                            "std_mean_diff": res_m["std_mean_diff"], "log_var_ratio": res_m["log_var_ratio"]})
        tab["cond_p"] = stats.chi2.sf(tab["cond_lrt"].clip(lower=0), tab["cond_df"])
        tab["exact_redundant"] = tab["r2_on_S"] > 1 - 1e-7
        tab = tab.sort_values("cond_lrt", ascending=False)
        iter_tables.append(tab)
        add = tab[(tab["cond_dbic"] < 0) & ~tab["exact_redundant"]]["feature"].astype(int).tolist()
        # backward check of the members of S
        back = []
        for j in S:
            rest = tuple(c for c in S if c != j)
            *_, _, U_r = L.basis(rest, src)
            rb = L.class_dependence(Z_all[:, [j]], r, U_r)
            back.append({"iter": it, "feature": j, "cond_lrt": float(rb["lrt"][0]), "cond_dbic": float(rb["dbic"][0]),
                         "r2_on_rest": float(rb["r2"][0])})
        back = pd.DataFrame(back)
        bic = -2 * ll * n + L.n_params(2, U.shape[1], "full") * logn
        history.append({"iter": it, "S": " ".join(map(str, S)), "dim_S": int(U.shape[1]), "n_S": len(S),
                        "mean_loglik": ll, "gmm_bic": bic, "frac_r_boundary": float(((r > 0.1) & (r < 0.9)).mean()),
                        "n_boundary_all": int(((r > 0.1) & (r < 0.9)).sum()),
                        "added_next": " ".join(map(str, add)),
                        "top5_cond": "; ".join(f"{int(f)}:{l:.0f}(df{d})" for f, l, d in tab[["feature", "cond_lrt", "cond_df"]].head(5).to_numpy()),
                        "backward_min_cond_lrt": float(back["cond_lrt"].min()),
                        "backward_weakest": int(back.loc[back["cond_lrt"].idxmin(), "feature"]),
                        "backward_n_dbic_pos": int((back["cond_dbic"] > 0).sum())})
        run.save_table(f"backward_iter{it}", back)
        print(f"iter {it}: |S|={len(S)} dim={U.shape[1]} boundary={history[-1]['n_boundary_all']} add={add}")
        print(tab.head(12).to_string(index=False))
        print("backward weakest:", back.sort_values("cond_lrt").head(4).to_string(index=False))
        np.save(run.dir / f"posterior_iter{it}.npy", r)
        if not add:
            break
        S = S + add
    hist = pd.DataFrame(history)
    run.save_table("iterations", hist)
    hist.to_csv(tables / "r100_iterations.csv", index=False)
    allt = pd.concat(iter_tables)
    run.save_table("feature_scores", allt)
    iter_tables[0].to_csv(tables / "r100_feature_scores_iter0.csv", index=False)
    S_final = S
    out["S_final"] = S_final
    out["n_iter"] = len(history)

    # 6. Gaussianity of components on the final set
    cols = tuple(S_final)
    *_, U = L.basis(cols, src)
    post, ll, fit = L.polished_posterior(cols, 2, "full", 5, 0, 1e-6, src)
    gof = []
    d = U.shape[1]
    for k in range(2):
        m = post[:, k] > 0.99
        Xc = U[m] - fit["means"][k]
        Linv = np.linalg.inv(np.linalg.cholesky(fit["covs"][k]))
        Wk = Xc @ Linv.T
        md = (Wk**2).sum(1)
        ks = stats.kstest(md, stats.chi2(d).cdf)
        sk = stats.skew(Wk, axis=0)
        ku = stats.kurtosis(Wk, axis=0)
        gof.append({"component": k, "n_confident": int(m.sum()), "dim": d, "mahal_mean": float(md.mean()),
                    "chi2_mean": d, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                    "max_abs_skew_whitened": float(np.abs(sk).max()), "max_abs_exkurt_whitened": float(np.abs(ku).max())})
    gof = pd.DataFrame(gof)
    run.save_table("component_gaussianity", gof)
    print(gof.to_string(index=False))
    out["gaussianity"] = gof.to_dict("records")

    # BIC over K on the final set (full covariance)
    kb = []
    for K in range(1, 7):
        g = L.sk_gmm(cols, K, "full", 5, 0, 1e-6, 1e-3, src)
        kb.append({"K": K, "bic": float(g.bic(U)), "mean_loglik": float(g.score(U))})
    kb = pd.DataFrame(kb)
    run.save_table("bic_by_K_final_set", kb)
    print(kb.to_string(index=False))
    out["bic_best_K_final_set"] = int(kb.loc[kb["bic"].idxmin(), "K"])

    # posterior comparison block vs final
    r_block = np.load(run.dir / "posterior_iter0.npy")
    r_fin = post[:, 0] if post[:, 0].mean() < 0.5 else post[:, 1]
    # orient both with the training labels only for the comparison (descriptive)
    if np.corrcoef(r_block[:250], ds.y)[0, 1] < 0:
        r_block = 1 - r_block
    if np.corrcoef(r_fin[:250], ds.y)[0, 1] < 0:
        r_fin = 1 - r_fin
    bs = L.boundary_stats(r_fin[250:], r_block[250:])
    out["final_vs_block_test"] = bs
    s3 = L.s003_pred(src)
    np.save(run.dir / "s003_pred_recomputed.npy", s3)
    out["s003_vs_block_polished_spearman"] = float(stats.spearmanr(s3, r_block[250:])[0])
    out["s003_vs_final"] = L.boundary_stats(r_fin[250:], s3)
    print("final vs block:", bs)

    # 7. supervised in-fold selection (descriptive: uses fold labels)
    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
    scheme = Scheme.from_dict(official)
    splits = make_splits(ds.y, scheme)
    Ztr = Z_all[:250]
    k_sel = len(S_final)
    freq_lrt = np.zeros(P)
    freq_corr = np.zeros(P)
    for s in splits:
        yt = ds.y[s.train_idx].astype(float)
        res = L.class_dependence(Ztr[s.train_idx], yt, None)
        freq_lrt[np.argsort(-res["lrt"])[:k_sel]] += 1
        c = np.abs([np.corrcoef(Ztr[s.train_idx, j], yt)[0, 1] for j in range(P)])
        freq_corr[np.argsort(-c)[:k_sel]] += 1
    freq_lrt /= len(splits)
    freq_corr /= len(splits)
    sup = pd.DataFrame({"feature": np.arange(P), "in_S_final": np.isin(np.arange(P), S_final),
                        "in_block": np.isin(np.arange(P), L.BLOCK),
                        "freq_top_k_marginal_lrt": freq_lrt, "freq_top_k_abs_corr": freq_corr})
    sup = sup.sort_values("freq_top_k_marginal_lrt", ascending=False)
    run.save_table("supervised_infold_selection", sup)
    sup.to_csv(tables / "r100_supervised_infold_selection.csv", index=False)
    top_lrt = sup.head(k_sel)["feature"].tolist()
    out["supervised_top_k_by_freq_lrt"] = top_lrt
    out["overlap_supervised_lrt_with_S_final"] = int(len(set(top_lrt) & set(S_final)))
    out["supervised_mean_overlap_per_fold_lrt"] = float(sup.loc[sup["in_S_final"], "freq_top_k_marginal_lrt"].sum() / k_sel)
    out["supervised_mean_overlap_per_fold_corr"] = float(sup.loc[sup["in_S_final"], "freq_top_k_abs_corr"].sum() / k_sel)
    print(sup.head(25).to_string(index=False))

    # 8. training rows that disagree with the mixture (descriptive)
    dis = pd.DataFrame({"train_row": np.arange(250), "y": ds.y, "r_block": r_block[:250], "r_final": r_fin[:250]})
    dis = dis[((dis["r_block"] >= 0.5).astype(int) != dis["y"]) | ((dis["r_final"] >= 0.5).astype(int) != dis["y"]) |
              dis["r_block"].between(0.01, 0.99) | dis["r_final"].between(0.01, 0.99)]
    run.save_table("train_rows_uncertain_or_disagree", dis)
    out["train_rows_uncertain_or_disagree"] = dis.to_dict("records")
    print(dis.to_string(index=False))

    np.save(run.dir / "posterior_final.npy", r_fin)
    run.finish(extra_summary=out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("gaussianity",)}, default=str, indent=1)[:4000])
