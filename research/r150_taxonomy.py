"""R150 - feature taxonomy and test of the sklearn make_classification hypothesis (real data).

Label-free steps use all 20,000 feature rows (train + test). Train labels are used ONLY for
(a) orienting the mixture components (which one is "class 1"), (b) |corr| with the target and
(c) the label-noise check (train rows whose label contradicts a confident posterior).

Outputs (run folder + reports/tables|figures with prefix r150_):
  r150_feature_taxonomy.csv   150 rows: role, std, mean, structure statistics, class-mean difference
  r150_dependencies.csv       the 8 exact linear dependencies (redundant = sum beta_i * informative_i + c)
  r150_B_unscaled.csv         implied unscaled make_classification B matrix (should be within [-1, 1])
  r150_basis_search.csv       top candidate informative sets (log det criterion)
  r150_noise_model.csv        tests of the noise model (std ~ U(0.5,3), mean ~ U(-5,5), Gaussian, independent)
  r150_mixture_bic.csv        BIC over K overall and within each class (n_clusters_per_class test)
  r150_label_noise_train.csv  train rows with posterior / label disagreement
  r150_flip_likelihood.csv    profile likelihood of flip_y from the train labels
  r150_posterior_agreement.csv  block vs structured-set vs informative-basis posteriors (question iv)
  r150_hypothesis.csv/.md     parameter-by-parameter verdict on the make_classification hypothesis

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rgen --high-mem -- research/r150_taxonomy.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_gen_replica as G  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.features import abs_correlation  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
a = p.parse_args()

ds = data.load(a.data)
X = unlabeled_features(a.data)
y = ds.y
n_tr = len(y)
n, P = X.shape
tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
print(f"cpus={os.cpu_count()} rows={n} features={P}", flush=True)


def save(run, name, df, md=False):
    run.save_table(name, df)
    df.to_csv(tables / f"r150_{name}.csv", index=False)
    if md:
        (tables / f"r150_{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


HYP = ("make_classification(n_samples=20000, n_features=150, n_informative=12, n_redundant=8, n_repeated=0, "
       "n_clusters_per_class=1, weights=None, hypercube=True, flip_y small) followed by a per-column affine map "
       "(noise std 0.5-3, means spread over about +-5). Predictions: 20 structured columns of rank 12 (8 exact "
       "dependencies, no duplicates); 130 independent Gaussian noise columns; one Gaussian per class (BIC within "
       "class prefers K=1); some informative columns with exactly zero class-mean difference; balanced classes; "
       "the 13-column high-variance block spans all 12 informative dimensions.")

cfg = {"name": "r150_taxonomy", "inputs": "train+test features (20,000 rows); train labels only for orientation, "
                                          "|corr| and label-noise check", "library": "research/lib_gen_replica.py"}
with Run("r150_taxonomy", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    out = {}
    st = G.identify_structure(X)
    S, N, I, R = st["structured"], st["noise"], st["informative"], st["redundant"]
    r = st["rank"]
    hv = G.highvar_columns(X)
    print(f"structured={len(S)} rank={r} informative={I} redundant={R} repeated={st['repeated_pairs']}", flush=True)

    # ---------------------------------------------------------------- orientation of the posterior
    post = st["posterior"]
    if post[:n_tr][y == 1].mean() < post[:n_tr][y == 0].mean():
        post = 1 - post
        flip_sign = -1.0
    else:
        flip_sign = 1.0

    # ---------------------------------------------------------------- per-feature taxonomy
    std, mu = X.std(0), X.mean(0)
    kurt, skew = stats.kurtosis(X, axis=0), stats.skew(X, axis=0)
    normp = np.array([stats.normaltest(X[:, j]).pvalue for j in range(P)])
    corr_t = abs_correlation(X[:n_tr], y)
    a_hat = np.full(P, np.nan)
    a_hat[I] = st["a_informative"]
    a_hat[R] = st["a_redundant"]
    a_hat[N] = std[N]
    zz = np.full(P, np.nan)
    dd = np.full(P, np.nan)
    for c, v in st["z_mean_diff"].items():
        zz[c] = flip_sign * v
    for c, v in st["d_std_mean_diff"].items():
        dd[c] = flip_sign * v
    role = np.array(["noise"] * P, dtype=object)
    role[[c for c in I if c in st["active"]]] = "informative_active"
    role[[c for c in I if c not in st["active"]]] = "informative_zero_diff"
    role[R] = "redundant"
    tax = pd.DataFrame({"feature": np.arange(P), "role": role, "in_highvar_block": np.isin(np.arange(P), hv),
                        "std": std, "mean": mu, "max_abs_corr_other": st["maxcorr"], "r2_on_all_others": st["r2_on_others"],
                        "class_mean_diff_z": zz, "class_mean_diff_std": dd, "scale_a_hat": a_hat,
                        "skew": skew, "excess_kurtosis": kurt, "normaltest_p": normp, "abs_corr_target_train": corr_t})
    save(run, "feature_taxonomy", tax)
    counts = tax["role"].value_counts().to_dict()
    out.update({"n_structured": len(S), "rank_structured": r, "n_dependencies": len(S) - r,
                "repeated_pairs": st["repeated_pairs"], "informative": I, "redundant": R,
                "informative_active": st["active"], "informative_zero_diff": st["zero_diff"],
                "role_counts": counts, "highvar_block": hv.tolist(),
                "structured_outside_block": sorted(set(S) - set(hv.tolist())),
                "eigenvalues_structured": [float(e) for e in st["eigenvalues_S"]]})

    # ---------------------------------------------------------------- the 8 dependencies
    dep_rows = []
    for jj, c in enumerate(R):
        row = {"redundant_feature": c, "const": float(st["const"][jj]), "r2": float(st["dep_r2"][jj]),
               "max_abs_residual": float(st["dep_max_abs_resid"][jj]), "scale_a_hat": float(st["a_redundant"][jj])}
        for ii, i in enumerate(I):
            row[f"beta_{i}"] = float(st["beta"][ii, jj])
        terms = " ".join(f"{st['beta'][ii, jj]:+.4f}*x{i}" for ii, i in enumerate(I))
        row["formula"] = f"x{c} = {st['const'][jj]:+.4f} {terms}"
        dep_rows.append(row)
    deps = pd.DataFrame(dep_rows)
    save(run, "dependencies", deps)
    Bdf = pd.DataFrame(st["B_unscaled"], columns=[f"red_{c}" for c in R])
    Bdf.insert(0, "informative", I)
    save(run, "B_unscaled", Bdf)
    Bv = st["B_unscaled"].ravel()
    out.update({"dependency_r2_min": float(st["dep_r2"].min()), "dependency_max_abs_residual": float(st["dep_max_abs_resid"].max()),
                "B_unscaled_min": float(Bv.min()), "B_unscaled_max": float(Bv.max()),
                "B_unscaled_ks_uniform_p": float(stats.kstest(Bv, "uniform", args=(-1, 2)).pvalue),
                "a_informative": [round(float(v), 3) for v in st["a_informative"]],
                "a_redundant": [round(float(v), 3) for v in st["a_redundant"]]})

    # ---------------------------------------------------------------- basis search
    bt = pd.DataFrame(st["basis_table"])
    save(run, "basis_search", bt)
    out.update({"basis_candidates": st["n_basis_candidates"], "basis_logdet_margin": st["basis_margin_logdet"]})

    # ---------------------------------------------------------------- noise model
    ns, nm = std[N], mu[N]
    Cn = np.corrcoef(X[:, N], rowvar=False)
    offn = np.abs(Cn[np.triu_indices(len(N), 1)])
    null_sd = 1 / np.sqrt(n)
    tests = [
        ("noise_std_ks_U(0.5,3)", stats.kstest(ns, "uniform", args=(0.5, 2.5)).pvalue, f"min {ns.min():.3f} max {ns.max():.3f}"),
        ("noise_std_ks_U(1,101) (sklearn scale=None)", stats.kstest(ns, "uniform", args=(1, 100)).pvalue, ""),
        ("noise_mean_ks_U(-5,5)", stats.kstest(nm, "uniform", args=(-5, 10)).pvalue, f"min {nm.min():.3f} max {nm.max():.3f}"),
        ("spearman(|mean|, std) (sklearn shift-then-scale predicts > 0)", stats.spearmanr(np.abs(nm), ns).pvalue,
         f"rho {stats.spearmanr(np.abs(nm), ns).statistic:.3f}"),
        ("noise normaltest p<0.001 count (expected 0.13)", float((normp[N] < 1e-3).sum()), f"of {len(N)}"),
        ("noise min normaltest p (Bonferroni x130)", float(normp[N].min() * len(N)), ""),
        ("noise max |pairwise corr| / null sd", float(offn.max() / null_sd), f"max {offn.max():.4f}, pairs {len(offn)}"),
        ("noise max |corr| with structured", float(np.abs(np.corrcoef(np.c_[X[:, N], X[:, S]], rowvar=False)[:len(N), len(N):]).max()), ""),
        ("noise |pairwise corr| ks vs half-normal(1/sqrt n)", stats.kstest(offn / null_sd, "halfnorm").pvalue, ""),
    ]
    nmod = pd.DataFrame(tests, columns=["test", "value", "detail"])
    save(run, "noise_model", nmod)
    out["noise_tests"] = {t: float(v) for t, v, _ in tests}

    # ---------------------------------------------------------------- mixture structure in the informative basis
    L = (X[:, I] - X[:, I].mean(0)) / X[:, I].std(0)
    bic_rows = []
    for k in range(1, 7):
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0, reg_covar=1e-6, max_iter=1000).fit(L)
        bic_rows.append({"scope": "all_rows", "K": k, "bic": gm.bic(L), "loglik": gm.score(L) * len(L)})
    hard = post > 0.5
    for cls, mask in (("class1_post>0.5", hard), ("class0_post<=0.5", ~hard)):
        Lc = L[mask]
        for k in range(1, 4):
            gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=0, reg_covar=1e-6, max_iter=1000).fit(Lc)
            bic_rows.append({"scope": cls, "K": k, "bic": gm.bic(Lc), "loglik": gm.score(Lc) * len(Lc)})
    bic = pd.DataFrame(bic_rows)
    save(run, "mixture_bic", bic)

    # Mardia multivariate kurtosis / skewness within confident class rows
    mard = []
    rng = np.random.default_rng(0)
    for cls, mask in (("class1", post > 0.999), ("class0", post < 0.001)):
        Lc = L[mask]
        m = Lc.mean(0)
        Sinv = np.linalg.inv(np.cov(Lc, rowvar=False, bias=True))
        D = np.einsum("ij,jk,ik->i", Lc - m, Sinv, Lc - m)
        b2 = float((D ** 2).mean())
        nn = len(Lc)
        zk = (b2 - r * (r + 2)) / np.sqrt(8 * r * (r + 2) / nn)
        sub = Lc[rng.choice(nn, min(3000, nn), replace=False)] - m
        Mx = sub @ Sinv @ sub.T
        b1 = float((Mx ** 3).mean())
        ns_ = len(sub)
        chi = ns_ * b1 / 6
        dfree = r * (r + 1) * (r + 2) / 6
        mard.append({"class": cls, "n": nn, "mardia_b2": b2, "expected_b2": r * (r + 2), "z_kurtosis": float(zk),
                     "mardia_b1_sub3000": b1, "skew_chi2": float(chi), "skew_df": dfree,
                     "skew_p": float(stats.chi2.sf(chi, dfree))})
    mard = pd.DataFrame(mard)
    save(run, "within_class_normality", mard)
    out["within_class_mardia"] = mard.to_dict("records")
    best_k = int(bic[bic.scope == "all_rows"].sort_values("bic").iloc[0]["K"])
    within_best = {s: int(g.sort_values("bic").iloc[0]["K"]) for s, g in bic[bic.scope != "all_rows"].groupby("scope")}
    out.update({"bic_best_K_all": best_k, "bic_best_K_within_class": within_best})

    # ---------------------------------------------------------------- class balance
    n1_soft = float(post.sum())
    n1_hard = int((post > 0.5).sum())
    out.update({"gmm_weights": [float(w) for w in st["gmm_weights"]], "n_class1_soft": n1_soft, "n_class1_hard": n1_hard,
                "balance_binom_p_vs_random_0.5": float(stats.binomtest(n1_hard, n, 0.5).pvalue),
                "train_pos_rate": float(y.mean()), "train_cluster1_rate": float((post[:n_tr] > 0.5).mean())})

    # ---------------------------------------------------------------- label noise in train
    pt = post[:n_tr]
    ln = pd.DataFrame({"train_row": np.arange(n_tr), "id": ds.ids, "y": y, "posterior_class1": pt})
    ln["disagree"] = (pt > 0.5) != (y == 1)
    ln["confident_contradiction"] = ((pt > 0.99) & (y == 0)) | ((pt < 0.01) & (y == 1))
    lnd = ln[(ln.disagree) | ((pt > 0.01) & (pt < 0.99))].sort_values("posterior_class1")
    save(run, "label_noise_train", lnd)
    fgrid = np.r_[0.0, np.geomspace(1e-4, 0.2, 60)]
    pi = np.clip(pt, 1e-12, 1 - 1e-12)
    ll = np.array([np.sum(np.log(np.where(y == 1, (1 - f) * pi + f / 2, (1 - f) * (1 - pi) + f / 2))) for f in fgrid])
    fl = pd.DataFrame({"flip_y": fgrid, "loglik_train": ll, "lr_vs_max": 2 * (ll.max() - ll)})
    save(run, "flip_likelihood", fl)
    f_mle = float(fgrid[ll.argmax()])
    f_up = float(fgrid[(2 * (ll.max() - ll) <= 3.84)].max())
    out.update({"train_disagreements": int(ln.disagree.sum()), "train_confident_contradictions": int(ln.confident_contradiction.sum()),
                "train_boundary_rows_0.01_0.99": int(((pt > 0.01) & (pt < 0.99)).sum()),
                "flip_y_mle_train": f_mle, "flip_y_95_upper_train": f_up,
                "loglik_ratio_flip0.01_vs_0": float(2 * (ll[0] - np.interp(0.01, fgrid, ll)))})

    # ---------------------------------------------------------------- posteriors: block vs structured vs informative (question iv)
    variants = {"block13_std": G.UnsupervisedGMM(X, "highvar", 2).fit(),
                "block13_pca12": G.UnsupervisedGMM(X, "highvar", 2, pca_dims="rank").fit(),
                "structured20_pca12": G.UnsupervisedGMM(X, "structured", 2, pca_dims="rank").fit(),
                "informative12": G.UnsupervisedGMM(X, I, 2).fit()}
    ref = post
    ag = []
    for k, g in variants.items():
        p1 = g.R[:, 1] if np.corrcoef(g.R[:, 1], ref)[0, 1] > 0 else g.R[:, 0]
        lo = np.log(np.clip(p1, 1e-300, 1)) - np.log(np.clip(1 - p1, 1e-300, 1))
        lo_ref = np.log(np.clip(ref, 1e-300, 1)) - np.log(np.clip(1 - ref, 1e-300, 1))
        bnd = (ref > 0.01) & (ref < 0.99)
        ag.append({"variant": k, "dims": g.dims_, "bic": g.bic, "max_abs_post_diff": float(np.abs(p1 - ref).max()),
                   "n_side_changes": int(((p1 > 0.5) != (ref > 0.5)).sum()),
                   "spearman_boundary_rows": float(stats.spearmanr(p1[bnd], ref[bnd]).statistic),
                   "max_abs_logodds_diff_boundary": float(np.abs(lo - lo_ref)[bnd].max()),
                   "n_boundary_0.1_0.9": int(((p1 > 0.1) & (p1 < 0.9)).sum()),
                   "n_test_boundary_0.1_0.9": int(((p1[n_tr:] > 0.1) & (p1[n_tr:] < 0.9)).sum())})
    ag = pd.DataFrame(ag)
    save(run, "posterior_agreement", ag)
    run.save_array("posterior_structured_pca12", post)
    out["posterior_agreement"] = ag.to_dict("records")

    # ---------------------------------------------------------------- hypercube / class_sep / covariance checks
    zI = np.array([zz[c] for c in I])
    zzero = np.array([zz[c] for c in st["zero_diff"]])
    du = np.abs(st["delta_u"]) / 2
    act = pd.DataFrame({"informative": I, "z_mean_diff": zI, "active": [c in st["active"] for c in I],
                        "abs_delta_u_over_2": du, "scale_a_hat": st["a_informative"]})
    save(run, "informative_coordinates", act)
    cs_vals = du[[c in st["active"] for c in I]]
    out.update({"hamming_hat": st["hamming"], "class_sep_hat": st["class_sep_hat"],
                "class_sep_per_active": [round(float(v), 4) for v in cs_vals],
                "class_sep_se_naive": float(cs_vals.std(ddof=1) / np.sqrt(len(cs_vals))) if len(cs_vals) > 1 else float("nan"),
                "zero_group_abs_z_max": st["z_zero_max"], "active_abs_z_min": st["z_active_min"],
                "redundant_abs_z_min": st["z_redundant_min"],
                "zero_group_ks_normal_p": float(stats.kstest(zzero, "norm").pvalue) if len(zzero) > 1 else float("nan"),
                "u_cov_offdiag_sd": st["u_cov_offdiag_sd"], "u_cov_offdiag_sd_expected": float(np.sqrt(r / 9)),
                "u_cov_diag_ratio_range": [float(st["u_cov_diag_ratio"].min()), float(st["u_cov_diag_ratio"].max())]})

    # ---------------------------------------------------------------- hypothesis table
    hyp = pd.DataFrame([
        ("n_samples", "20000", "250 train + 19,750 test; GMM weights %.4f/%.4f" % tuple(st["gmm_weights"])),
        ("n_features", "150", "columns 0..149"),
        ("n_informative", str(r), f"rank of the {len(S)} structured columns = {r} (8 eigenvalues < 1e-9 x max)"),
        ("n_redundant", str(len(R)), f"{len(R)} exact linear dependencies, min R^2 {st['dep_r2'].min():.12f}"),
        ("n_repeated", str(len(st["repeated_pairs"])), "no pair with |corr| > 1-1e-8"),
        ("n_useless (noise)", str(len(N)), "independent Gaussian columns, max |pairwise corr| %.4f (null sd %.4f)" % (offn.max(), null_sd)),
        ("n_clusters_per_class", "1", f"BIC best K={best_k} overall; within each class best K={within_best}"),
        ("weights", "None (balanced)", f"{n1_hard} vs {n - n1_hard} rows by posterior > 0.5"),
        ("hypercube", "True", f"{len(st['zero_diff'])} informative columns with zero class-mean difference (max |z| {st['z_zero_max']:.2f}); "
                              f"active |z| >= {st['z_active_min']:.1f}"),
        ("class_sep", f"{st['class_sep_hat']:.3f} (raw estimate; see R152 calibration)",
         f"hamming distance of the two centroids = {st['hamming']}; per active coordinate {np.round(cs_vals, 3).tolist()}"),
        ("flip_y", f"MLE {f_mle:.4f}, 95% upper {f_up:.4f} (train labels only)",
         f"{int(ln.confident_contradiction.sum())} confident contradictions in 250 train rows"),
        ("shift / scale", "0.0 / 1.0, then x*a+b with a~U(0.5,3), b~U(-5,5)",
         "noise std KS p=%.3f vs U(0.5,3); mean KS p=%.3f vs U(-5,5); |mean| independent of std" % (tests[0][1], tests[2][1])),
        ("A_k entries U(-1,1)", "consistent", f"within-class u-cov off-diagonal sd {st['u_cov_offdiag_sd']:.3f} (expected {np.sqrt(r / 9):.3f})"),
        ("B entries U(-1,1)", "consistent", f"implied B in [{Bv.min():.2f}, {Bv.max():.2f}], KS p vs U(-1,1) {out['B_unscaled_ks_uniform_p']:.3f}"),
        ("shuffle", "True", "roles scattered over the column indices"),
    ], columns=["parameter", "estimate", "evidence"])
    save(run, "hypothesis", hyp, md=True)

    # ---------------------------------------------------------------- figure
    colors = {"informative_active": "tab:red", "informative_zero_diff": "tab:orange", "redundant": "tab:blue", "noise": "lightgrey"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    o = np.argsort(-std)
    ax[0, 0].bar(range(P), std[o], color=[colors[role[j]] for j in o])
    ax[0, 0].axhline(1.5 * np.median(std), ls="--", c="k", lw=0.8)
    ax[0, 0].set_title("feature std (sorted), colour = role; dashed = high-variance rule")
    for k_, c_ in colors.items():
        ax[0, 0].bar([0], [0], color=c_, label=k_)
    ax[0, 0].legend(fontsize=8)
    sc = tax[tax.role != "noise"].sort_values("class_mean_diff_z")
    ax[0, 1].barh([str(f) for f in sc.feature], sc.class_mean_diff_z, color=[colors[x] for x in sc.role])
    ax[0, 1].axvline(4, c="k", lw=0.5, ls="--"); ax[0, 1].axvline(-4, c="k", lw=0.5, ls="--")
    ax[0, 1].set_title("class-mean difference z (GMM posterior), structured columns")
    ax[1, 0].scatter(nm, ns, s=12, c="grey")
    ax[1, 0].set_xlabel("noise column mean"); ax[1, 0].set_ylabel("noise column std")
    ax[1, 0].set_title("noise columns: std ~ U(0.5,3), mean ~ U(-5,5), independent")
    ba = bic[bic.scope == "all_rows"]
    ax[1, 1].plot(ba.K, ba.bic, marker="o")
    ax[1, 1].set_title("GMM BIC in the 12 informative columns (20,000 rows)"); ax[1, 1].set_xlabel("K")
    fig.tight_layout()
    fig.savefig(figs / "r150_taxonomy.png", dpi=150)
    run.save_figure("taxonomy", fig)

    run.finish(extra_summary=out)
    for k_, v_ in out.items():
        print(f"{k_}: {v_}")
    print(hyp.to_string(index=False))
