"""R320 (ext34) - E4 follow-up: the bootstrap-stable |corr| set vs the label-free high-variance block.

E4 (extensions/e4_feature_stability) ranks features by |corr(feature, target)| on bootstrap
resamples and asks whether the stably selected features are the ones worth keeping. R001
found a LABEL-FREE candidate for "the informative features": the 13 features with
std > 1.5 x median std on train+test (8, 15, 26, 30, 46, 49, 64, 67, 77, 80, 98, 100, 112);
R003 showed that the class difference lives mainly in the covariance of that block
(QDA 0.996 and a transductive GMM 1.000 vs linear models 0.85-0.90).

Steps
  1. Same bootstrap frequencies and drop rules as E4 (the E4 config and the E4 functions are
     reused, so the stable sets are identical to the E4 run with the same config).
  2. Compare the stable sets (perm / half / gap rule, every k) with the block: overlap,
     features outside the block, block features never stable.
  3. Per-feature table: |corr| with y, frequency, label-free std ratio, R^2 on the block and on
     all other features (train+test, label-free), and label-free separation of the two
     components of the block GMM (standardised mean difference d and log variance ratio);
     the latter explains why some block features have small |corr| yet carry class signal.
  4. Usefulness on the OFFICIAL CV splits (5-fold x 20, seed 2026), identical splits, paired:
     three model families (L2 logreg = the E4 base model, QDA reg 0.1, transductive GMM K=2)
     x feature sets {all 150, block, stable (in-fold), block & stable, block - stable,
     block | stable}. "stable" = StabilityTopK(k, n_boot_fold, threshold) refitted on the
     training part of each fold (exactly the E4 honest_stable selection), so every set except
     the label-free block is chosen without the validation labels.

Label-free steps use train+test FEATURES (decision D007); labels only from the training part.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext34 --high-mem -- research/r320_e4_stable_vs_block.py --data raw --n-jobs 8
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from lib_ext34 import MODEL_FAMILIES, FoldColumnsModel, train_key  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.data import FEATURES  # noqa: E402
from ssc.features import StabilityTopK, abs_correlation, bootstrap_topk_frequency  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import _basis, _mixture, resolve_columns, unlabeled_features  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits, paired_compare  # noqa: E402

P = len(FEATURES)
E4_DIR = paths.ROOT / "extensions" / "e4_feature_stability"
SETS = ("all150", "block", "stable", "block_and_stable", "block_minus_stable", "block_or_stable")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (fixed order)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_e4():
    spec = importlib.util.spec_from_file_location("e4_run", E4_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r2_on(Z: np.ndarray, j: int, cols: np.ndarray) -> float:
    cols = np.array([c for c in cols if c != j])
    A = np.c_[np.ones(len(Z)), Z[:, cols]]
    coef, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
    resid = Z[:, j] - A @ coef
    return float(1 - resid.var() / Z[:, j].var())


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="raw", choices=paths.SOURCES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--k-use", type=int, default=10, help="k of the in-fold stable set used for usefulness")
    ap.add_argument("--models", nargs="*", default=list(MODEL_FAMILIES))
    ap.add_argument("--quick", action="store_true", help="smoke test: small B, 3 split seeds")
    a = ap.parse_args()
    src = a.data

    e4 = load_e4()
    cfg4 = e4.load_config(E4_DIR / "config.yaml", a.quick)
    bc, dc, uc = cfg4["bootstrap"], cfg4["drop"], cfg4["usefulness"]
    B, ks, method, bseed = int(bc["n_boot"]), bc["ks"], bc["method"], int(bc["seed"])
    n_perm, Bn = int(dc["null_permutations"]), int(dc["null_n_boot"])
    thr, nbf, fseed = float(uc["threshold"]), int(uc["n_boot_fold"]), int(uc["fold_seed"])
    official = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {})["official_cv"]
    if a.quick:
        official = {**official, "n_repeats": 3}
    scheme = Scheme.from_dict(official)
    k_use = a.k_use if a.k_use in ks else ks[0]

    ds = data.load(src)
    X, y = ds.X.to_numpy(dtype=float), ds.y
    X_all = unlabeled_features(src)
    block = resolve_columns("highvar", X_all)
    block_set = set(int(c) for c in block)
    std_all = X_all.std(0)

    cfg = {"name": "r320_e4_stable_vs_block", "e4_config": cfg4, "official_cv": scheme.to_dict(),
           "k_use": k_use, "models": a.models, "sets": list(SETS), "quick": a.quick,
           "block_rule": "std > 1.5 x median std on train+test (label-free)",
           "block_features": [FEATURES[j] for j in block], "uses_unlabeled_test_features": True}
    hyp = ("The bootstrap-stable |corr| set is NOT the set worth keeping: it is mostly a subset of the label-free "
           "high-variance block (plus feature 17), and the block features that are never stable carry class "
           "signal through their variance (covariance), not their mean, so QDA / the transductive GMM on the block "
           "beat the same models on the stable set, while L2 logreg is indifferent or prefers the stable set.")
    t_start = time.perf_counter()
    with Run("r320_e4_stable_vs_block", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
        print(f"R320 | data={src} | quick={a.quick} | B={B} | ks={ks} | k_use={k_use} | run_id={run.run_id}")
        print("block:", [FEATURES[j] for j in block])

        # ---- 1. bootstrap frequencies + drop rules (identical to E4) ----
        A = e4.boot_abs_corr(X, y, B, bseed, method)
        R = e4.ranks_desc(A)
        freqs = {k: (R <= k).mean(axis=0) for k in ks}
        same = bool(np.allclose(freqs[ks[0]], bootstrap_topk_frequency(X, y, ks[0], B, bseed, method)))
        null_sorted = {k: np.empty((n_perm, P)) for k in ks}
        for p in range(n_perm):
            y_p = np.random.default_rng(int(dc["label_seed"]) + p).permutation(y)
            R_p = e4.ranks_desc(e4.boot_abs_corr(X, y_p, Bn, bseed + 1 + p, method))
            for k in ks:
                null_sorted[k][p] = np.sort((R_p <= k).mean(axis=0))[::-1]
        drops, sets = [], {}
        for k in ks:
            row, s = e4.drop_rules(freqs[k], k, B, null_sorted[k], dc)
            drops.append(row)
            sets[k] = s
        drops = pd.DataFrame(drops)
        print(f"frequencies identical to ssc.features.bootstrap_topk_frequency: {same}")

        # ---- 2. stable sets vs block ----
        Zs = (X_all - X_all.mean(0)) / X_all.std(0)
        span_set = {j for j in range(P) if r2_on(Zs, j, block) > 0.999}  # block + exact linear combinations of it
        ov = []
        for k in ks:
            for rule in ("perm", "half", "gap", "binom"):
                S = set(int(i) for i in sets[k][rule])
                ov.append({"k": k, "rule": rule, "m": len(S), "n_in_block": len(S & block_set),
                           "jaccard_with_block": len(S & block_set) / len(S | block_set) if S | block_set else 1.0,
                           "stable_outside_block": " ".join(FEATURES[i] for i in sorted(S - block_set)),
                           "block_not_stable": " ".join(FEATURES[i] for i in sorted(block_set - S)),
                           "stable_outside_block_span": " ".join(FEATURES[i] for i in sorted(S - span_set)),
                           "stable_features": " ".join(FEATURES[i] for i in sorted(S))})
        overlap = pd.DataFrame(ov)

        # ---- 3. per-feature table ----
        corr_full = abs_correlation(X, y, method)
        Z = (X_all - X_all.mean(0)) / X_all.std(0)
        key = tuple(int(c) for c in block)
        gmm = _mixture(src, key, None, 2, "full", 5, 0, 1e-6)  # = TransductiveGMM("highvar") default mixture
        *_, Zb = _basis(src, key, None)
        comp = gmm.predict(Zb)
        if X[y == 1].shape[0] and (comp[: len(y)][y == 1].mean() < 0.5):
            comp = 1 - comp  # orient: component 1 = the one holding the train positives (labels only orient)
        c0, c1 = Z[comp == 0], Z[comp == 1]
        pooled = np.sqrt((c0.var(0) + c1.var(0)) / 2)
        d_comp = (c1.mean(0) - c0.mean(0)) / pooled
        lvr_comp = np.log(c1.var(0) / c0.var(0))
        Xs = (X - X.mean(0)) / X.std(0)
        x0, x1 = Xs[y == 0], Xs[y == 1]
        d_cls = (x1.mean(0) - x0.mean(0)) / np.sqrt((x0.var(0) + x1.var(0)) / 2)
        lvr_cls = np.log(x1.var(0) / x0.var(0))
        allc = np.arange(P)
        ft = pd.DataFrame({"feature": FEATURES, "in_block": [j in block_set for j in range(P)],
                           "std_ratio_to_median": std_all / np.median(std_all), "abs_corr_y": corr_full,
                           "rank_abs_corr": pd.Series(-corr_full).rank(method="first").astype(int).to_numpy()})
        for k in ks:
            ft[f"freq_k{k}"] = freqs[k]
            ft[f"stable_perm_k{k}"] = np.isin(allc, sets[k]["perm"])
        ft["r2_on_block_labelfree"] = [r2_on(Z, j, block) for j in range(P)]
        ft["r2_on_all_others_labelfree"] = [r2_on(Z, j, allc) for j in range(P)]
        ft["in_block_span"] = ft["r2_on_block_labelfree"] > 0.999  # exact linear combination of block features
        ft["gmm_comp_d_labelfree"] = d_comp
        ft["gmm_comp_log_var_ratio_labelfree"] = lvr_comp
        ft["class_d_train"] = d_cls
        ft["class_log_var_ratio_train"] = lvr_cls
        ft = ft.sort_values(["in_block", "abs_corr_y"], ascending=False).reset_index(drop=True)

        # ---- 4. usefulness on the official splits ----
        t0 = time.perf_counter()
        splits = make_splits(y, scheme)

        def fold_stable(s):
            sel = StabilityTopK(k=k_use, n_boot=nbf, threshold=thr, method=method, random_state=fseed)
            sel.fit(X[s.train_idx], y[s.train_idx])
            return train_key(X[s.train_idx], y[s.train_idx]), tuple(int(i) for i in sel.get_support(indices=True))

        stab = dict(Parallel(n_jobs=a.n_jobs)(delayed(fold_stable)(s) for s in splits))
        if len(stab) != len(splits):
            raise RuntimeError("duplicate training parts: train_key collision")
        maps = {"stable": {}, "block_and_stable": {}, "block_minus_stable": {}, "block_or_stable": {}}
        for kk, S in stab.items():
            S = set(S)
            maps["stable"][kk] = tuple(sorted(S))
            maps["block_and_stable"][kk] = tuple(sorted(S & block_set)) or tuple(sorted(S))[:1]
            maps["block_minus_stable"][kk] = tuple(sorted(block_set - S)) or tuple(sorted(block_set))
            maps["block_or_stable"][kk] = tuple(sorted(S | block_set))
        infold = pd.DataFrame([{"set": name, "n_features": len(c), "features": " ".join(FEATURES[i] for i in c)}
                               for name, mp in maps.items() for c in mp.values()])
        infold_summary = (infold.groupby("set")["n_features"].agg(["mean", "min", "max"]).reset_index()
                          .merge(infold.groupby("set")["features"].agg(lambda s: s.value_counts().index[0])
                                 .rename("most_common_set").reset_index(), on="set")
                          .merge(infold.groupby("set")["features"].nunique().rename("n_distinct_sets").reset_index(),
                                 on="set"))

        fold_rows, summ, pairs, fs = [], [], [], {}
        for fam in a.models:
            for name in SETS:
                if fam == "gmm" and name == "all150":
                    continue  # R003 gmm2_all already measured it (1.0000, run 20261022-050732_r003_gmm2_all)
                if fam == "qda" and name == "all150":
                    continue  # not estimable: ~110 rows per class < 150 features (sklearn QDA refuses)
                if name == "all150":
                    est = FoldColumnsModel(fam, columns=tuple(range(P)), source=src)
                elif name == "block":
                    est = FoldColumnsModel(fam, columns=tuple(int(c) for c in block), source=src)
                else:
                    est = FoldColumnsModel(fam, column_map=maps[name], source=src)
                t1 = time.perf_counter()
                cv = cross_validate(est, X, y, splits, scheme, test_pred="none", n_jobs=a.n_jobs)
                f = cv.fold_scores.assign(model=fam, set=name)
                fs[(fam, name)] = cv.fold_scores
                fold_rows.append(f)
                rep = cv.fold_scores.groupby("repeat")["auc"].mean()
                nf = (P if name == "all150" else len(block) if name == "block"
                      else float(np.mean([len(c) for c in maps[name].values()])))
                summ.append({"model": fam, "set": name, "n_features_mean": nf, "auc_mean": float(rep.mean()),
                             "auc_std_repeat": float(rep.std(ddof=1)) if len(rep) > 1 else float("nan"),
                             "auc_min_fold": float(cv.fold_scores["auc"].min()),
                             "split_sig": cv.split_signature})
                print(f"  {fam:6s} {name:20s} n_feat={nf:6.1f} auc={rep.mean():.4f} +/- {rep.std(ddof=1):.4f}"
                      f" ({time.perf_counter() - t1:.1f}s)")
            for name in SETS:
                if name != "block" and (fam, name) in fs:
                    pairs.append({"model": fam, "a": name, "b": "block", **paired_compare(fs[(fam, name)], fs[(fam, "block")])})
            if (fam, "all150") in fs:
                pairs.append({"model": fam, "a": "stable", "b": "all150", **paired_compare(fs[(fam, "stable")], fs[(fam, "all150")])})
        # across families on the same set (is the model or the set doing the work?)
        for name in ("block", "stable"):
            for f1, f2 in (("qda", "logreg"), ("gmm", "logreg"), ("gmm", "qda")):
                if (f1, name) in fs and (f2, name) in fs:
                    pairs.append({"model": f"{f1}-vs-{f2}", "a": f"{f1}:{name}", "b": f"{f2}:{name}",
                                  **paired_compare(fs[(f1, name)], fs[(f2, name)])})
        useful = pd.DataFrame(summ)
        paired = pd.DataFrame(pairs)
        fold_scores = pd.concat(fold_rows, ignore_index=True)
        t_use = time.perf_counter() - t0

        # ---- 5. save ----
        for name, df in (("feature_table", ft), ("overlap_with_block", overlap), ("drop", drops),
                         ("usefulness", useful), ("paired", paired), ("infold_sets", infold_summary),
                         ("usefulness_fold_scores", fold_scores)):
            run.save_table(name, df)
        tdir = run.dir / "report_preview" if a.quick else paths.tables_dir(src)
        fdir = run.dir / "report_preview" if a.quick else paths.figures_dir(src)
        tdir.mkdir(parents=True, exist_ok=True)
        fdir.mkdir(parents=True, exist_ok=True)
        for name, df in (("feature_table", ft), ("overlap_with_block", overlap), ("usefulness", useful),
                         ("paired", paired), ("infold_sets", infold_summary)):
            df.to_csv(tdir / f"r320_{name}.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.0, 1.2]})
        ax = axes[0]
        fk = f"freq_k{k_use}"
        st = ft[f"stable_perm_k{k_use}"].to_numpy()
        inb = ft["in_block"].to_numpy()
        span = ft["in_block_span"].to_numpy() & ~inb
        noise = ~inb & ~span
        # panel (a) uses its own hues (violet / yellow slots) so that blue / orange / aqua mean models in (b)
        BLOCK_C, SPAN_C = "#6250d6", "#eda100"
        groups = ((noise, MUTED, "noise (R$^2$ on block ~ 0)", 22),
                  (span, SPAN_C, "outside block, exact linear comb. of block", 34),
                  (inb, BLOCK_C, "label-free high-variance block", 34))
        for mask, color, lab, size in groups:
            ax.scatter(ft.loc[mask & ~st, "abs_corr_y"], ft.loc[mask & ~st, fk], s=size, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.8, label=lab, zorder=3)
            if (mask & st).any():
                ax.scatter(ft.loc[mask & st, "abs_corr_y"], ft.loc[mask & st, fk], s=70, color=color, marker="D",
                           edgecolor=INK, linewidth=0.9, zorder=4 if color == SPAN_C else 3.5)
        ax.scatter([], [], s=70, marker="D", color="white", edgecolor=INK, label=f"stable (perm rule, k={k_use})")
        top = ft[ft[fk] > 0.9]
        top_names = ", ".join(top.sort_values("abs_corr_y", ascending=False)["feature"])
        ax.annotate(f"freq > 0.9: {top_names}", xy=(top["abs_corr_y"].min(), 0.985), xytext=(0.05, 0.9),
                    textcoords="data", fontsize=7.5, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
        offsets = {"49": (6, -2), "67": (6, -2), "8": (6, -2), "26": (6, -3), "64": (2, 16), "30": (-4, 9),
                   "15": (-6, 9), "17": (-4, -14)}
        for _, r in ft[(inb | span) & ((ft[fk] <= 0.9) | (ft["feature"] == "17"))].iterrows():
            if span[_] and r["feature"] != "17":
                continue  # the other in-span features sit at frequency ~0; the legend names the group
            dx, dy = offsets.get(r["feature"], (5, 3))
            ax.annotate(r["feature"], (r["abs_corr_y"], r[fk]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, color=INK2)
        ax.axhline(k_use / P, color=INK2, lw=1, ls="--")
        ax.text(0.415, k_use / P + 0.015, f"chance {k_use}/150", color=INK2, fontsize=7.5, ha="right")
        ax.set_xlim(-0.01, 0.42)
        style(ax, f"(a) |corr| with y vs bootstrap top-{k_use} frequency (B = {B})",
              "|corr(feature, y)| on all 250 rows", "selection frequency")
        leg = ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
        for t in leg.get_texts():
            t.set_color(INK2)

        ax = axes[1]
        xs = np.arange(len(SETS))
        offs = np.linspace(-0.2, 0.2, len(a.models))
        for i, fam in enumerate(a.models):
            u = useful[useful["model"] == fam].set_index("set")
            have = [s for s in SETS if s in u.index]
            xi = np.array([SETS.index(s) for s in have]) + offs[i]
            ax.errorbar(xi, u.loc[have, "auc_mean"], yerr=u.loc[have, "auc_std_repeat"], ls="", marker="o", ms=7,
                        color=SERIES[i % len(SERIES)], capsize=3, lw=2,
                        label={"logreg": "L2 logistic (E4 base model)", "qda": "QDA (reg 0.1)",
                               "gmm": "transductive GMM K=2"}[fam])
        short = {"all150": "all 150", "block": "block", "stable": "stable", "block_and_stable": "block ∩\nstable",
                 "block_minus_stable": "block −\nstable", "block_or_stable": "block ∪\nstable"}
        labels = []
        for s_ in SETS:
            n = P if s_ == "all150" else len(block) if s_ == "block" else np.mean([len(c) for c in maps[s_].values()])
            labels.append(f"{short[s_]}\n({n:.0f})" if float(n).is_integer() else f"{short[s_]}\n({n:.1f})")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        style(ax, f"(b) honest CV AUC by feature set ({scheme.label()}, identical splits, mean +/- SD over seeds)",
              f"feature set (mean number of features); stable = in-fold StabilityTopK(k={k_use})", "CV AUC")
        leg = ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
        for t in leg.get_texts():
            t.set_color(INK2)
        fig.tight_layout()
        run.save_figure("stable_vs_block", fig)
        fig.savefig(fdir / "r320_stable_vs_block.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        run.summary.update({
            "block_features": [FEATURES[j] for j in block], "k_use": k_use, "B": B,
            "block_span_features": [FEATURES[j] for j in sorted(span_set)],
            "frequencies_match_ssc": same,
            "m_perm": {int(r["k"]): int(r["m_perm"]) for _, r in drops.iterrows()},
            "stable_perm_sets": {int(k): [FEATURES[i] for i in sets[k]["perm"]] for k in ks},
            "usefulness": {f"{r['model']}:{r['set']}": round(float(r["auc_mean"]), 5) for _, r in useful.iterrows()},
            "timings_seconds": {"usefulness": round(t_use, 1), "total": round(time.perf_counter() - t_start, 1)},
        })
        run.finish(extra_summary={"official_cv": not a.quick})

    pd.set_option("display.width", 200)
    print("\nDrop positions (same rules as E4):")
    print(drops[["k", "m_gap", "m_half", "m_binom", "m_perm", "top_freq"]].to_string(index=False))
    print("\nStable sets vs block:")
    print(overlap.drop(columns=["stable_features"]).to_string(index=False))
    print("\nPer-feature table (block + stable):")
    show = ["feature", "in_block", "std_ratio_to_median", "abs_corr_y", f"freq_k{k_use}", "r2_on_block_labelfree",
            "r2_on_all_others_labelfree", "gmm_comp_d_labelfree", "gmm_comp_log_var_ratio_labelfree",
            "class_d_train", "class_log_var_ratio_train"]
    print(ft[ft["in_block"] | ft[f"stable_perm_k{k_use}"]][show].round(3).to_string(index=False))
    print("\nIn-fold sets:")
    print(infold_summary.to_string(index=False))
    print("\nUsefulness:")
    print(useful.drop(columns=["split_sig"]).round(4).to_string(index=False))
    print("\nPaired (a - b):")
    print(paired[["model", "a", "b", "mean_diff", "se_split", "win_rate_fold", "corrected_p"]].round(4).to_string(index=False))
    print(f"\nrun folder: {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
