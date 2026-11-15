"""R441 - independent re-run of the R206 label-budget curve (D009 evidence 1) with a different seed set
(research round 2, label "r2_audit").

Question: is R206's statement "to reach AUC >= 0.99 the transductive GMM needs 4 labels, self-training QDA about 20,
supervised QDA (train+test block) about 40 and strictly supervised QDA about 80" stable under other splits and
other label subsamples, and does it hold for the ACTUAL candidate procedures (S003, S005 = r201, S006 = r209)?

Design (differs from R206 on purpose):
  * splits: the OFFICIAL scheme (project.yaml official_cv), but only repeats 4..19 (R206 used repeats 0..3), so the
    validation folds are new; 80 folds.
  * label subsamples: --draws stratified draws per (fold, budget) from the 200-row training fold, rng keyed by
    np.random.default_rng([4410, repeat, fold, n_labels, draw]) (R206: 100000 + 1000 r + 100 f + 10 d + n);
    n_pos = clip(round(n * fold positive rate), 1, n - 1), so n = 2 means one label per class.
  * every model is scored on the full 50-row validation fold (as in R206).
  * code: the candidate procedures come from the verifier's independent re-implementation research/r931_cand_lib.py
    (bit-identical to the archived test predictions, r932), applied to a split whose training part is the
    subsample; the R206-equivalent curves are re-written here from sklearn primitives (not R206's code).

Models (uses test features? in brackets):
  s003_gmm2_hv          [yes] S003: TransductiveGMM K=2 on the train+test block; labels only map the components
  r209_selftrain_qda_lw [yes] S006: SelfTraining(QDA Ledoit-Wolf, threshold 0.99) on all 20,000 rows
  r201_qda_block        [no ] S005: in-fold block + scaling, QDA reg by nested inner CV (AUC), grid of r201
  r210_qda_block_nll    [no ] same nested procedure with inner log-loss (r210)
  r255_bqda_eb_hv       [yes, block/scaling only] Bayesian QDA (EB NIW prior) on the train+test block
  r255_bqda_eb_trainonly[no ] Bayesian QDA, block and scaling from the labelled rows only (lib_fm)
  eq206_qda_infold_r003 [no ] = R206 sup_qda_infold_r003 (QDA reg 0.003, in-fold block)
  eq206_qda_infold_lw   [no ] = R206 sup_qda_infold_lw   (QDA Ledoit-Wolf, in-fold block)
  eq206_qda_tt_r003     [block only] = R206 sup_qda_tt_r003 (QDA reg 0.003, train+test block)
  eq206_qda_tt_r01      [block only] = R206 sup_qda_tt_r01 = S002 / R304 qda_hv (QDA reg 0.1, train+test block)
  eq206_selftrain_lw_t075 [yes] = R206 ssl_selftrain_lw (SelfTraining QDA-LW, threshold 0.75)

Outputs: run tables label_budget_evals / label_budget_summary / label_budget_first_hit, reports/tables/raw
r441_label_budget.{csv,md}, reports/figures/raw r441_label_budget.png. The comparison with R206 and R304 is done by
research/r440_core_numbers.py (it needs the archived R206 / R304 tables).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2aud --high-mem -- research/r441_label_budget_rerun.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import warnings
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.semi_supervised import SelfTrainingClassifier  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("r931_cand_lib")
from lib_fm import BayesQDA, TrainHighVar  # noqa: E402
from lib_sup import HighVarColumns  # noqa: E402
from ssc import data, paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import unlabeled_features  # noqa: E402
from ssc.validation import Scheme, make_splits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", type=int, default=8)
p.add_argument("--repeats", type=int, nargs=2, default=[4, 20], help="official repeats [start, stop)")
p.add_argument("--draws", type=int, default=5)
p.add_argument("--budgets", type=int, nargs="*", default=[2, 4, 6, 10, 20, 40, 80, 120, 160, 200])
p.add_argument("--max-folds", type=int, default=0, help="smoke test: use only the first N folds")
p.add_argument("--models", nargs="*", default=None)
a = p.parse_args()
src = a.data

MODELS = ["s003_gmm2_hv", "r209_selftrain_qda_lw", "r201_qda_block", "r210_qda_block_nll", "r255_bqda_eb_hv",
          "r255_bqda_eb_trainonly", "eq206_qda_infold_r003", "eq206_qda_infold_lw", "eq206_qda_tt_r003",
          "eq206_qda_tt_r01", "eq206_selftrain_lw_t075"]
if a.models:
    MODELS = [m for m in MODELS if m in a.models]
USES_TEST = {"s003_gmm2_hv": "yes", "r209_selftrain_qda_lw": "yes", "r201_qda_block": "no", "r210_qda_block_nll": "no",
             "r255_bqda_eb_hv": "block+scaling", "r255_bqda_eb_trainonly": "no", "eq206_qda_infold_r003": "no",
             "eq206_qda_infold_lw": "no", "eq206_qda_tt_r003": "block+scaling", "eq206_qda_tt_r01": "block+scaling",
             "eq206_selftrain_lw_t075": "yes"}
R206_NAME = {"eq206_qda_infold_r003": "sup_qda_infold_r003", "eq206_qda_infold_lw": "sup_qda_infold_lw",
             "eq206_qda_tt_r003": "sup_qda_tt_r003", "eq206_qda_tt_r01": "sup_qda_tt_r01",
             "eq206_selftrain_lw_t075": "ssl_selftrain_lw", "s003_gmm2_hv": "trans_gmm2"}
LABELS = {"s003_gmm2_hv": "Transductive GMM K=2 (S003)", "r209_selftrain_qda_lw": "Self-training QDA-LW, thr 0.99 (S006)",
          "r201_qda_block": "Strict supervised QDA, nested reg, AUC (S005)",
          "r210_qda_block_nll": "Strict supervised QDA, nested reg, log-loss (r210)",
          "r255_bqda_eb_hv": "Bayesian QDA EB, train+test block", "r255_bqda_eb_trainonly": "Bayesian QDA EB, in-fold block (strict)",
          "eq206_qda_infold_r003": "QDA reg 0.003, in-fold block (strict; R206)",
          "eq206_qda_infold_lw": "QDA-LW, in-fold block (strict; R206)", "eq206_qda_tt_r003": "QDA reg 0.003, train+test block (R206)",
          "eq206_qda_tt_r01": "QDA reg 0.1, train+test block (S002; R206/R304)",
          "eq206_selftrain_lw_t075": "Self-training QDA-LW, thr 0.75 (R206)"}

HYP = ("Re-running R206's design on new folds (official repeats 4-19) and new label subsamples reproduces its "
       "label-budget statements within sampling noise: the transductive GMM (S003) is at AUC ~1.0 from 2-4 labels "
       "(1-2 per class; R304 saw 1.0 at 2 labels in 50/50 subsets); self-training QDA-LW reaches mean AUC >= 0.99 at "
       "about 20 labels (threshold 0.75, R206) and the S006 configuration (threshold 0.99) at 20-40; QDA reg 0.003 on "
       "the train+test block at about 40; strictly supervised QDA (reg 0.003, in-fold block) at about 80; the actual "
       "nested S005 procedure (r201) is undefined below ~40 labels and reaches 0.99 at 80-120. At every common budget "
       "the mean AUC of the R206-equivalent curves is within 3 combined fold-cluster SEs of R206's. A difference "
       "beyond 3 SE at several budgets, or a first-hit budget that moves by more than one grid step, would mean "
       "R206's numbers depend on its seeds and must be quoted with an interval.")


def subsample(train_idx, y, n_l, rng):
    if n_l >= len(train_idx):
        return np.asarray(train_idx)
    yt = y[train_idx]
    pos, neg = train_idx[yt == 1], train_idx[yt == 0]
    n_pos = int(np.clip(round(n_l * yt.mean()), 1, n_l - 1))
    return np.sort(np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_l - n_pos, replace=False)])


def _auc(yv, sc):
    sc = np.asarray(sc, dtype=float)
    if not np.all(np.isfinite(sc)):
        raise ValueError("non-finite scores")
    return C.roc_auc_exact(yv, sc) if np.ptp(sc) > 0 else 0.5


def eval_split(name, lf, Xtr, y, sp):
    """Validation AUC of one model on one (subsampled) split; returns (auc, info string)."""
    tr, va = sp.train_idx, sp.val_idx
    yv = y[va]
    if name == "s003_gmm2_hv":
        cv = []
        C.gmm_scores(lf.R_s003, lf.n_tr, y, [sp], cv=cv)
        return cv[0], ""
    if name == "r209_selftrain_qda_lw":
        cv = []
        _, inf = C.selftrain_scores(lf, y, [sp], cv=cv)
        return cv[0], f"n_pseudo={inf['selftrain_n_pseudo_mean']:.0f}"
    if name in ("r201_qda_block", "r210_qda_block_nll"):
        val = {}
        _, chosen = C.nested_qda_scores(Xtr, Xtr[:2], y, [sp], grid=C.REG, n_base=C.N_BASE, val_out=val)
        return val[name][0], f"reg={chosen[name][0]}"
    if name == "r255_bqda_eb_hv":
        cv = []
        C.bqda_scores(lf, y, [sp], cv=cv)
        return cv[0], ""
    if name == "r255_bqda_eb_trainonly":
        th = TrainHighVar(1.5).fit(Xtr[tr])
        m = BayesQDA().fit(th.transform(Xtr[tr]), y[tr])
        return _auc(yv, m.predict_proba(th.transform(Xtr[va]))[:, 1]), f"ncols={len(th.columns_)}"
    if name in ("eq206_qda_infold_r003", "eq206_qda_infold_lw"):
        sel = HighVarColumns().fit(Xtr[tr])
        sc = StandardScaler().fit(sel.transform(Xtr[tr]))
        q = QDA(reg_param=0.003) if name.endswith("r003") else QDA(solver="eigen", shrinkage="auto")
        q.fit(sc.transform(sel.transform(Xtr[tr])), y[tr])
        return _auc(yv, q.predict_proba(sc.transform(sel.transform(Xtr[va])))[:, 1]), f"ncols={len(sel.block_)}"
    if name in ("eq206_qda_tt_r003", "eq206_qda_tt_r01"):
        Z = lf.Zhv[: lf.n_tr]
        q = QDA(reg_param=0.003 if name.endswith("r003") else 0.1).fit(Z[tr], y[tr])
        return _auc(yv, q.predict_proba(Z[va])[:, 1]), ""
    if name == "eq206_selftrain_lw_t075":
        yp = np.full(len(lf.Zhv), -1, dtype=int)
        yp[tr] = y[tr]
        st = SelfTrainingClassifier(QDA(solver="eigen", shrinkage="auto"), threshold=0.75, criterion="threshold",
                                    k_best=10, max_iter=10).fit(lf.Zhv, yp)
        return _auc(yv, st.predict_proba(lf.Zhv[va])[:, 1]), f"n_pseudo={int((st.labeled_iter_ > 0).sum())}"
    raise KeyError(name)


def fold_task(lf, Xtr, y, s, budgets, draws, models):
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n_l in budgets:
            for d in (range(draws) if n_l < len(s.train_idx) else [0]):
                rng = np.random.default_rng([4410, s.repeat, s.fold, n_l, d])
                tr = subsample(s.train_idx, y, n_l, rng)
                sp = SimpleNamespace(repeat=s.repeat, fold=s.fold, train_idx=tr, val_idx=s.val_idx)
                for name in models:
                    t0 = time.perf_counter()
                    try:
                        auc, info = eval_split(name, lf, Xtr, y, sp)
                        err = ""
                    except Exception as e:  # noqa: BLE001 - undefined fits at tiny budgets are recorded as NaN
                        auc, info, err = np.nan, "", f"{type(e).__name__}: {str(e)[:60]}"
                    out.append({"model": name, "n_labels": n_l, "repeat": s.repeat, "fold": s.fold, "draw": d,
                                "n_pos": int(y[tr].sum()), "auc": auc, "info": info, "error": err,
                                "seconds": round(time.perf_counter() - t0, 4)})
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, n), g in df.groupby(["model", "n_labels"]):
        ok = g.dropna(subset=["auc"])
        fm = ok.groupby(["repeat", "fold"])["auc"].mean()  # fold-cluster means (draws share a validation fold)
        rows.append({"model": m, "n_labels": n, "n_evals": len(g), "n_failed": int(g["auc"].isna().sum()),
                     "auc_mean": ok["auc"].mean() if len(ok) else np.nan,
                     "auc_median": ok["auc"].median() if len(ok) else np.nan,
                     "auc_q10": ok["auc"].quantile(0.10) if len(ok) else np.nan,
                     "auc_sd": ok["auc"].std(ddof=1) if len(ok) > 1 else np.nan,
                     "auc_se_cluster": fm.std(ddof=1) / np.sqrt(len(fm)) if len(fm) > 1 else np.nan,
                     "n_folds": len(fm), "frac_ge_099": float((ok["auc"] >= 0.99).mean()) if len(ok) else np.nan,
                     "frac_eq_1": float((ok["auc"] >= 1 - 1e-12).mean()) if len(ok) else np.nan})
    s = pd.DataFrame(rows)
    s["uses_test_features"] = s["model"].map(USES_TEST)
    s["label"] = s["model"].map(LABELS)
    return s


def first_hits(summ: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, t in summ.groupby("model"):
        t = t.sort_values("n_labels")
        full = t[t["n_failed"] == 0]
        def first(cond):  # only budgets where every evaluation was defined (a partial mean would be biased)
            hit = t.loc[cond(t) & (t["n_failed"] == 0), "n_labels"]
            return int(hit.iloc[0]) if len(hit) else None
        out.append({"model": m, "label": LABELS[m], "uses_test_features": USES_TEST[m],
                    "first_defined": int(t.loc[t["n_failed"] < t["n_evals"], "n_labels"].min()) if (t["n_failed"] < t["n_evals"]).any() else None,
                    "first_fully_defined": int(full["n_labels"].min()) if len(full) else None,
                    "min_labels_mean_ge_0.99": first(lambda x: x["auc_mean"] >= 0.99),
                    "min_labels_mean_minus_2se_ge_0.99": first(lambda x: x["auc_mean"] - 2 * x["auc_se_cluster"] >= 0.99),
                    "min_labels_median_ge_0.99": first(lambda x: x["auc_median"] >= 0.99),
                    "min_labels_mean_ge_0.95": first(lambda x: x["auc_mean"] >= 0.95)})
    return pd.DataFrame(out)


scheme = Scheme.from_dict(yaml.safe_load(open(paths.PROJECT_YAML, encoding="utf-8"))["official_cv"])
ds = data.load(src)
y = np.asarray(ds.y).astype(int)
Xtr = ds.X.to_numpy(float)
X_all = unlabeled_features(src)
n_tr = len(y)
assert np.allclose(X_all[:n_tr], Xtr)
splits = [s for s in make_splits(y, scheme) if a.repeats[0] <= s.repeat < a.repeats[1]]
if a.max_folds:
    splits = splits[: a.max_folds]

cfg = {"name": "r441_label_budget_rerun", "models": {m: LABELS[m] for m in MODELS},
       "uses_unlabeled_test_features": {m: USES_TEST[m] for m in MODELS}, "budgets": a.budgets, "draws": a.draws,
       "official_repeats_used": a.repeats, "n_folds": len(splits), "cv": scheme.to_dict(),
       "draw_rng": "np.random.default_rng([4410, repeat, fold, n_labels, draw])",
       "validation": "full 50-row validation fold of each official split",
       "r206_counterparts": R206_NAME, "code": "research/r441_label_budget_rerun.py + research/r931_cand_lib.py + lib_fm"}
with Run("r441_label_budget_rerun", cfg, kind="analysis", source=src, author=a.author, hypothesis=HYP) as run:
    t0 = time.time()
    lf = C.LabelFree(X_all, n_tr, need=["s003_gmm2_hv"])
    print(f"[r441] label-free fits {time.time() - t0:.1f}s; {len(splits)} folds x {len(a.budgets)} budgets x "
          f"{len(MODELS)} models, cpus={os.cpu_count()}", flush=True)
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(fold_task)(lf, Xtr, y, s, a.budgets, a.draws, MODELS)
                                               for s in splits)
    df = pd.DataFrame([r for chunk in res for r in chunk])
    run.save_table("label_budget_evals", df)
    summ = summarise(df)
    run.save_table("label_budget_summary", summ)
    hits = first_hits(summ)
    run.save_table("label_budget_first_hit", hits)

    tables, figs = paths.tables_dir(src), paths.figures_dir(src)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    summ.to_csv(tables / "r441_label_budget.csv", index=False)
    wide = summ.pivot(index="n_labels", columns="model", values="auc_mean")[[m for m in MODELS if m in set(summ["model"])]]
    (tables / "r441_label_budget.md").write_text(
        f"Mean validation-fold AUC by number of labelled rows (official repeats {a.repeats[0]}-{a.repeats[1] - 1}, "
        f"{len(splits)} folds x {a.draws} draws; NaN = undefined fit; run {run.run_id})\n\n"
        + wide.round(4).to_markdown() + "\n\nFirst budget reaching AUC >= 0.99\n\n" + hits.to_markdown(index=False) + "\n",
        encoding="utf-8")

    # figure: two panels (unlabeled rows used / strictly supervised), fixed categorical slots
    panels = [("Uses the 19,750 unlabeled test rows (or their block)",
               ["s003_gmm2_hv", "r209_selftrain_qda_lw", "eq206_selftrain_lw_t075", "r255_bqda_eb_hv", "eq206_qda_tt_r003",
                "eq206_qda_tt_r01"]),
              ("Strictly supervised (labelled rows only)",
               ["r201_qda_block", "r210_qda_block_nll", "r255_bqda_eb_trainonly", "eq206_qda_infold_r003", "eq206_qda_infold_lw"])]
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, ms) in zip(axes, panels):
        for i, m in enumerate([m for m in ms if m in set(summ["model"])]):
            d = summ[(summ["model"] == m) & (summ["n_failed"] == 0)].sort_values("n_labels")
            c = "#0b0b0b" if m == "s003_gmm2_hv" else slots[i % len(slots)]
            ax.plot(d["n_labels"], d["auc_mean"], "-", color=c, lw=2, marker="o", ms=4, label=LABELS[m])
            ax.fill_between(d["n_labels"], d["auc_mean"] - 2 * d["auc_se_cluster"], np.minimum(1, d["auc_mean"] + 2 * d["auc_se_cluster"]),
                            color=c, alpha=0.12, lw=0)
        ax.axhline(0.99, color="#8a8984", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(a.budgets)
        ax.set_xticklabels([str(b) for b in a.budgets])
        ax.set_ylim(0.5, 1.005)
        ax.set_xlabel("labelled training rows (log scale)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("validation-fold ROC AUC (mean; band = 2 fold-cluster SE)")
    fig.suptitle(f"Label budget re-run (R441): official repeats {a.repeats[0]}-{a.repeats[1] - 1}, new label draws; "
                 "only fully defined points plotted", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "r441_label_budget.png", dpi=160)
    run.save_figure("label_budget", fig)
    print(wide.round(4).to_string(), flush=True)
    print(hits.to_string(), flush=True)
    run.finish(extra_summary={"wall_seconds": round(time.time() - t0, 1), "n_evals": int(len(df)),
                              "n_failed_evals": int(df["auc"].isna().sum()), "n_folds": len(splits),
                              "first_hits": hits.to_dict(orient="records")})
