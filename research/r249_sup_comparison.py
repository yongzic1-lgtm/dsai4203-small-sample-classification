"""R249 - compact comparison of the supervised-only and semi-supervised track (r200-r248).

Analysis run. Collects every r20x experiment run produced by earlier jobs of THIS Colab
session (job folders /content/jobs/<job>/archive/experiments, i.e. the parent of this job's
SSC_ARCHIVE_ROOT; extra folders with --search), and recomputes the transductive reference
TransductiveGMM K=2 on the block (R003 gmm2_hv, archived as 20261022-050708_r003_gmm2_hv) on
the identical official splits. For every candidate:
  * mean CV AUC +- std across repeats, pooled OOF log-loss, n_configs (nested grid size);
  * paired comparison with the best STRICTLY supervised model (no test features): mean diff,
    split SE (= std of repeat-mean diffs / sqrt(20)), fold win rate, Nadeau-Bengio corrected p;
  * paired comparison with TransductiveGMM;
  * proxy evidence on the 19,750 test rows: AUC of the candidate's test scores against the
    GMM's hard labels (pseudo-AUC; S003 = this GMM scored 0.99923 on the public LB), and the
    Spearman correlation with the GMM posterior on its ~1% boundary rows (posterior in (0.1, 0.9)).
Writes reports/tables/r249_comparison.{csv,md}, reports/figures/r249_comparison.png.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r249_sup_comparison.py
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from lib_sup import plot_label_budget, prob_metrics, setup  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import TransductiveGMM  # noqa: E402
from ssc.validation import cross_validate, paired_compare  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--search", nargs="*", default=[], help="extra experiments/ folders to scan")
p.add_argument("--n-jobs", type=int, default=-1)
a = p.parse_args()

scheme, ds, splits = setup(a.data)
roots = [Path(s) for s in a.search]
if os.environ.get("SSC_ARCHIVE_ROOT"):
    jobs_dir = Path(os.environ["SSC_ARCHIVE_ROOT"]).resolve().parents[1]
    roots += sorted(jobs_dir.glob("*/archive/experiments"))
roots.append(paths.experiments_dir(a.data))

found = {}
for root in roots:
    for d in sorted(root.glob("*_r2[0-4][0-9]_*")):
        if not (d / "summary.json").exists() or not (d / "fold_scores.csv").exists():
            continue
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        if summ.get("status") != "done" or not summ.get("official_cv") or cfg.get("cv", {}).get("n_repeats") != 20:
            continue
        name = cfg["name"]
        if name in found and found[name]["run_id"] >= d.name:  # keep the latest run of a name
            continue
        found[name] = {"run_id": d.name, "dir": d, "summary": summ, "config": cfg}
print(f"[r249] scanned {len(roots)} folders; found {len(found)} official r20x runs", flush=True)

# transductive reference, recomputed on the identical splits
gmm_cv = cross_validate(TransductiveGMM("highvar", 2, source=a.data), ds.X, ds.y, splits, scheme, X_test=ds.X_test,
                        n_jobs=a.n_jobs)
gmm_test = gmm_cv.test_pred
pseudo = (gmm_test > 0.5).astype(int)
boundary = (gmm_test > 0.1) & (gmm_test < 0.9)
print(f"[r249] GMM reference: auc {gmm_cv.fold_scores['auc'].mean():.4f}; boundary test rows {int(boundary.sum())}", flush=True)

rows, fold_tables = [], {}
for name, r in found.items():
    fs = pd.read_csv(r["dir"] / "fold_scores.csv")
    s, cfg = r["summary"], r["config"]
    tp = np.load(r["dir"] / "test_pred.npy") if (r["dir"] / "test_pred.npy").exists() else None
    fold_tables[name] = fs
    rows.append({"candidate": name, "run_id": r["run_id"], "group": cfg.get("group", s.get("group")),
                 "features": cfg.get("features"), "uses_unlabeled_test_features": bool(cfg.get("uses_unlabeled_test_features")),
                 "n_configs": int(cfg.get("n_configs", 1) or 1), "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
                 "oof_auc_mean": s.get("oof_auc_mean"), "oof_logloss": s.get("oof_logloss"),
                 "test_pseudo_auc_vs_gmm": float(roc_auc_score(pseudo, tp)) if tp is not None else np.nan,
                 "test_spearman_gmm_boundary": float(spearmanr(tp[boundary], gmm_test[boundary]).statistic) if tp is not None else np.nan,
                 "description": cfg.get("description", "")})
gmm_name = "R003_gmm2_hv (recomputed)"
fold_tables[gmm_name] = gmm_cv.fold_scores
rows.append({"candidate": gmm_name, "run_id": "20261022-050708_r003_gmm2_hv", "group": "transductive",
             "features": "block(train+test) + 20,000 unlabeled rows", "uses_unlabeled_test_features": True, "n_configs": 1,
             "auc_mean": float(gmm_cv.fold_scores["auc"].mean()), "auc_std_repeat": float(gmm_cv.fold_scores.groupby("repeat")["auc"].mean().std(ddof=1)),
             "oof_auc_mean": gmm_cv.summary().get("oof_auc_mean"), "oof_logloss": prob_metrics(gmm_cv.oof, ds.y)["oof_logloss"],
             "test_pseudo_auc_vs_gmm": 1.0, "test_spearman_gmm_boundary": 1.0,
             "description": "TransductiveGMM K=2 full covariance on the block (R003 / S003)"})
res = pd.DataFrame(rows)

strict = res[res["group"].str.startswith("supervised") & ~res["uses_unlabeled_test_features"]]
# ties (identical fold scores, e.g. QDA on the block vs on its PCA rotation) go to the smaller grid
best_sup = (strict.assign(_a=-strict["auc_mean"].round(6)).sort_values(["_a", "n_configs", "candidate"])
            .iloc[0]["candidate"])
print(f"[r249] best strictly supervised: {best_sup}", flush=True)
pc_rows = []
for name in res["candidate"]:
    c1 = paired_compare(fold_tables[name], fold_tables[best_sup]) if name != best_sup else None
    c2 = paired_compare(fold_tables[name], fold_tables[gmm_name]) if name != gmm_name else None
    pc_rows.append({"candidate": name,
                    "diff_vs_best_sup": c1["mean_diff"] if c1 else 0.0, "se_split_vs_best_sup": c1["se_split"] if c1 else 0.0,
                    "win_rate_vs_best_sup": c1["win_rate_fold"] if c1 else 0.5, "p_corr_vs_best_sup": c1["corrected_p"] if c1 else 1.0,
                    "diff_vs_gmm": c2["mean_diff"] if c2 else 0.0, "se_split_vs_gmm": c2["se_split"] if c2 else 0.0,
                    "p_corr_vs_gmm": c2["corrected_p"] if c2 else 1.0})
res = res.merge(pd.DataFrame(pc_rows), on="candidate")
res["verdict_vs_best_sup"] = np.where(res["candidate"] == best_sup, "reference",
                                      np.where((res["diff_vs_best_sup"].abs() > 2 * res["se_split_vs_best_sup"]) & (res["p_corr_vs_best_sup"] < 0.05),
                                               np.where(res["diff_vs_best_sup"] > 0, "better", "worse"), "not distinguishable"))
order = {"supervised-linear": 0, "supervised-generative": 1, "supervised-kernel": 2, "supervised-trees": 3,
         "selection": 4, "semi-supervised": 5, "transductive": 6}
res["group_order"] = res["group"].map(order).fillna(9)
res = res.sort_values(["group_order", "auc_mean"], ascending=[True, False]).drop(columns="group_order").reset_index(drop=True)

n_total_configs = int(res.loc[res["group"] != "transductive", "n_configs"].sum())
n_strict = int(len(strict))
cfg = {"name": "r249_sup_comparison", "candidates": list(res["candidate"]), "best_strict_supervised": best_sup,
       "reference_transductive": "TransductiveGMM('highvar', 2), recomputed on the official splits",
       "model_code": "research/r249_sup_comparison.py", "cv": scheme.to_dict()}
H = ("Summary analysis (no new model). Expected: the best strictly supervised model is a QDA variant at ~0.995; "
     "every linear and tree model is clearly worse; sklearn semi-supervised baselines lie between QDA and the "
     "TransductiveGMM; the GMM is better than the best supervised model by ~0.004-0.005 AUC with p_corr < 0.05.")
with Run("r249_sup_comparison", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=H) as run:
    run.save_table("comparison", res)
    tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    res.to_csv(tables / "r249_comparison.csv", index=False)
    show = res[["candidate", "group", "features", "n_configs", "auc_mean", "auc_std_repeat", "oof_logloss",
                "diff_vs_best_sup", "se_split_vs_best_sup", "p_corr_vs_best_sup", "verdict_vs_best_sup",
                "test_pseudo_auc_vs_gmm", "run_id"]].copy()
    for c in ("auc_mean", "auc_std_repeat", "oof_logloss", "diff_vs_best_sup", "se_split_vs_best_sup", "test_pseudo_auc_vs_gmm"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    show["p_corr_vs_best_sup"] = res["p_corr_vs_best_sup"].map(lambda v: f"{v:.3g}")
    (tables / "r249_comparison.md").write_text(
        f"Supervised-only vs semi-supervised vs transductive (official 5-fold x 20 CV; paired vs best strictly "
        f"supervised = {best_sup}; analysis run {run.run_id}). Configurations tried in this track: {n_total_configs} "
        f"(sum of nested grid sizes over {len(res) - 1} candidates).\n\n" + show.to_markdown(index=False) + "\n",
        encoding="utf-8")

    # figure: 1 - AUC on a log axis (all models cluster near 1), colour = how unlabeled rows are used
    col = {"none": "#2a78d6", "label-free block only": "#1baf7a", "semi-supervised": "#eb6834", "transductive": "#0b0b0b"}

    def use_kind(r):
        if r["group"] == "transductive":
            return "transductive"
        if r["group"] == "semi-supervised":
            return "semi-supervised"
        return "label-free block only" if r["uses_unlabeled_test_features"] else "none"

    res["unlabeled_use"] = res.apply(use_kind, axis=1)
    floor = 2e-4
    err = (1 - res["auc_mean"]).clip(lower=floor)
    fig, ax = plt.subplots(figsize=(7.6, 0.24 * len(res) + 1.4))
    ypos = np.arange(len(res))[::-1]
    for k, c in col.items():
        m = res["unlabeled_use"] == k
        ax.errorbar(err[m], ypos[m], xerr=np.minimum(res.loc[m, "auc_std_repeat"].fillna(0), err[m] * 0.95), fmt="o",
                    color=c, ms=5, elinewidth=1.2, capsize=0, label={"none": "no test features (strict supervised)",
                                                                       "label-free block only": "block chosen on train+test features",
                                                                       "semi-supervised": "semi-supervised (20,000 rows)",
                                                                       "transductive": "transductive GMM (all folds AUC 1.0, drawn at floor)"}[k])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_yticks(ypos)
    ax.set_yticklabels([n.replace("r2", "R2", 1) for n in res["candidate"]], fontsize=7)
    ax.set_xlabel("1 - CV AUC (log scale; right = better)")
    ax.set_title(f"Supervised-only vs semi-supervised (official CV); best strict supervised: {best_sup}", fontsize=9)
    ax.grid(True, axis="x", color="#e6e5e0", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "r249_comparison.png", dpi=160)
    run.save_figure("comparison", fig)
    # label-budget figure redrawn from the archived R206 summary (latest r206 analysis run in the scanned folders)
    lb = sorted((d for root in roots for d in root.glob("*_r206_label_budget") if (d / "label_budget_summary.csv").exists()),
                key=lambda d: d.name)
    if lb:
        lbs = pd.read_csv(lb[-1] / "label_budget_summary.csv")
        plot_label_budget(lbs, figs / "r249_label_budget.png",
                          f"Value of unlabeled data: AUC vs number of labelled rows (official splits; R206 run {lb[-1].name})")
        lbs.to_csv(tables / "r249_label_budget_source.csv", index=False)
        run.save_table("label_budget_source", lbs)
        print(f"[r249] label-budget figure redrawn from {lb[-1].name}", flush=True)
    print(show.to_string(index=False), flush=True)
    run.finish(extra_summary={"best_strict_supervised": best_sup, "n_candidates": int(len(res) - 1),
                              "n_strict_supervised": n_strict, "n_configs_total": n_total_configs,
                              "gmm_boundary_rows": int(boundary.sum())})
