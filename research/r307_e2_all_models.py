"""R307 - extension E2 (learning curve) at full settings for three models in one job, plus a comparison.

1. (a) extensions/e2_learning_curve/run.py with its config.yaml: supervised L1 logistic C=0.1,
       all 150 features (report prefix e2_);
2. (b) research/r301_run_extension.py --ext e2 --prefix e2_qda_ with research/r302_e2_qda_block.yaml:
       supervised QDA reg 0.1 on the label-free 13-column block;
3. (c) research/r301_run_extension.py --ext e2 --prefix e2_gmm_ with research/r303_e2_gmm.yaml:
       TransductiveGMM K=2 on the block (mixture from the 20,000 unlabeled feature rows; labels
       only map components);
4. a kind="analysis" run r307_e2_model_comparison: one figure with the three learning curves and
   one table with per-size means/stds and the extrapolation to 500 training rows
   -> reports/tables/r307_e2_model_comparison.{csv,md}, reports/figures/r307_e2_model_comparison.png.
The three E2 runs use identical subsets (subset_seed 2026), inner CV (5-fold x 4, seed 7000) and fit settings.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext12 --high-mem -- research/r307_e2_all_models.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", default="-1")
p.add_argument("--quick", action="store_true", help="smoke test only (synthetic data)")
p.add_argument("--only", nargs="*", default=None)
a = p.parse_args()

common = ["--data", a.data, "--n-jobs", str(a.n_jobs)] + (["--quick"] if a.quick else []) \
    + (["--author", a.author] if a.author else [])
wrap = [sys.executable, str(ROOT / "research/r301_run_extension.py"), "--ext", "e2"]
MODELS = {
    "l1_all": {"slug": "e2_learning_curve", "label": "(a) supervised L1 logistic C=0.1, all 150 features",
               "color": "#1baf7a", "cmd": [sys.executable, str(ROOT / "extensions/e2_learning_curve/run.py")] + common},
    "qda_hv": {"slug": "e2_learning_curve_qda_block", "label": "(b) supervised QDA reg 0.1, 13-column block",
               "color": "#eb6834",
               "cmd": wrap + ["--prefix", "e2_qda_", "--", "--config", str(ROOT / "research/r302_e2_qda_block.yaml")] + common},
    "gmm2_hv": {"slug": "e2_learning_curve_gmm", "label": "(c) transductive GMM K=2, block (labels map components)",
                "color": "#2a78d6",
                "cmd": wrap + ["--prefix", "e2_gmm_", "--", "--config", str(ROOT / "research/r303_e2_gmm.yaml")] + common},
}
keys = a.only or list(MODELS)


def newest_run(slug: str) -> Path:
    runs = sorted(d for d in paths.experiments_dir(a.data).glob(f"*_{slug}*")
                  if d.name.split("_", 1)[1] == slug or d.name.split("_", 1)[1].startswith(slug + "-"))
    if not runs:
        raise FileNotFoundError(f"no run folder for {slug}")
    return runs[-1]


run_dirs = {}
for key in keys:
    spec = MODELS[key]
    print(f"[r307] === {key}: {' '.join(spec['cmd'][1:])}", flush=True)
    rc = subprocess.run(spec["cmd"], cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"[r307] {key} failed with exit code {rc}")
    run_dirs[key] = newest_run(spec["slug"])
    print(f"[r307] {key} -> {run_dirs[key].name}", flush=True)

# ---- comparison ----
rows, ext_rows = [], []
for key, d in run_dirs.items():
    tab = pd.read_csv(d / "learning_curve_table.csv")
    ext = pd.read_csv(d / "extrapolation.csv")
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    lo_c = [c for c in ext.columns if c.startswith("boot_lo_")][0]
    hi_c = [c for c in ext.columns if c.startswith("boot_hi_")][0]
    for _, r in tab.iterrows():
        rows.append({"model": key, "run_id": d.name, "size": int(r["size"]), "x": r["x"], "cv_mean": r["cv_mean"],
                     "cv_std": r["cv_std"], "cv_min": r["cv_min"], "cv_max": r["cv_max"], "oos_mean": r["oos_mean"],
                     "oos_std": r["oos_std"], "frac_constant_folds": r["frac_constant_folds"]})
    for t in sorted(ext["x"].unique()):
        g = ext[ext["x"] == t].set_index("model")
        gain = summ.get("gain", {}).get("pl", {})
        ext_rows.append({"model": key, "run_id": d.name, "x": t,
                         "power_law_point": g.loc["power_law", "point"],
                         "power_law_median": g.loc["power_law", "boot_median"],
                         "power_law_lo": g.loc["power_law", lo_c], "power_law_hi": g.loc["power_law", hi_c],
                         "log_linear_point": g.loc["log_linear", "point"],
                         "envelope_lo": g.loc["envelope", lo_c], "envelope_hi": g.loc["envelope", hi_c],
                         "at_bounds": g.loc["power_law", "at_bounds"],
                         "gain_250_500_median": gain.get("median"), "gain_lo": gain.get("lo"),
                         "gain_hi": gain.get("hi"), "p_gain_pos": gain.get("frac_positive")})
curves = pd.DataFrame(rows)
extr = pd.DataFrame(ext_rows)

HYP = ("Labels needed for the same AUC differ by orders of magnitude between the three models: the L1 "
       "baseline is still rising at 200 training rows, supervised QDA approaches the ceiling only near 200 rows, "
       "and the transductive GMM is flat at 1.0 from the smallest size.")
with Run("r307_e2_model_comparison", {"name": "r307_e2_model_comparison",
                                      "models": {k: MODELS[k]["label"] for k in run_dirs},
                                      "source_runs": {k: v.name for k, v in run_dirs.items()}},
         kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    run.save_table("curves", curves)
    run.save_table("extrapolation", extr)
    out_t = run.dir / "report_preview" if a.quick else paths.tables_dir(a.data)
    out_f = run.dir / "report_preview" if a.quick else paths.figures_dir(a.data)
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_t / "r307_e2_model_comparison.csv", index=False)
    extr.to_csv(out_t / "r307_e2_extrapolation_comparison.csv", index=False)
    wide = curves.pivot(index="size", columns="model", values=["cv_mean", "cv_std"])
    wide.columns = [f"{m}_{s}" for s, m in wide.columns]
    with open(out_t / "r307_e2_model_comparison.md", "w", encoding="utf-8") as f:
        f.write("Source runs: " + "; ".join(f"{k} = {v.name}" for k, v in run_dirs.items())
                + f"; comparison = {run.run_id}\n\n")
        f.write(wide.reset_index().to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write(extr[["model", "x", "power_law_point", "power_law_lo", "power_law_hi", "envelope_lo", "envelope_hi",
                      "log_linear_point", "at_bounds", "gain_250_500_median", "p_gain_pos"]]
                .to_markdown(index=False, floatfmt=".4f") + "\n")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, d in run_dirs.items():
        c = curves[curves["model"] == key].sort_values("x")
        ax.errorbar(c["x"], c["cv_mean"], yerr=c["cv_std"], fmt="o-", color=MODELS[key]["color"], ms=5, capsize=3,
                    linewidth=1.6, label=f"{MODELS[key]['label']}")
        e = extr[(extr["model"] == key) & (extr["x"] == 500)]
        if len(e):
            e = e.iloc[0]
            ax.plot([500], [e["power_law_point"]], marker="D", color=MODELS[key]["color"], ms=6)
            ax.plot([500, 500], [e["envelope_lo"], e["envelope_hi"]], color=MODELS[key]["color"], linewidth=3,
                    alpha=0.5, solid_capstyle="butt")
            ax.plot([c["x"].max(), 500], [c["cv_mean"].iloc[-1], e["power_law_point"]], color=MODELS[key]["color"],
                    linewidth=1.0, linestyle="--")
    ax.set_xlabel("training rows behind each score (effective size n*4/5; 500 = extrapolation target)", fontsize=9)
    ax.set_ylabel("CV AUC inside subset (mean +/- std over 10 subsets)", fontsize=9)
    ax.set_title("Learning curves of three models on the same subsets (diamond + bar: AUC(500) power law and envelope)",
                 fontsize=9.5, loc="left")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_f / "r307_e2_model_comparison.png", dpi=200, bbox_inches="tight")
    run.save_figure("model_comparison", fig)
    run.finish(extra_summary={"source_runs": {k: v.name for k, v in run_dirs.items()},
                              "curves": curves.to_dict(orient="records"),
                              "extrapolation": extr.to_dict(orient="records")})

print(wide.reset_index().to_string(index=False))
print(extr.to_string(index=False))
