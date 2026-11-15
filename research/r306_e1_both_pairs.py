"""R306 - extension E1 at full settings for BOTH candidate pairs in one job, plus a comparison.

1. extensions/e1_validation_protocol/run.py with its config.yaml (main pair, linear regime:
   in-fold |corr| top-30 vs top-50 + L2 logistic C=0.05; report prefix e1_);
2. research/r301_run_extension.py --ext e1 --prefix e1_qda_ with research/r305_e1_qda_pair.yaml
   (second pair, ceiling regime: QDA reg 0.1 vs 0.3 on the label-free block);
3. a kind="analysis" run r306_e1_pair_comparison that joins the two paired-difference tables
   by scheme -> reports/tables/r306_e1_pair_comparison.{csv,md}, reports/figures/r306_e1_pair_comparison.png.
Both pairs were chosen by the pre-registered rule of R300 (official CV gap closest to 0.01).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext12 --high-mem -- research/r306_e1_both_pairs.py --n-jobs -1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
p.add_argument("--n-jobs", default="-1")
p.add_argument("--quick", action="store_true", help="smoke test only (synthetic data)")
a = p.parse_args()

common = ["--data", a.data, "--n-jobs", str(a.n_jobs)] + (["--quick"] if a.quick else []) \
    + (["--author", a.author] if a.author else [])
PAIRS = {
    "linear": {"slug": "e1_validation_protocol", "label": "corr top-30 vs top-50 + L2 (AUC ~0.80)",
               "cmd": [sys.executable, str(ROOT / "extensions/e1_validation_protocol/run.py")] + common},
    "qda": {"slug": "e1_validation_protocol_qda_pair", "label": "QDA reg 0.1 vs 0.3 on block (AUC ~0.99)",
            "cmd": [sys.executable, str(ROOT / "research/r301_run_extension.py"), "--ext", "e1", "--prefix", "e1_qda_",
                    "--", "--config", str(ROOT / "research/r305_e1_qda_pair.yaml")] + common},
}


def newest_run(slug: str) -> Path:
    runs = sorted(d for d in paths.experiments_dir(a.data).glob(f"*_{slug}*")
                  if d.name.split("_", 1)[1] in (slug,) or d.name.split("_", 1)[1].startswith(slug + "-"))
    if not runs:
        raise FileNotFoundError(f"no run folder for {slug}")
    return runs[-1]


run_dirs = {}
for key, spec in PAIRS.items():
    print(f"[r306] === {key} pair: {' '.join(spec['cmd'][1:])}", flush=True)
    rc = subprocess.run(spec["cmd"], cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"[r306] {key} pair failed with exit code {rc}")
    run_dirs[key] = newest_run(spec["slug"])
    print(f"[r306] {key} pair -> {run_dirs[key].name}", flush=True)

# ---- comparison of the two pairs, scheme by scheme ----
cols = ["mean_diff", "std_diff", "std_diff_unpaired", "corr_AB", "mdd_2sd", "mdd_lo", "mdd_hi",
        "sign_agree_ref", "p_correct_at_target", "frac_nb_p_lt_0.05", "note"]
parts = []
for key, d in run_dirs.items():
    pdif = pd.read_csv(d / "paired_difference.csv")
    spr = pd.read_csv(d / "scheme_spread.csv")
    sA = spr[spr["model"] == "A"].set_index("scheme")["std"]
    t = pdif[["scheme", "label", "fits_per_decision"] + cols].copy()
    t["std_A"] = t["scheme"].map(sA)
    t = t.rename(columns={c: f"{key}_{c}" for c in cols + ["std_A"]})
    parts.append(t.set_index(["scheme", "label", "fits_per_decision"]))
comp = pd.concat(parts, axis=1).reset_index()
comp["mdd_ratio_linear_over_qda"] = comp["linear_mdd_2sd"] / comp["qda_mdd_2sd"]
comp["source_runs"] = "; ".join(f"{k}={v.name}" for k, v in run_dirs.items())

HYP = ("Both pairs differ by about 0.01 on the official CV; the MDD of each scheme is much smaller for the QDA "
       "pair near the ceiling than for the linear pair at AUC 0.80, so the scheme needed for a 0.01 decision "
       "depends on the regime of the two models.")
with Run("r306_e1_pair_comparison", {"name": "r306_e1_pair_comparison", "pairs": {k: v["label"] for k, v in PAIRS.items()},
                                     "source_runs": {k: v.name for k, v in run_dirs.items()}},
         kind="analysis", source=a.data, author=a.author, hypothesis=HYP) as run:
    run.save_table("pair_comparison", comp)
    out_t = run.dir / "report_preview" if a.quick else paths.tables_dir(a.data)
    out_f = run.dir / "report_preview" if a.quick else paths.figures_dir(a.data)
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_t / "r306_e1_pair_comparison.csv", index=False)
    md_cols = ["scheme", "fits_per_decision", "linear_std_A", "linear_std_diff", "linear_mdd_2sd", "linear_mdd_hi",
               "linear_p_correct_at_target", "qda_std_A", "qda_std_diff", "qda_mdd_2sd", "qda_mdd_hi",
               "qda_p_correct_at_target"]
    with open(out_t / "r306_e1_pair_comparison.md", "w", encoding="utf-8") as f:
        f.write(f"Source runs: linear pair = {run_dirs['linear'].name}; QDA pair = {run_dirs['qda'].name}; "
                f"comparison = {run.run_id}\n\n")
        f.write(comp[md_cols].to_markdown(index=False, floatfmt=".4f") + "\n")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(comp))
    w = 0.38
    for off, key, color in ((-w / 2, "linear", "#2a78d6"), (w / 2, "qda", "#eb6834")):
        sd = comp[f"{key}_std_diff"].to_numpy()
        lo = (comp[f"{key}_mdd_lo"] / 2).fillna(pd.Series(sd)).to_numpy()
        hi = (comp[f"{key}_mdd_hi"] / 2).fillna(pd.Series(sd)).to_numpy()
        ax.bar(x + off, np.where(sd > 0, sd, np.nan), width=w, color=color, label=PAIRS[key]["label"], zorder=2)
        ax.errorbar(x + off, sd, yerr=[sd - lo, hi - sd], fmt="none", ecolor="#52514e", elinewidth=1, capsize=2,
                    zorder=3)
    ax.axhline(0.005, color="#0b0b0b", linewidth=1.2, linestyle="--", label="std needed for MDD = 2 sd <= 0.01")
    ax.set_yscale("log")
    names = [f"{s}\n{int(f)} fits" for s, f in zip(comp["scheme"], comp["fits_per_decision"])]
    ax.set_xticks(x, names, fontsize=7)
    ax.set_ylabel("std of paired difference A - B (log)", fontsize=9)
    ax.set_title("Split-seed noise of a 0.01 decision depends on the pair (50 split seeds per scheme)",
                 fontsize=10, loc="left")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_f / "r306_e1_pair_comparison.png", dpi=200, bbox_inches="tight")
    run.save_figure("pair_comparison", fig)
    run.finish(extra_summary={"source_runs": {k: v.name for k, v in run_dirs.items()},
                              "comparison": comp.drop(columns=["source_runs"]).to_dict(orient="records")})

print(comp[md_cols].to_string(index=False))
print(f"\nsource runs: { {k: v.name for k, v in run_dirs.items()} }")
