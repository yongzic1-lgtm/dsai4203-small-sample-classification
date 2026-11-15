"""Figures and small tables for the core report (reports/report_core.md), built from archived files only.

    .venv\\Scripts\\python.exe scripts\\make_core_figures.py            # write reports/figures/core_*.png, reports/tables/core_*
    .venv\\Scripts\\python.exe scripts\\make_core_figures.py --check    # compute and print the tables, write nothing

Nothing is fitted and no Run is created. Inputs (read-only):
  registry/submissions.csv                       S001-S006: CV mean / sd over repeats, pre-registered expected_public, public_lb
  experiments/20261022-063908_e3_leakage/summary.json   leaky CV of the E3 model (S004)
  reports/tables/core_submission_history.csv     optional cross-check (scripts/submission_history.py)
  reports/tables/r441_label_budget.csv           R441 label-budget re-run, run 20261022-182022_r441_label_budget_rerun
  reports/tables/r404_boundary.csv               R404 boundary-row AUC on 222 replicas, run 20261022-194238_r404_inductive_pooled
Outputs (this script may overwrite them; docs/ARCHIVE_RULES.md section 2):
  reports/figures/core_journey.png               Figure 1 of the core report
  reports/tables/core_journey.{csv,md}           its table view = Table 1 of the core report
  reports/figures/core_unlabeled_value.png       Figure 2: (a) label budget (R441), (b) boundary rows (R404)
  reports/tables/core_label_budget.{csv,md}      table view of panel (a)
  reports/tables/core_boundary.{csv,md}          table view of panel (b)
  reports/tables/core_manifest.json              inputs with sha256, outputs, conventions

Public-LB standard error: Hanley & McNeil (1982) via ssc.validation.auc_standard_error, evaluated at the
pre-registered expected AUC for 395 public rows split 198 positive / 197 negative, the convention of the
audited journey table R440 (reports/tables/r440_journey.csv). z(exp) = (public - expected) / SE and
z(CV) = (public - CV mean) / SE. (scripts/record_lb.py prints a slightly different SE: at the observed public
score, 179/216 rows from the training positive rate; registry/beliefs.md quotes that one.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common as C
from ssc import paths
from ssc.tracking import now_iso

SOURCE = "raw"
N_POS, N_NEG = 198, 197          # 395 public rows (2% of 19,750); test set is ~50% positive (as R440)
E3_RUN = "20261022-063908_e3_leakage"
R441_RUN = "20261022-182022_r441_label_budget_rerun"
R404_RUN = "20261022-194238_r404_inductive_pooled"

# presentation only: short names for the report (numbers come from the registry)
SUB_LABEL = {  # (model name in Table 1, unlabelled test features used, x label in Figure 1)
    "S001": ("L1 logistic, all features", "no", "S001\nL1, all 150"),
    "S002": ("QDA, label-free block", "block", "S002\nQDA, block"),
    "S003": ("Transductive GMM (final)", "yes", "S003\nGMM"),
    "S004": ("E3: corr top-20 + L2", "no", "S004\nE3 top-20"),
    "S005": ("Strict QDA, nested reg", "no", "S005\nstrict QDA"),
    "S006": ("Self-training QDA-LW", "yes", "S006\nself-train"),
}
LEFT_PANEL = ("S001", "S004")                 # linear models, labels only
RIGHT_PANEL = ("S002", "S003", "S005", "S006")

# Figure 2: the six procedures that both R441 (label budget) and R404 (boundary rows) evaluated, one row each:
# (label, colour role, `model` in r441_label_budget.csv, `proc` in r404_boundary.csv)
ROWS = [
    ("S003 GMM (test features)", "s003", "s003_gmm2_hv", "s003_gmm2_hv"),
    ("S006 self-training QDA-LW (test features)", "s006", "r209_selftrain_qda_lw", "r209_selftrain_qda_lw"),
    ("Bayesian QDA, train+test block (r255)", "other", "r255_bqda_eb_hv", "r255_bqda_eb_hv"),
    ("QDA, log-loss reg (r210, labels only)", "other", "r210_qda_block_nll", "r210_qda_block_nll"),
    ("S005 strict QDA (r201, labels only)", "s005", "r201_qda_block", "r201_qda_block"),
    ("Bayesian QDA, in-fold block (r255, labels only)", "other", "r255_bqda_eb_trainonly", "r255_bqda_eb_trainonly"),
]
# table views also list these (not in the figure)
# reg 0.003 was taken from R201's nested choices on all 250 labels (research/reports/r2_audit.md 2.3), so these two
# rows borrow information from the full label set; the report quotes them only with that caveat
BUDGET_EXTRA = [("eq206_qda_tt_r003", "QDA reg 0.003 (from R201, all 250 labels), train+test block (R206 model)"),
                ("eq206_qda_infold_r003", "strict QDA reg 0.003 (from R201, all 250 labels), in-fold block (R206 model)")]
BOUNDARY_EXTRA = [("bayes", "True class probability (Bayes posterior, ceiling)"),
                  ("x_qda_block_nll_ext", "QDA, ext. reg grid (r930, labels only)")]

# colours: dataviz reference palette, light mode; categorical slots 1-3 follow the entities S003, S005, S006
# (validated: scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --pairs all passes; aqua < 3:1 contrast,
# so every series is also named in a legend and in the table view)
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COL = {"s003": "#2a78d6", "s005": "#eb6834", "s006": "#1baf7a", "other": INK2, "bayes": INK}
PUBLIC = "#2a78d6"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def need(p: Path) -> Path:
    if not p.exists():
        C.die(f"input {p} not found (this script reads the real archive; is $SSC_ARCHIVE_ROOT set?)", 2)
    return p


def signed(v: float, nd: int) -> str:
    """Signed number with a typographic minus (U+2212), as printed in the report."""
    return f"{v:+.{nd}f}".replace("-", "−")


def hm_se(auc: float) -> float:
    from ssc.validation import auc_standard_error
    return auc_standard_error(float(auc), N_POS, N_NEG)


# ---------------------------------------------------------------------------------------------- tables

def journey_table(subs_csv: Path, hist_csv: Path) -> pd.DataFrame:
    s = pd.read_csv(subs_csv, dtype=str).fillna("")
    s = s[s["public_lb"] != ""]
    rows = []
    for _, r in s.iterrows():
        sid = r["sub_id"]
        name, tf, _ = SUB_LABEL.get(sid, (r["description"], "", sid))
        cv, sd = float(r["cv_auc_mean"]), float(r["cv_auc_std_repeat"])
        exp, pub = float(r["expected_public"]), float(r["public_lb"])
        se = hm_se(exp)
        rows.append({"sub_id": sid, "day_utc": r["submitted_at_utc"][:10], "run_id": r["run_id"], "model": name,
                     "test_features": tf, "cv_auc_mean": cv, "cv_auc_sd": sd, "expected_public": exp,
                     "public_lb": pub, "hm_se_at_expected": se, "public_minus_expected": pub - exp,
                     "public_minus_cv": pub - cv, "z_expected": (pub - exp) / se, "z_cv": (pub - cv) / se,
                     "sha256_prefix": r["sha256"]})
    t = pd.DataFrame(rows)
    if hist_csv.exists():  # the same numbers must come out of scripts/submission_history.py
        h = pd.read_csv(hist_csv).set_index("sub_id")
        for _, r in t.iterrows():
            if r["sub_id"] in h.index:
                hr = h.loc[r["sub_id"]]
                for a, b in (("cv_auc_mean", "cv_auc_mean"), ("expected_public", "expected_public"),
                             ("public_lb", "public_lb")):
                    if abs(float(hr[b]) - r[a]) > 1e-9:
                        C.die(f"{r['sub_id']}: {a} differs between submissions.csv and {hist_csv.name}", 1)
    return t


def journey_md(t: pd.DataFrame) -> pd.DataFrame:
    """Report Table 1 (7 columns, fits the page); day, test-feature use and the SE are in core_journey.csv."""
    return pd.DataFrame({
        "Sub": t["sub_id"],
        "Model": t["model"],
        "CV AUC ± sd": [f"{m:.4f} ± {s:.4f}" for m, s in zip(t["cv_auc_mean"], t["cv_auc_sd"])],
        "Expected": [f"{v:.4f}" for v in t["expected_public"]],
        "Public": [f"{v:.5f}" for v in t["public_lb"]],
        "z(exp)": [signed(v, 2) for v in t["z_expected"]],
        "z(CV)": [signed(v, 2) for v in t["z_cv"]],
    })


def budget_models() -> list[tuple[str, str, bool]]:
    """(r441 model, label, in the figure) for the table view."""
    return [(m, lab, True) for lab, _, m, _ in ROWS] + [(m, lab, False) for m, lab in BUDGET_EXTRA]


def budget_table(r441_csv: Path) -> pd.DataFrame:
    b = pd.read_csv(r441_csv)
    keep = [m for m, *_ in budget_models()]
    missing = sorted(set(keep) - set(b["model"]))
    if missing:
        C.die(f"{r441_csv.name} has no rows for {missing}", 1)
    b = b[b["model"].isin(keep)].copy()
    b["defined_all"] = (b["n_failed"] == 0) & b["auc_mean"].notna()
    return b[["model", "n_labels", "n_evals", "n_failed", "defined_all", "auc_mean", "auc_median", "auc_q10",
              "auc_se_cluster", "uses_test_features"]].reset_index(drop=True)


def first_hit(b: pd.DataFrame, model: str, level: float = 0.99):
    """Smallest budget whose mean AUC reaches `level`, among budgets where every evaluation is defined
    (the rule of R440/R441: a mean over the few fits that succeed would be biased)."""
    d = b[(b["model"] == model) & b["defined_all"] & (b["auc_mean"] >= level)]
    return int(d["n_labels"].min()) if len(d) else None


def budget_md(b: pd.DataFrame) -> pd.DataFrame:
    budgets = sorted(b["n_labels"].unique())
    out = {"Model": [], **{str(n): [] for n in budgets}, "First ≥ 0.99": []}
    for m, label, in_fig in budget_models():
        out["Model"].append(label + ("" if in_fig else " *"))
        for n in budgets:
            r = b[(b["model"] == m) & (b["n_labels"] == n)]
            out[str(n)].append(f"{r['auc_mean'].iloc[0]:.4f}" if len(r) and bool(r["defined_all"].iloc[0]) else "–")
        fh = first_hit(b, m)
        out["First ≥ 0.99"].append(str(fh) if fh else "never")
    return pd.DataFrame(out)


def boundary_table(r404_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(r404_csv).set_index("proc")
    procs = [(p, lab, role, True) for lab, role, _, p in ROWS] + [(p, lab, "other", False) for p, lab in BOUNDARY_EXTRA]
    missing = sorted({p for p, *_ in procs} - set(d.index))
    if missing:
        C.die(f"{r404_csv.name} has no rows for {missing}", 1)
    rows = []
    for proc, label, role, in_fig in procs:
        r = d.loc[proc]
        rows.append({"proc": proc, "label": label, "in_figure": in_fig, "inductive": bool(r["inductive"]),
                     "n_replicas": int(r["n_replicas"]), "boundary_rows_mean": float(r["n_bnd_rows_mean"]),
                     "auc": float(r["bnd_auc_clean_mean"]), "auc_mc_se": float(r["bnd_auc_clean_mc_se"]),
                     "s003_minus_proc": float(r["bnd_auc_clean_s003_minus_proc"]),
                     "share_s003_better": float(r["bnd_auc_clean_share_s003_better"]), "role": role})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def boundary_md(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Procedure": [lab + ("" if f else " *") for lab, f in zip(t["label"], t["in_figure"])],
        "Boundary-row AUC (MC SE)": [f"{a:.3f} ({s:.3f})" for a, s in zip(t["auc"], t["auc_mc_se"])],
        "S003 minus procedure": [("–" if p == "s003_gmm2_hv" else signed(v, 3)) for p, v in zip(t["proc"], t["s003_minus_proc"])],
        "Replicas where S003 is better": [("–" if p == "s003_gmm2_hv" else f"{v * n:.0f}/{n}")
                                          for p, v, n in zip(t["proc"], t["share_s003_better"], t["n_replicas"])],
    })


# ---------------------------------------------------------------------------------------------- figures

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Calibri", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 3,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "savefig.dpi": 220, "figure.dpi": 100, "legend.frameon": False,
    })
    return plt


FIG1_HEIGHT = 2.1    # inches; was 2.55 (core report page budget, 2026-11-01); the margins below keep the
                     # absolute space for the legend and the tick labels (0.46 in below, 0.54 in above the axes)


def fig_journey(t: pd.DataFrame, leaky_cv: float, out: Path) -> None:
    plt = style()
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(1, 2, figsize=(6.3, FIG1_HEIGHT),
                             gridspec_kw={"width_ratios": [2, 4.2], "wspace": 0.22})
    ti = t.set_index("sub_id")
    dx = 0.17
    for ax, subs, ylim, title in ((axes[0], LEFT_PANEL, (0.745, 0.895), "(a) Linear models, labels only"),
                                  (axes[1], RIGHT_PANEL, (0.9865, 1.0006), "(b) Models on the high-variance block")):
        for i, sid in enumerate(subs):
            r = ti.loc[sid]
            ax.errorbar(i - dx, r["cv_auc_mean"], yerr=r["cv_auc_sd"], fmt="o", ms=5.2, color=INK2, mec="white",
                        mew=0.8, elinewidth=1.4, capsize=0, zorder=3)
            ax.plot(i, r["expected_public"], marker="D", ms=5.2, mfc="white", mec=INK, mew=1.2, ls="none", zorder=3)
            lo = r["hm_se_at_expected"]
            hi = min(lo, 1.0 - r["public_lb"])          # an AUC cannot exceed 1: clip the bar at 1
            ax.errorbar(i + dx, r["public_lb"], yerr=[[lo], [hi]], fmt="s", ms=5.2, color=PUBLIC, mec="white",
                        mew=0.8, elinewidth=1.6, capsize=0, zorder=4)
            if sid == "S004":
                ax.plot(i - dx, leaky_cv, marker="o", ms=5.2, mfc="white", mec=INK2, mew=1.2, ls="none", zorder=3)
                ax.annotate("leaky CV (E3)", (i - dx, leaky_cv), xytext=(-6, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(-0.6, len(subs) - 0.4)
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels([SUB_LABEL[s][2] for s in subs])
        ax.set_ylim(*ylim)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", color=INK, pad=4)
    axes[0].set_ylabel("ROC AUC")
    axes[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.004))
    axes[1].axhline(1.0, color=AXIS, lw=0.8, zorder=1)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="none", color=INK2, mec="white", ms=5.5, label="CV mean ± sd (5-fold × 20)"),
               Line2D([], [], marker="D", ls="none", mfc="white", mec=INK, mew=1.2, ms=5.2, label="expected public (pre-registered)"),
               Line2D([], [], marker="s", ls="none", color=PUBLIC, mec="white", ms=5.5, label="public LB ± 1 SE (Hanley–McNeil)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
               columnspacing=1.4)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.459 / FIG1_HEIGHT, top=1 - 0.536 / FIG1_HEIGHT)
    fig.savefig(out)
    plt.close(fig)


def fig_unlabeled(b: pd.DataFrame, bt: pd.DataFrame, out: Path) -> None:
    """Two dot panels sharing one row per procedure: (a) labels needed for mean AUC >= 0.99 (R441),
    (b) AUC on the replicas' boundary rows (R404)."""
    plt = style()
    fig, (ax, bx) = plt.subplots(1, 2, sharey=True, figsize=(6.3, 2.15),
                                 gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    y = np.arange(len(ROWS))[::-1]
    bnd = bt.set_index("proc")
    for yi, (label, role, m, p) in zip(y, ROWS):
        c = COL[role]
        n = first_hit(b, m)
        ax.hlines(yi, 1, n, color=c, lw=1.6, zorder=2)
        ax.plot(n, yi, "o", ms=5.6, color=c, mec="white", mew=0.8, zorder=3)
        ax.text(n * 1.2, yi, str(n), va="center", fontsize=7.8, color=INK2)
        r = bnd.loc[p]
        bx.errorbar(r["auc"], yi, xerr=2 * r["auc_mc_se"], fmt="o", ms=5.6, color=c, mec="white", mew=0.8,
                    elinewidth=1.3, capsize=0, zorder=3)
        bx.text(r["auc"] + 2 * r["auc_mc_se"] + 0.005, yi, f"{r['auc']:.3f}", va="center", fontsize=7.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlim(1, 330)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("labelled training rows needed (log scale)")
    ax.set_title("(a) Labels for mean AUC ≥ 0.99 (R441)", loc="left", color=INK, pad=4)
    bayes = bnd.loc["bayes", "auc"]
    bx.axvline(bayes, color=INK, lw=0.9, zorder=1)
    bx.text(bayes - 0.004, 0.5, f"true probabilities\n(ceiling) {bayes:.3f}", ha="right", va="center", fontsize=8,
            color=INK2)
    bx.set_xlim(0.635, 0.83)
    bx.set_xticks([0.65, 0.70, 0.75, 0.80])
    bx.grid(axis="y", visible=False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("AUC on rows with true probability 0.1–0.9")
    bx.set_title("(b) Boundary rows, 222 replicas (R404)", loc="left", color=INK, pad=4)
    fig.subplots_adjust(left=0.335, right=0.985, bottom=0.21, top=0.885)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main

def write_md(df: pd.DataFrame, path: Path, sources: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!-- generated by scripts/make_core_figures.py from {sources}; do not edit by hand -->\n")
        f.write(df.to_markdown(index=False, disable_numparse=True) + "\n")


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compute and print only; write nothing")
    args = ap.parse_args()
    C.print_archive(SOURCE)

    tdir, fdir = paths.tables_dir(SOURCE), paths.figures_dir(SOURCE)
    subs_csv = need(paths.submissions_csv(SOURCE))
    hist_csv = tdir / "core_submission_history.csv"
    e3_json = need(paths.experiments_dir(SOURCE) / E3_RUN / "summary.json")
    r441_csv = need(tdir / "r441_label_budget.csv")
    r404_csv = need(tdir / "r404_boundary.csv")

    jt = journey_table(subs_csv, hist_csv)
    leaky_cv = float(json.loads(e3_json.read_text(encoding="utf-8"))["leaky_auc_mean"])
    bt_budget = budget_table(r441_csv)
    bt_bound = boundary_table(r404_csv)
    jmd, bmd, dmd = journey_md(jt), budget_md(bt_budget), boundary_md(bt_bound)

    from tabulate import tabulate
    print(f"\nTable 1 (core_journey; SE = Hanley-McNeil at the expected AUC, {N_POS}/{N_NEG} public rows):")
    print(tabulate(jmd.to_numpy().tolist(), headers=list(jmd.columns), tablefmt="simple", disable_numparse=True))
    print(f"\nE3 leaky CV (S004 model, {E3_RUN}): {leaky_cv:.4f}")
    print(f"\nLabel budget (R441, {R441_RUN}): first budget with mean AUC >= 0.99 (all evaluations defined):")
    for m, label, in_fig in budget_models():
        print(f"  {label:<50} {first_hit(bt_budget, m)}{'' if in_fig else '   (table only)'}")
    print(f"\nBoundary rows (R404, {R404_RUN}; mean {bt_bound['boundary_rows_mean'].iloc[0]:.0f} rows per replica):")
    print(tabulate(dmd.to_numpy().tolist(), headers=list(dmd.columns), tablefmt="simple", disable_numparse=True))
    if args.check:
        print("\n--check: nothing written")
        return 0

    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    out = {
        "core_journey.png": fdir / "core_journey.png", "core_unlabeled_value.png": fdir / "core_unlabeled_value.png",
        "core_journey.csv": tdir / "core_journey.csv", "core_journey.md": tdir / "core_journey.md",
        "core_label_budget.csv": tdir / "core_label_budget.csv", "core_label_budget.md": tdir / "core_label_budget.md",
        "core_boundary.csv": tdir / "core_boundary.csv", "core_boundary.md": tdir / "core_boundary.md",
    }
    jt.to_csv(out["core_journey.csv"], index=False)
    write_md(jmd, out["core_journey.md"], "registry/submissions.csv (SE: Hanley-McNeil at the expected AUC, 198/197)")
    bt_budget.to_csv(out["core_label_budget.csv"], index=False)
    write_md(bmd, out["core_label_budget.md"], f"reports/tables/r441_label_budget.csv (run {R441_RUN})")
    bt_bound.to_csv(out["core_boundary.csv"], index=False)
    write_md(dmd, out["core_boundary.md"], f"reports/tables/r404_boundary.csv (run {R404_RUN})")
    fig_journey(jt, leaky_cv, out["core_journey.png"])
    fig_unlabeled(bt_budget, bt_bound, out["core_unlabeled_value.png"])

    inputs = [subs_csv, e3_json, r441_csv, r404_csv] + ([hist_csv] if hist_csv.exists() else [])
    manifest = {
        "generated_by": "scripts/make_core_figures.py", "generated_at": now_iso(),
        "inputs": {C.rel(p).replace("\\", "/"): sha256(p) for p in inputs},
        "source_runs": {"core_journey": "registry/submissions.csv (S001-S006); " + E3_RUN,
                        "core_unlabeled_value (a)": R441_RUN, "core_unlabeled_value (b)": R404_RUN},
        "conventions": {"public_se": f"Hanley-McNeil at the pre-registered expected AUC, {N_POS} pos / {N_NEG} neg (as R440)",
                        "label_budget": "smallest budget whose mean validation AUC (80 folds x 5 label draws, R441) reaches "
                                        "0.99, among budgets where every evaluation is defined; rows marked * are table-only",
                        "boundary_rows": "test rows whose true class probability (Bayes posterior under the fitted "
                                         "generator) is in (0.1, 0.9), clean labels, 222 replicas (R404); "
                                         "bars = 2 MC SE; rows marked * are table-only"},
        "outputs": sorted(C.rel(p).replace("\\", "/") for p in out.values()),
    }
    (tdir / "core_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nwritten:")
    for p in list(out.values()) + [tdir / "core_manifest.json"]:
        print(f"  {C.rel(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
