"""Figures for the four extensions (reports/extensions/e1-e4), built from archived files only.

    .venv\\Scripts\\python.exe scripts\\make_ext_figures.py                 # write reports/figures/ext_e*.png + tables
    .venv\\Scripts\\python.exe scripts\\make_ext_figures.py --only e2,e3     # some figures only (manifest is merged)
    .venv\\Scripts\\python.exe scripts\\make_ext_figures.py --check         # compute and print the tables, write nothing

Nothing is fitted and no Run is created: the script reads run folders and report tables, takes descriptive
statistics (means, quantiles, Gaussian kernel densities for drawing) and evaluates archived curve parameters.
It never overwrites the run-generated e1_* .. e4_* figures; its outputs all start with ext_.

Inputs (read-only)
  E1  experiments/20261022-064054_e1_validation_protocol/{estimates,paired_estimates,paired_difference,scheme_spread}.csv
      experiments/20261022-064148_e1_validation_protocol_qda_pair/{paired_estimates,paired_difference}.csv
  E2  experiments/20261022-070248_e2_learning_curve/{subset_scores,bootstrap,extrapolation}.csv, summary.json
      experiments/20261022-070339_e2_learning_curve_qda_block/{subset_scores,bootstrap,extrapolation}.csv, summary.json
      experiments/20261022-070445_e2_learning_curve_gmm/{subset_scores,extrapolation}.csv
      reports/tables/r941_e2_replica_truth.csv (R941), reports/tables/r942_e2_coverage.csv (R942)
  E3  experiments/20261022-063908_e3_leakage/{repeat_scores_grid,cells,null_repeat_scores,null_cells,leaderboard}.csv
      reports/tables/e3_leaderboard.csv (public score of S004, refreshed by the E3 run.py --refresh-lb)
  E4  experiments/20261022-071414_e4_feature_stability/{frequency,drop,null_sorted_curves}.csv, config.yaml
      reports/tables/r320_feature_table.csv (in_block: the 13 label-free high-variance columns; run R320)
Outputs (this script may overwrite them; docs/ARCHIVE_RULES.md section 2)
  reports/figures/ext_e1.png .. ext_e4.png      Figure 1 of each extension
  reports/tables/ext_e1_ridges.{csv,md}         table views (every plotted summary)
  reports/tables/ext_e2_fan.{csv,md}
  reports/tables/ext_e3_curves.{csv,md}
  reports/tables/ext_e4_curves.{csv,md}
  reports/tables/ext_manifest.json              inputs with sha256, outputs, conventions

Style: src/ssc/plotstyle.py (Times New Roman, printed size, fixed entity colours). Colours: logistic-regression
models and honest controls grey ink, QDA orange, transductive GMM blue, leaky CV red, k ramp blue light -> dark.
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
from ssc import plotstyle as ps
from ssc.tracking import now_iso

SOURCE = "raw"
E1_RUN = "20261022-064054_e1_validation_protocol"
E1Q_RUN = "20261022-064148_e1_validation_protocol_qda_pair"
E2_RUN = "20261022-070248_e2_learning_curve"
E2Q_RUN = "20261022-070339_e2_learning_curve_qda_block"
E2G_RUN = "20261022-070445_e2_learning_curve_gmm"
R941_RUN = "20261022-092201_r941_e2_replica_truth"
R942_RUN = "20261022-094142_r942_e2_coverage"
E3_RUN = "20261022-063908_e3_leakage"
E4_RUN = "20261022-071414_e4_feature_stability"
R320_RUN = "20261022-074624_r320_e4_stable_vs_block"

FIGS = ("e1", "e2", "e3", "e4")
QLO, QHI = 0.10, 0.90                 # descriptive bands: 10-90 % of the plotted replicates


# ------------------------------------------------------------------------------------------- helpers

class Inputs:
    """Every file read is recorded (path -> sha256) for the manifest."""

    def __init__(self) -> None:
        self.used: dict[str, list[str]] = {}

    def path(self, fig: str, p: Path) -> Path:
        if not p.exists():
            C.die(f"input {p} not found (this script reads the real archive; is $SSC_ARCHIVE_ROOT set?)", 2)
        self.used.setdefault(fig, [])
        if p not in self.used[fig]:
            self.used[fig].append(p)
        return p

    def csv(self, fig: str, p: Path, **kw) -> pd.DataFrame:
        return pd.read_csv(self.path(fig, p), **kw)

    def json(self, fig: str, p: Path) -> dict:
        return json.loads(self.path(fig, p).read_text(encoding="utf-8"))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_dir(run_id: str) -> Path:
    return paths.experiments_dir(SOURCE) / run_id


def check(cond: bool, msg: str) -> None:
    if not cond:
        C.die(f"consistency check failed: {msg}", 1)


def sig2(v: float) -> str:
    """Two significant digits, trailing zero kept (0.030, 0.0083), as in the E1 table."""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return "0"
    nd = max(0, 1 - int(np.floor(np.log10(abs(v)))))
    return f"{v:.{nd}f}"


def q(v, p):
    return float(np.quantile(np.asarray(v, dtype=float), p))


def write_md(df: pd.DataFrame, path: Path, sources: str, note: str = "") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!-- generated by scripts/make_ext_figures.py from {sources}; do not edit by hand -->\n")
        if note:
            f.write(note.rstrip() + "\n\n")
        f.write(df.to_markdown(index=False, disable_numparse=True) + "\n")


def fmt(df: pd.DataFrame, nd: int = 4) -> pd.DataFrame:
    """Rounded copy for the Markdown table views."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    return out


# =========================================================================================== E1

E1_SCHEMES = [("holdout20", "hold-out 20%"), ("holdout30", "hold-out 30%"), ("3fold", "3-fold"),
              ("5fold", "5-fold"), ("10fold", "10-fold"), ("5foldx10", "5-fold × 10"),
              ("5foldx20", "5-fold × 20"), ("5fold_noshuffle", "5-fold, no shuffle")]
E1_OFFICIAL = "5foldx20"


def e1_data(inp: Inputs) -> tuple[dict, pd.DataFrame]:
    lin, qda = run_dir(E1_RUN), run_dir(E1Q_RUN)
    est = inp.csv("e1", lin / "estimates.csv")
    spread = inp.csv("e1", lin / "scheme_spread.csv").set_index(["scheme", "model"])
    panels = {
        "a": {"title": "Model A alone", "xlabel": "AUC estimate of model A", "kind": "auc",
              "vals": {s: est[(est.scheme == s) & (est.model == "A")].sort_values("execution")["estimate"].to_numpy()
                       for s, _ in E1_SCHEMES},
              "center": float(spread.loc[(E1_OFFICIAL, "A"), "mean"]), "color": ps.color("baseline"),
              "run": E1_RUN},
    }
    for key, d, title, col, run in (("b", lin, "A − B, linear pair", ps.color("baseline"), E1_RUN),
                                    ("c", qda, "A − B, QDA pair", ps.color("qda"), E1Q_RUN)):
        pe = inp.csv("e1", d / "paired_estimates.csv")
        pdiff = inp.csv("e1", d / "paired_difference.csv").set_index("scheme")
        vals = {s: pe[pe.scheme == s].sort_values("execution")["diff"].to_numpy() for s, _ in E1_SCHEMES}
        for s, _ in E1_SCHEMES:     # the archived MDD is 2 sd (ddof 1) of these 50 differences
            v = vals[s]
            check(len(v) == 50, f"E1 {key} {s}: {len(v)} estimates")
            check(abs(2 * np.std(v, ddof=1) - pdiff.loc[s, "mdd_2sd"]) < 1e-9, f"E1 {key} {s}: MDD mismatch")
        panels[key] = {"title": title, "xlabel": "paired difference A − B", "kind": "diff", "vals": vals,
                       "center": float(pdiff.loc[E1_OFFICIAL, "mean_diff"]), "color": col, "pdiff": pdiff,
                       "run": run}
    fits = inp.csv("e1", lin / "paired_difference.csv").set_index("scheme")["fits_per_decision"]
    rows = []
    for key, p in panels.items():
        for s, name in E1_SCHEMES:
            v = p["vals"][s]
            r = {"panel": key, "quantity": "AUC of model A" if p["kind"] == "auc" else "A - B",
                 "pair": "QDA reg 0.1 vs 0.3" if key == "c" else "corr top-30 vs top-50 + L2",
                 "scheme": s, "label": name, "fits_per_decision": int(fits[s]), "n": len(v),
                 "mean": v.mean(), "sd": np.std(v, ddof=1), "q10": q(v, QLO), "median": q(v, 0.5),
                 "q90": q(v, QHI), "min": v.min(), "max": v.max(),
                 "band_lo": p["center"] - 0.01, "band_hi": p["center"] + 0.01, "run_id": p["run"]}
            if p["kind"] == "diff":
                pd_ = p["pdiff"].loc[s]
                r.update(mdd_2sd=pd_["mdd_2sd"], mdd_lo=pd_["mdd_lo"], mdd_hi=pd_["mdd_hi"],
                         resolves_0p01=bool(pd_["mdd_hi"] < 0.01))   # whole CI below 0.01 (bold in the figure)
            r["ridge_fill_t"] = _fits_t(int(fits[s]))       # fill lightness: 0 = 2 fits, 1 = 200 fits
            rows.append(r)
    return panels, pd.DataFrame(rows)


def _place(fig, x0: float, y0: float, w: float, h: float):
    """Axes at (x0, y0) with size (w, h), all in inches from the figure's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x0 / W, y0 / H, w / W, h / H))


def _mix(c0: str, c1: str, t: float) -> tuple:
    """Colour at fraction t of the straight RGB path from c0 to c1 (a one-hue lightness ramp)."""
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * float(np.clip(t, 0.0, 1.0)))


E1_H, E1_RUG, E1_BAR = 0.64, -0.13, -0.35       # ridge height, rug and MDD bar below each row baseline (row units)
E1_BAND = "#ebeae4"                              # +-0.01 band (with AXIS hairlines at both edges)
E1_RAMP = {"a": (ps.LIGHT, ps.INK2), "b": (ps.LIGHT, ps.INK2), "c": ("#f9d5c6", ps.ORANGE)}   # 2 -> 200 fits


def _fits_t(fits: float) -> float:
    """2 fits -> 0, 200 fits -> 1, on a log scale (fill lightness of the ridge)."""
    return float(np.log(fits / 2.0) / np.log(100.0))


def _ridge(ax, y0, vals, fill):
    """Half-violin above the row baseline (Gaussian KDE, Scott bandwidth, evaluated only on [min, max] of the
    data, i.e. cut = 0, scaled to height E1_H) and the 50 values as a rug below it."""
    from scipy.stats import gaussian_kde
    vals = np.asarray(vals, dtype=float)
    if np.ptp(vals) < 1e-12:                        # every seed gave the same split
        ax.plot([vals[0]], [y0 + E1_RUG], "o", ms=4.0, mfc=fill, mec=ps.INK2, mew=0.6, zorder=6)
        return
    xg = np.linspace(vals.min(), vals.max(), 400)
    dens = gaussian_kde(vals)(xg)
    dens = dens / dens.max() * E1_H
    ax.fill_between(xg, y0, y0 + dens, color=fill, lw=0, zorder=2)
    ax.plot(np.r_[xg[0], xg, xg[-1]], np.r_[y0, y0 + dens, y0], color=ps.INK2, lw=0.6, zorder=3,
            solid_joinstyle="round")
    ax.plot([xg[0], xg[-1]], [y0, y0], color=ps.INK2, lw=0.4, zorder=3)
    ax.scatter(vals, np.full(len(vals), y0 + E1_RUG), marker="|", s=11, color=ps.INK2, lw=0.5, alpha=0.75,
               zorder=4)


def fig_e1(panels: dict, out: Path) -> None:
    plt = ps.setup()
    fig = plt.figure(figsize=(ps.FULL_W, 2.5))
    Y0, HA = 0.36, 1.9
    axes = {"a": _place(fig, 0.98, Y0, 1.45, HA), "b": _place(fig, 2.57, Y0, 1.49, HA),
            "c": _place(fig, 4.52, Y0, 1.49, HA)}
    n = len(E1_SCHEMES)
    ys = {s: n - 1 - i for i, (s, _) in enumerate(E1_SCHEMES)}
    ytop = n - 1 + 0.88                             # band label and column header
    lims = {"a": (0.655, 0.94), "b": (-0.07, 0.075), "c": (-0.07, 0.075)}
    fits = panels["b"]["pdiff"]["fits_per_decision"]
    for key, ax in axes.items():
        p = panels[key]
        c0 = p["center"]
        yb0, yb1 = -0.52, n - 1 + 0.74                # the band stops below its label
        ax.fill_between([c0 - 0.01, c0 + 0.01], yb0, yb1, color=E1_BAND, lw=0, zorder=0)
        for xe in (c0 - 0.01, c0 + 0.01):
            ax.plot([xe, xe], [yb0, yb1], color=ps.AXIS, lw=0.5, zorder=1)
        ax.text(c0, ytop, "±0.01", fontsize=7.5, color=ps.MUTED, ha="center", va="center")
        if key == "b":                              # (c): 0 would double the band edge at 0.0026
            ps.hairline(ax, 0.0, "x", color=ps.GRID, lw=0.5, zorder=0)
        lo_c, hi_c = E1_RAMP[key]
        for s, _ in E1_SCHEMES:
            y0, v = ys[s], p["vals"][s]
            _ridge(ax, y0, v, _mix(lo_c, hi_c, _fits_t(fits[s])))
            if p["kind"] != "diff":
                continue
            if np.ptp(v) > 0:
                mdd, mu, yb = p["pdiff"].loc[s, "mdd_2sd"], v.mean(), y0 + E1_BAR
                ax.plot([mu - mdd, mu + mdd], [yb, yb], color=ps.INK2, lw=0.7, solid_capstyle="butt", zorder=5)
                for xe in (mu - mdd, mu + mdd):
                    ax.plot([xe, xe], [yb - 0.07, yb + 0.07], color=ps.INK2, lw=0.6, zorder=5)
                ax.plot([mu, mu], [yb - 0.05, yb + 0.05], color=ps.INK2, lw=0.6, zorder=5)
                res = bool(p["pdiff"].loc[s, "mdd_hi"] < 0.01)
                ax.text(1.03, y0 + 0.2, sig2(mdd), transform=ax.get_yaxis_transform(), fontsize=7.8,
                        color=ps.INK if res else ps.MUTED, fontweight="bold" if res else "normal",
                        ha="left", va="center")
            else:                                   # (b) text right of the dot, (c) left of it (MDD column)
                right = key == "b"
                ax.text(v[0] + (0.006 if right else -0.006), y0 + E1_RUG, "one split, sd 0", fontsize=7.5,
                        color=ps.INK2, va="center", ha="left" if right else "right")
        if p["kind"] == "diff":
            ax.text(1.03, ytop, "MDD", transform=ax.get_yaxis_transform(), fontsize=7.8, color=ps.INK2,
                    ha="left", va="center")
        else:
            v = p["vals"]["holdout20"]
            for xv in (v.min(), v.max()):           # the hold-out range quoted in the E1 text
                ax.text(xv, ys["holdout20"] + E1_BAR, f"{xv:.3f}", fontsize=7.5, color=ps.INK2, ha="center",
                        va="center")
        ax.set_xlim(*lims[key])
        ax.set_ylim(-0.52, n - 1 + 1.05)
        ax.set_yticks([])
        ps.clean(ax, grid="", left=False)
        ax.set_xlabel(p["xlabel"])
        if key == "a":
            ax.set_xticks([0.7, 0.8, 0.9])
        else:
            ax.set_xticks([-0.05, 0.0, 0.05])
            ax.set_xticklabels(["−0.05", "0", "0.05"])
        ps.panel_title(ax, p["title"], letter=key)
    ax0 = axes["a"]
    for s, name in E1_SCHEMES:                       # row labels: scheme, then fits per decision
        off = s == E1_OFFICIAL
        ax0.text(-0.04, ys[s] + 0.24, name, transform=ax0.get_yaxis_transform(), fontsize=8,
                 color=ps.INK if off else ps.INK2, fontweight="bold" if off else "normal", ha="right",
                 va="center")
        ax0.text(-0.04, ys[s] - 0.13, f"{int(fits[s])} fits", transform=ax0.get_yaxis_transform(), fontsize=7.2,
                 color=ps.MUTED, ha="right", va="center")
    ps.save(fig, out)
    plt.close(fig)


def e1_md(t: pd.DataFrame) -> pd.DataFrame:
    d = t.copy()
    d["MDD [95 % CI]"] = [("" if pd.isna(r.get("mdd_2sd")) else "0 (one split for every seed)"
                           if r["mdd_2sd"] == 0 else f"{sig2(r['mdd_2sd'])} [{sig2(r['mdd_lo'])}, "
                                                     f"{sig2(r['mdd_hi'])}]")
                          for _, r in d.iterrows()]
    d["resolves 0.01"] = ["" if pd.isna(r.get("mdd_hi")) else ("yes" if r["mdd_hi"] < 0.01 else "no")
                          for _, r in d.iterrows()]
    d = d[["panel", "quantity", "label", "fits_per_decision", "n", "mean", "sd", "q10", "median", "q90", "min",
           "max", "MDD [95 % CI]", "resolves 0.01"]]
    return fmt(d)


# =========================================================================================== E2

def _pl(p: dict, x):
    x = np.asarray(x, dtype=float)
    return p["a"] - p["b"] * x ** (-p["c"])


def e2_data(inp: Inputs) -> tuple[dict, pd.DataFrame]:
    out = {}
    rows = []
    truth = inp.csv("e2", paths.tables_dir(SOURCE) / "r941_e2_replica_truth.csv")
    cov = inp.csv("e2", paths.tables_dir(SOURCE) / "r942_e2_coverage.csv").set_index("model")
    for key, run, rmodel in (("l1", E2_RUN, "l1_all"), ("qda", E2Q_RUN, "qda_hv"), ("gmm", E2G_RUN, None)):
        d = run_dir(run)
        ss = inp.csv("e2", d / "subset_scores.csv")
        per = ss.groupby("x")["cv_auc"]
        curve = pd.DataFrame({"x": per.mean().index.to_numpy(), "n": per.size().to_numpy(),
                              "mean": per.mean().to_numpy(), "q10": per.quantile(QLO).to_numpy(),
                              "q90": per.quantile(QHI).to_numpy(), "min": per.min().to_numpy(),
                              "max": per.max().to_numpy()})
        e = {"subsets": ss, "curve": curve, "run": run}
        if key in ("l1", "qda"):
            summ = inp.json("e2", d / "summary.json")
            ext = inp.csv("e2", d / "extrapolation.csv")
            boot = inp.csv("e2", d / "bootstrap.csv")
            boot = boot[boot["pl_ok"].astype(bool)]
            pl = summ["power_law_params"]
            pw = ext[(ext.model == "power_law") & (ext.x == 500.0)].iloc[0]
            env = ext[(ext.model == "envelope") & (ext.x == 500.0)].iloc[0]
            check(abs(float(_pl(pl, 500.0)) - pw["point"]) < 1e-9, f"E2 {key}: power-law point")
            qb = np.quantile(boot["pl_pred_500"], [0.05, 0.95])
            check(abs(qb[0] - pw["boot_lo_0.05"]) < 1e-6 and abs(qb[1] - pw["boot_hi_0.95"]) < 1e-6,
                  f"E2 {key}: bootstrap interval")
            xg = np.linspace(200.0, 500.0, 61)
            curves = np.array([_pl({"a": a, "b": b, "c": c}, xg)
                               for a, b, c in boot[["pl_a", "pl_b", "pl_c"]].to_numpy()])
            tr = truth[(truth.model == rmodel) & (truth.t == 500)].iloc[0]
            cv = cov.loc[rmodel]
            e.update(pl=pl, xg=xg, fan_lo=np.quantile(curves, 0.05, axis=0), fan_hi=np.quantile(curves, 0.95, axis=0),
                     point=pw["point"], nom=(pw["boot_lo_0.05"], pw["boot_hi_0.95"]), env=(env["boot_lo_0.05"],
                     env["boot_hi_0.95"]), truth=float(tr["mean"]), cover_nom=float(cv["cover_power_law"]),
                     cover_env=float(cv["cover_envelope"]), rmse=float(cv["rmse_power_law_point"]),
                     bias=float(cv["bias_power_law_point"]))
            rows.append({"series": key, "kind": "at 500 rows", "x": 500.0, "n": len(boot), "mean": pw["point"],
                         "q10": np.nan, "q90": np.nan, "min": np.nan, "max": np.nan,
                         "nominal90_lo": e["nom"][0], "nominal90_hi": e["nom"][1], "envelope_lo": e["env"][0],
                         "envelope_hi": e["env"][1], "replica_truth": e["truth"], "cover_nominal": e["cover_nom"],
                         "cover_envelope": e["cover_env"], "replica_rmse": e["rmse"], "replica_bias": e["bias"],
                         "run_id": f"{run}; R941 {R941_RUN}; R942 {R942_RUN}"})
        if key == "gmm":                            # drawn as a flat dashed line at 1.0 to 500 rows
            ext = inp.csv("e2", d / "extrapolation.csv")
            pw = ext[(ext.model == "power_law") & (ext.x == 500.0)].iloc[0]
            env = ext[(ext.model == "envelope") & (ext.x == 500.0)].iloc[0]
            check(pw["point"] > 0.99999 and env["boot_lo_0.05"] > 0.99999, "E2 GMM: extrapolation at 1.0")
            rows.append({"series": key, "kind": "at 500 rows", "x": 500.0, "n": int(pw["n_boot_ok"]),
                         "mean": pw["point"], "nominal90_lo": pw["boot_lo_0.05"], "nominal90_hi": pw["boot_hi_0.95"],
                         "envelope_lo": env["boot_lo_0.05"], "envelope_hi": env["boot_hi_0.95"], "run_id": run})
        for _, r in curve.iterrows():
            rows.append({"series": key, "kind": "subsets", "x": r["x"], "n": int(r["n"]), "mean": r["mean"],
                         "q10": r["q10"], "q90": r["q90"], "min": r["min"], "max": r["max"], "run_id": run})
        out[key] = e
    t = pd.DataFrame(rows)
    order = {"subsets": 0, "at 500 rows": 1}
    t = t.sort_values(["series", "kind"], key=lambda s: s.map(order) if s.name == "kind" else s.map(
        {"l1": 0, "qda": 1, "gmm": 2})).reset_index(drop=True)
    return out, t


def _lead(ax, text, xy, xytext, relpos=(1.0, 0.5), ha="right", size=8):
    """ps.callout with the leader starting at `relpos` of the text box (e.g. its right edge), not its centre."""
    return ax.annotate(text, xy=xy, xytext=xytext, fontsize=size, color=ps.INK2, ha=ha, va="center",
                       arrowprops=dict(arrowstyle="-", color=ps.INK2, lw=0.5, shrinkA=1.5, shrinkB=2.0,
                                       relpos=relpos))


def _interval_bar(ax, x, lo, hi, color, lw, alpha=1.0, z=5):
    ax.plot([x, x], [lo, hi], color=color, lw=lw, alpha=alpha, solid_capstyle="butt", zorder=z)


E2_ZOOM = (0.845, 0.99)          # y range of the "500 rows" zoom panel (its box is drawn in (a))
E2_ENV = "#dcdbd5"               # envelope bar


def _nines_break(ax, t: float) -> None:
    """Axis-break mark on the left spine of a nines axis at -log10(1 - AUC) = t (between 0.9999 and the 1.0 cap)."""
    lo, hi = (float(ps.nines_fwd(v)) for v in ax.get_ylim())
    yf = (t - lo) / (hi - lo)
    from matplotlib.patches import Rectangle
    tr = ax.transAxes
    ax.add_patch(Rectangle((-0.03, yf - 0.018), 0.06, 0.036, transform=tr, facecolor="white", edgecolor="none",
                           clip_on=False, zorder=10))
    for dy in (-0.018, 0.018):
        ax.plot([-0.028, 0.028], [yf + dy - 0.014, yf + dy + 0.014], transform=tr, color=ps.INK2, lw=0.6,
                clip_on=False, zorder=11)


def fig_e2(d: dict, out: Path) -> None:
    plt = ps.setup()
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Patch, Rectangle
    fig = plt.figure(figsize=(ps.FULL_W, 2.4))
    Y0, H = 0.40, 1.72
    ax = _place(fig, 0.42, Y0, 2.22, H)
    zx = _place(fig, 2.76, Y0, 1.22, H)
    bx = _place(fig, 4.84, Y0, 1.40, H)
    rng = np.random.default_rng(11)                 # x jitter of the subset dots only
    grey, ink = ps.color("baseline"), ps.INK

    # (a) L1 baseline: subsets, 10-90 % band, mean; power law with its nominal bootstrap fan
    l1 = d["l1"]
    c, ss = l1["curve"], l1["subsets"]
    ax.fill_between(c["x"], c["q10"], c["q90"], color=ps.LIGHT, alpha=0.6, lw=0, zorder=1)
    ax.scatter(ss["x"] + rng.uniform(-5, 5, len(ss)), ss["cv_auc"], s=6, color=grey, alpha=0.45, lw=0, zorder=2)
    ax.plot(c["x"], c["mean"], color=grey, lw=1.2, marker="o", ms=3.4, mec="white", mew=0.5, zorder=4)
    xg = l1["xg"]
    ax.fill_between(xg, l1["fan_lo"], l1["fan_hi"], color=ps.MID, lw=0, zorder=1)
    for edge in (l1["fan_lo"], l1["fan_hi"]):
        ax.plot(xg, edge, color=ps.INK2, lw=0.5, zorder=2)
    ax.plot(xg, _pl(l1["pl"], xg), color=ink, lw=0.9, ls=(0, (3.5, 2)), zorder=3)
    z0, z1 = E2_ZOOM
    ax.add_patch(Rectangle((486, z0), 28, z1 - z0, fill=False, edgecolor=ps.INK2, lw=0.5, zorder=5))
    for ya, yb in ((z1, 1.0), (z0, 0.0)):           # zoom connectors from the box to the side panel
        fig.add_artist(ConnectionPatch(xyA=(514, ya), coordsA=ax.transData, xyB=(0.0, yb), coordsB=zx.transAxes,
                                       color=ps.AXIS, lw=0.5, zorder=0))
    h = [(Patch(facecolor=ps.LIGHT, alpha=0.6, lw=0),
          Line2D([], [], color=grey, lw=1.2, marker="o", ms=3.4, mec="white", mew=0.5)),
         (Patch(facecolor=ps.MID, edgecolor=ps.INK2, lw=0.5), Line2D([], [], color=ink, lw=0.9, ls=(0, (3.5, 2))))]
    ax.legend(h, ["10 subsets per size: mean, 10–90%", "power law, nominal 90% band"], loc="lower right",
              fontsize=7.8, handlelength=1.8, borderaxespad=0.1, labelspacing=0.3)
    ax.set_xlim(20, 520)
    ax.set_ylim(0.46, 1.0)
    ax.set_xticks([40, 120, 200, 300, 400, 500])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ps.clean(ax, grid="y")
    ax.set_xlabel("training rows behind each score (4n/5)")
    ax.set_ylabel("CV AUC")
    ps.panel_title(ax, "L1 logistic, all 150 features", letter="a")

    # zoom at 500 rows: nominal interval, point +- replica RMSE, envelope; replica truth across all three
    zx.set_xlim(-0.55, 2.55)
    zx.set_ylim(z0, z1)
    p, r, tru = l1["point"], l1["rmse"], l1["truth"]
    zx.plot([0, 0], list(l1["nom"]), color=ink, lw=2.6, solid_capstyle="butt", zorder=4)
    zx.plot([1, 1], [p - r, p + r], color=ps.INK2, lw=0.8, zorder=4)
    for yy in (p - r, p + r):
        zx.plot([0.84, 1.16], [yy, yy], color=ps.INK2, lw=0.8, zorder=4)
    zx.plot([2, 2], list(l1["env"]), color=E2_ENV, lw=8, solid_capstyle="butt", zorder=2)
    for xx in (0, 1):
        zx.plot([xx], [p], "D", ms=4.4, mfc="white", mec=ink, mew=0.8, zorder=6)
    zx.plot([-0.55, 2.55], [tru, tru], color=ink, lw=0.6, zorder=5)
    zx.plot([-0.38], [tru], "o", ms=4.2, mfc="white", mec=ink, mew=0.9, zorder=6)
    zx.text(-0.5, tru + 0.0045, "replica\ntruth " + f"{tru:.3f}", fontsize=7.5, color=ps.INK2, ha="left",
            va="bottom", linespacing=1.0)
    zx.text(0.17, p, f"{p:.3f}", fontsize=7.5, color=ps.INK2, ha="left", va="center")   # clear of the connector
    zx.text(0, l1["nom"][0] - 0.003, f"{l1['cover_nom']:.0%}", fontsize=7.8, color=ink, ha="center", va="top")
    zx.text(2, l1["env"][0] - 0.003, f"{l1['cover_env']:.0%}", fontsize=7.8, color=ink, ha="center", va="top")
    zx.text(1, p - r - 0.003, f"±{r:.3f}", fontsize=7.5, color=ps.INK2, ha="center", va="top")
    zx.set_xticks([0, 1, 2])
    zx.set_xticklabels(["nominal\n90%", "point\n± RMSE", "envelope"], fontsize=7.5, linespacing=1.0)
    zx.tick_params(axis="x", length=0, pad=3)
    zx.yaxis.tick_right()
    zx.set_yticks([0.86, 0.9, 0.94, 0.98])
    ps.clean(zx, grid="y", left=False)
    zx.spines["right"].set_visible(True)
    zx.spines["bottom"].set_visible(False)
    zx.tick_params(axis="y", which="both", right=True, left=False, length=2.5)
    ps.panel_title(zx, "500 rows, % covering truth")

    # (b) QDA and GMM near the ceiling, nines axis
    qd, gm = d["qda"], d["gmm"]
    oc, bc = ps.color("qda"), ps.color("gmm")
    c, ss = qd["curve"], qd["subsets"]
    bx.fill_between(c["x"], ps.clip_auc(c["q10"]), ps.clip_auc(c["q90"]), color=oc, alpha=0.18, lw=0, zorder=1)
    bx.scatter(ss["x"] + rng.uniform(-5, 5, len(ss)), ps.clip_auc(ss["cv_auc"]), s=6, color=oc, alpha=0.5, lw=0,
               zorder=2)
    bx.plot(c["x"], ps.clip_auc(c["mean"]), color=oc, lw=1.2, marker="o", ms=3.4, mec="white", mew=0.5, zorder=4)
    bx.fill_between(qd["xg"], ps.clip_auc(qd["fan_lo"]), ps.clip_auc(qd["fan_hi"]), color=oc, alpha=0.06, lw=0,
                    zorder=1)
    for edge in (qd["fan_lo"], qd["fan_hi"]):
        bx.plot(qd["xg"], ps.clip_auc(edge), color=oc, lw=0.5, zorder=2)
    bx.plot(qd["xg"], ps.clip_auc(_pl(qd["pl"], qd["xg"])), color=oc, lw=0.9, ls=(0, (3.5, 2)), zorder=3)
    xb = 500.0
    _interval_bar(bx, xb, *ps.clip_auc(qd["nom"]), color=ink, lw=2.2, z=5)
    bx.plot([xb], [qd["point"]], "D", ms=4.2, mfc="white", mec=ink, mew=0.8, zorder=6)
    bx.plot([xb], [qd["truth"]], "o", ms=4.2, mfc="white", mec=ink, mew=0.9, zorder=6)
    gc, gss = gm["curve"], gm["subsets"]
    check(bool((gss["cv_auc"] == 1.0).all()), "E2 GMM: every subset scores 1.0")
    bx.plot(gc["x"], ps.clip_auc(gc["mean"]), color=bc, lw=1.2, marker="o", ms=3.4, mec="white", mew=0.5, zorder=4)
    bx.plot([200, 500], ps.clip_auc([1.0, 1.0]), color=bc, lw=0.9, ls=(0, (3.5, 2)), zorder=3)
    bx.text(515, ps.nines_inv(4.6), "GMM: 1.0 in every subset", fontsize=7.8, color=ps.INK2, ha="right",
            va="center")
    bx.text(98, ps.nines_inv(1.3), "QDA, reg 0.1", fontsize=7.8, color=ps.INK2, ha="left", va="center")
    _lead(bx, f"point {qd['point']:.4f}", xy=(xb - 6, qd["point"]), xytext=(455, ps.nines_inv(3.2)), size=7.5)
    bx.text(482, qd["truth"], f"truth {qd['truth']:.4f}", fontsize=7.5, color=ps.INK2, ha="right", va="top")
    bx.set_xlim(20, 530)
    ps.nines(bx, "y", lo=0.9, cap=True, minor=True)
    _nines_break(bx, 4.5)
    bx.set_xticks([40, 200, 500])
    ps.clean(bx, grid="y")
    bx.set_xlabel("training rows (4n/5)")
    bx.set_ylabel("CV AUC (log scale of 1 − AUC)")
    ps.panel_title(bx, "QDA (block) and GMM", letter="b")
    ps.save(fig, out)
    plt.close(fig)


# =========================================================================================== E3

E3_KF = 5            # the official 5-fold splits (Table 1 of E3)
E3_K_LB = 20         # k_feat of the submitted model S004


def e3_data(inp: Inputs) -> tuple[dict, pd.DataFrame]:
    d = run_dir(E3_RUN)
    rep = inp.csv("e3", d / "repeat_scores_grid.csv")
    cells = inp.csv("e3", d / "cells.csv")
    nrep = inp.csv("e3", d / "null_repeat_scores.csv")
    ncells = inp.csv("e3", d / "null_cells.csv")
    lb = inp.csv("e3", d / "leaderboard.csv")
    lb_tab = inp.csv("e3", paths.tables_dir(SOURCE) / "e3_leaderboard.csv")
    public = float(lb_tab.loc[lb_tab.k_fold == E3_KF, "public_lb"].iloc[0])
    check(abs(float(lb.loc[lb.k_fold == E3_KF, "leaky_cv_mean"].iloc[0])
              - float(cells[(cells.k_feat == E3_K_LB) & (cells.k_fold == E3_KF)]["leaky_mean"].iloc[0])) < 1e-12,
          "E3 leaderboard vs cells")
    rows = []
    curves = {}
    for labels, df, cdf in (("real", rep, cells), ("permuted", nrep, ncells)):
        sub = df[df.k_fold == E3_KF]
        for proc in ("leaky", "honest"):
            g = sub.groupby("k_feat")[proc]
            cur = pd.DataFrame({"k_feat": g.mean().index.to_numpy(), "n": g.size().to_numpy(),
                                "mean": g.mean().to_numpy(), "sd": g.std(ddof=1).to_numpy(),
                                "q10": g.quantile(QLO).to_numpy(), "q90": g.quantile(QHI).to_numpy()})
            ref = cdf[cdf.k_fold == E3_KF].set_index("k_feat")[f"{proc}_mean"]
            check(np.allclose(cur.set_index("k_feat")["mean"], ref.loc[cur.k_feat]), f"E3 {labels} {proc} means")
            curves[(labels, proc)] = cur
            for _, r in cur.iterrows():
                rows.append({"labels": labels, "procedure": proc, "k_fold": E3_KF, **r.to_dict()})
    # gap segments of panel (a): the archived gap_mean (= leaky mean - honest mean, checked)
    gap = cells[cells.k_fold == E3_KF].set_index("k_feat")["gap_mean"]
    lk, ho = curves[("real", "leaky")].set_index("k_feat"), curves[("real", "honest")].set_index("k_feat")
    check(np.allclose(gap.loc[lk.index], lk["mean"] - ho["mean"]), "E3 gap_mean = leaky - honest")
    for kf, g in gap.items():
        rows.append({"labels": "real", "procedure": "gap (leaky - honest)", "k_fold": E3_KF, "k_feat": kf,
                     "n": int(lk.loc[kf, "n"]), "mean": float(g)})
    rows.append({"labels": "real", "procedure": "S004 public LB", "k_fold": E3_KF, "k_feat": E3_K_LB, "n": 1,
                 "mean": public})
    t = pd.DataFrame(rows)
    t["k_feat"] = t["k_feat"].astype(int)
    t["n"] = t["n"].astype(int)
    return {"curves": curves, "public": public, "cells": cells, "gap": gap}, t


def fig_e3(d: dict, out: Path) -> None:
    plt = ps.setup()
    fig = plt.figure(figsize=(ps.FULL_W, 1.85))
    Y0, H = 0.36, 1.25
    ax = _place(fig, 0.44, Y0, 3.66, H)
    bx = _place(fig, 4.56, Y0, 1.58, H)
    red, grey = ps.color("leaky"), ps.color("honest")
    for axx, labels in ((ax, "real"), (bx, "permuted")):
        lk, ho = d["curves"][(labels, "leaky")], d["curves"][(labels, "honest")]
        k = lk["k_feat"].to_numpy()
        for cur, col, a, zo in ((ho, grey, 0.15, 3), (lk, red, 0.14, 4)):
            axx.fill_between(k, cur["q10"], cur["q90"], color=col, alpha=a, lw=0, zorder=1)
            axx.plot(k, cur["mean"], color=col, lw=1.2, marker="o", ms=3.0, mec="white", mew=0.5, zorder=zo)
        axx.set_xscale("log")
        axx.minorticks_off()
        axx.set_xlim(1.8, 165)
        ps.clean(axx, grid="y")
    ticks = [2, 5, 7, 10, 20, 50, 150]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("k_feat (features kept, log scale)")
    lk, ho = d["curves"][("real", "leaky")].set_index("k_feat"), d["curves"][("real", "honest")].set_index("k_feat")
    gap = d["gap"]
    for kf in lk.index:                                   # the gap as a vertical segment at every k_feat
        ax.plot([kf, kf], [ho.loc[kf, "mean"], lk.loc[kf, "mean"]], color=ps.INK2, lw=0.55, zorder=2)
    for kf, dy in ((7, 0.016), (50, 0.014)):              # labelled above the leaky point (k = 20: in the callout)
        ax.text(kf, lk.loc[kf, "mean"] + dy, f"gap {gap.loc[kf]:.3f}", fontsize=7.8, color=ps.INK2,
                ha="center", va="bottom")
    k0 = E3_K_LB
    ax.plot([k0], [d["public"]], "D", ms=5.6, mfc="none", mec=ps.INK, mew=0.8, zorder=6)
    _lead(ax, f"k_feat = 20 (S004): gap {gap.loc[k0]:.3f}\nleaky {lk.loc[k0, 'mean']:.3f}, honest "
              f"{ho.loc[k0, 'mean']:.3f}\nopen diamond: public LB {d['public']:.5f}",
          xy=(k0 * 1.035, d["public"] - 0.005), xytext=(25, 0.703), ha="left", relpos=(0.0, 0.85), size=7.8)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hdl = [(Patch(facecolor=col, alpha=a, lw=0), Line2D([], [], color=col, lw=1.2, marker="o", ms=3.0, mec="white",
                                                         mew=0.5)) for col, a in ((red, 0.14), (grey, 0.15))]
    ax.legend(hdl, ["leaky: filter on all 250 rows", "honest: filter inside each fold"], loc="upper left",
              fontsize=7.8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)
    ax.set_ylim(0.655, 0.935)
    ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9])
    ax.set_ylabel("5-fold CV AUC")
    ps.panel_title(ax, "Real labels: mean and 10–90% over 20 split seeds", letter="a")
    # (b) permutation null
    ps.hairline(bx, 0.5, "y", color=ps.INK2, lw=0.7, zorder=0)
    bx.text(160, 0.487, "chance", fontsize=7.5, color=ps.INK2, ha="right", va="top")
    tb = [2, 5, 10, 20, 50, 150]
    bx.set_xticks(tb)
    bx.set_xticklabels([str(t) for t in tb])
    bx.set_xlabel("k_feat (log scale)")
    bx.set_ylim(0.38, 0.78)
    bx.set_yticks([0.4, 0.5, 0.6, 0.7])
    bx.set_ylabel("CV AUC")
    ps.panel_title(bx, "Permuted labels", letter="b")
    ps.save(fig, out)
    plt.close(fig)


# =========================================================================================== E4

E4_KS = (5, 10, 13, 20)
E4_COL = {5: ps.BLUE_RAMP[1], 10: ps.BLUE_RAMP[3], 13: ps.BLUE_RAMP[5], 20: "#0a2852"}   # light -> dark by k
E4_LW = {5: 0.95, 10: 1.1, 13: 1.25, 20: 1.4}
E4_NULL_K = 20       # the permuted-label envelope is drawn for this k only (thinnest margin, E4 Conclusion)


def e4_data(inp: Inputs) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    from scipy.stats import rankdata
    d = run_dir(E4_RUN)
    fr = inp.csv("e4", d / "frequency.csv")
    drop = inp.csv("e4", d / "drop.csv").set_index("k")
    ns = inp.csv("e4", d / "null_sorted_curves.csv")
    cfg = inp.path("e4", d / "config.yaml").read_text(encoding="utf-8")
    check("null_quantile: 0.95" in cfg, "E4: null_quantile 0.95 in the run config")
    ft = inp.csv("e4", paths.tables_dir(SOURCE) / "r320_feature_table.csv")
    block = sorted(int(f) for f in ft.loc[ft["in_block"].astype(bool), "feature"])
    check(len(block) == 13, f"E4: {len(block)} block features (expected 13)")
    check(len(fr) == 150, "E4: 150 features")
    pos = [f"pos{i}" for i in range(1, 151)]
    res, rows, summ = {}, [], []
    for k in E4_KS:
        f = fr[["feature", f"freq_k{k}"]].rename(columns={f"freq_k{k}": "freq"})
        f = f.sort_values(["freq", "feature"], ascending=[False, True]).reset_index(drop=True)
        f["rank"] = np.arange(1, len(f) + 1)
        f["avg_rank"] = rankdata(-f["freq"].to_numpy(), method="average")
        f["in_block"] = f["feature"].isin(block)
        check(abs(f["freq"].sum() - k) < 1e-6, f"E4 k={k}: frequencies sum to k")
        m = int(drop.loc[k, "m_star"])
        check(m == int(drop.loc[k, "m_perm"]), f"E4 k={k}: m* is the perm rule")
        # rank-wise 95 % quantile of the 50 permuted-label sorted curves (the perm rule's envelope, run.py
        # drop_rules); m_perm re-derived from it must equal the archived value
        nul = ns[ns["k"] == k].sort_values("permutation")[pos].to_numpy(dtype=float)
        check(nul.shape == (50, 150), f"E4 k={k}: 50 null curves")
        q95 = np.quantile(nul, 0.95, axis=0)
        above = f["freq"].to_numpy() > q95
        check((150 if above.all() else int(np.argmin(above))) == m, f"E4 k={k}: m_perm from the null envelope")
        chance = float(drop.loc[k, "chance_rate"])
        cross = int(np.argmax(f["freq"].to_numpy() < chance)) + 1     # first rank below chance k/150
        res[k] = {"f": f, "m": m, "chance": chance, "q95": q95, "cross": cross}
        for _, r in f.iterrows():
            rows.append({"k": k, "rank": int(r["rank"]), "feature": int(r["feature"]), "freq": r["freq"],
                         "avg_rank_ties": r["avg_rank"], "in_block": bool(r["in_block"]),
                         "above_mstar_perm": int(r["rank"]) <= m, "null_q95_at_rank": q95[int(r["rank"]) - 1]})
        bl = f[f.in_block]
        summ.append({"k": k, "chance_k_over_150": chance, "m_star_perm": m,
                     "freq_at_m_star": float(f.loc[m - 1, "freq"]), "freq_after_m_star": float(f.loc[m, "freq"]),
                     "first_rank_below_chance": cross,
                     "block_features_above_m_star": int((bl["rank"] <= m).sum()),
                     "block_features_below_chance": int((bl["freq"] < chance).sum()),
                     "block_positions (feature:avg rank)": " ".join(f"{int(a)}:{b:g}" for a, b in
                                                                    zip(bl["feature"], bl["avg_rank"]))})
    return {"res": res, "block": block}, pd.DataFrame(rows), pd.DataFrame(summ)


def fig_e4(d: dict, out: Path) -> None:
    plt = ps.setup()
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(ps.FULL_W, 2.52))
    ax = _place(fig, 0.5, 1.2, 5.45, 1.12)
    sx = _place(fig, 0.5, 0.36, 5.45, 0.6)
    res = d["res"]
    XL = (0.92, 165)
    sq = (lambda v: np.sqrt(np.clip(v, 0, None)), np.square)

    # (a) sorted curves (steps: rank is discrete), m* dots, k = 20 null envelope, chance segments
    rn = res[E4_NULL_K]
    ranks = np.arange(1, 151)
    ax.fill_between(ranks, 0, rn["q95"], step="mid", color=ps.MID, lw=0, zorder=0)
    ax.plot(ranks, rn["q95"], drawstyle="steps-mid", color=ps.AXIS, lw=0.6, zorder=1)
    ax.text(1.0, 0.012, f"permuted labels, 95% envelope (k = {E4_NULL_K})", fontsize=7.5, color=ps.INK2,
            ha="left", va="bottom")
    for k in E4_KS:
        r, f = res[k], res[k]["f"]
        ax.plot([60, 150], [r["chance"]] * 2, color=E4_COL[k], lw=0.7, zorder=2, solid_capstyle="butt")
        ax.plot(f["rank"], f["freq"], color=E4_COL[k], lw=E4_LW[k], drawstyle="steps-mid", zorder=3)
    for k in E4_KS:
        r, f = res[k], res[k]["f"]
        ax.plot([r["m"]], [f.loc[r["m"] - 1, "freq"]], "o", ms=4.4, mfc=E4_COL[k], mec="white", mew=0.7, zorder=5)
    # chance values, staggered to the right of the segments
    lab_y = {5: 0.026, 10: 0.058, 13: 0.098, 20: 0.15}
    for k in E4_KS:
        ax.annotate(f"{res[k]['chance']:.3f}", xy=(150, res[k]["chance"]), xytext=(172, lab_y[k]),
                    fontsize=7.2, color=ps.INK2, ha="left", va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=ps.AXIS, lw=0.5, shrinkA=0.5, shrinkB=0.5))
    ax.text(172, 0.215, "chance\nk/150", fontsize=7.2, color=ps.MUTED, ha="left", va="bottom", linespacing=1.0)
    # legend (one row, light -> dark) plus direct labels where a curve has free space beside it: k = 5 under
    # its plateau, k = 10 under its tail, k = 20 above its tail (k = 13 lies between 10 and 20)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=E4_COL[k], lw=E4_LW[k] + 0.2) for k in E4_KS], [f"k = {k}" for k in E4_KS],
              loc="upper right", ncol=4, fontsize=7.8, handlelength=1.6, columnspacing=1.2, borderaxespad=0.15)
    ax.text(4.4, 0.33, "k = 5", fontsize=7.5, color=ps.INK2, ha="left", va="center")
    f10 = res[10]["f"].set_index("rank")["freq"]
    ax.text(10.2, (np.sqrt(f10.loc[10:15].min()) - 0.085) ** 2, "k = 10", fontsize=7.5, color=ps.INK2,
            ha="left", va="center")
    f20 = res[20]["f"].set_index("rank")["freq"]
    ax.text(10.2, (np.sqrt(f20.loc[10:15].max()) + 0.085) ** 2, "k = 20", fontsize=7.5, color=ps.INK2,
            ha="left", va="center")
    ax.set_xscale("log")
    ax.set_yscale("function", functions=sq)
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0, 0.03, 0.1, 0.25, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.03", "0.1", "0.25", "0.5", "1"])
    ax.set_ylabel("frequency (√ scale)")
    ps.clean(ax, grid="y")
    ps.panel_title(ax, "Sorted top-k selection frequency, 2,000 bootstrap resamples; dot: m* (perm rule)",
                   letter="a")

    # (b) strip: where the 13 label-free block features sit in each sorted order (ties stacked)
    nk = len(E4_KS)
    for i, k in enumerate(E4_KS):
        r, f = res[k], res[k]["f"]
        y = nk - 1 - i
        sx.add_patch(Rectangle((XL[0], y - 0.42), r["m"] + 0.5 - XL[0], 0.84, facecolor=ps.BLUE_RAMP[0],
                               edgecolor="none", zorder=0))
        sx.plot([r["m"] + 0.5, XL[1]], [y, y], color=ps.GRID, lw=0.6, zorder=0)
        sx.plot([r["cross"] - 0.5] * 2, [y - 0.42, y + 0.42], color=ps.MUTED, lw=0.6, zorder=1)
        bl = f[f.in_block]
        for xr, grp in bl.groupby("avg_rank"):
            n = len(grp)
            yy = y + (np.arange(n) - (n - 1) / 2) * 0.27
            sx.scatter(np.full(n, xr), yy, s=6.5, color=ps.INK2, lw=0, zorder=3)
        if k == 5:
            z = bl[bl.freq == 0]
            sx.text(z["avg_rank"].iloc[0] / 1.07, y, f"{len(z)} at frequency 0", fontsize=7.2, color=ps.INK2,
                    ha="right", va="center")
    fk, y10 = res[10]["f"].set_index("feature"), nk - 1 - E4_KS.index(10)
    for feat in (64, 30, 15):
        sx.text(fk.loc[feat, "avg_rank"] * 1.05, y10, str(feat), fontsize=7.2, color=ps.INK2, ha="left",
                va="center")
    sx.text((res[10]["cross"] - 0.5) * 1.04, y10, "chance", fontsize=7.2, color=ps.MUTED, ha="left",
            va="center")
    sx.set_yticks(range(nk))
    sx.set_yticklabels([f"k = {k}" for k in reversed(E4_KS)], fontsize=7.8)
    sx.set_ylim(-0.55, nk - 0.45)
    ps.clean(sx, grid="", left=False)
    sx.tick_params(axis="y", length=0)
    ticks = [1, 2, 5, 7, 10, 20, 50, 100, 150]
    for a in (ax, sx):
        a.set_xscale("log")
        a.set_xlim(*XL)
        a.set_xticks(ticks)
        a.minorticks_off()
    sx.set_xticklabels([str(t) for t in ticks])
    ax.set_xticklabels([])
    sx.set_xlabel("feature rank by selection frequency (log scale)")
    ps.panel_title(sx, "Position of the 13 label-free block features (stacked: ties); shaded: ranks 1 to m*",
                   letter="b")
    ps.save(fig, out)
    plt.close(fig)


# =========================================================================================== main

def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compute and print only; write nothing")
    ap.add_argument("--only", default=",".join(FIGS), help="comma-separated subset of e1,e2,e3,e4")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [s for s in only if s not in FIGS]
    if bad:
        C.die(f"--only: unknown {bad}; choose from {FIGS}", 2)
    C.print_archive(SOURCE)
    tdir, fdir = paths.tables_dir(SOURCE), paths.figures_dir(SOURCE)
    inp = Inputs()
    from tabulate import tabulate
    jobs = {}

    if "e1" in only:
        panels, t1 = e1_data(inp)
        md1 = e1_md(t1)
        print("\nE1 (ext_e1_ridges):")
        print(tabulate(md1.to_numpy().tolist(), headers=list(md1.columns), tablefmt="simple", disable_numparse=True))
        jobs["e1"] = (lambda: fig_e1(panels, fdir / "ext_e1.png"),
                      {"ext_e1_ridges.csv": t1, "ext_e1_ridges.md": md1})
    if "e2" in only:
        d2, t2 = e2_data(inp)
        md2 = fmt(t2.drop(columns=["run_id"]))
        print("\nE2 (ext_e2_fan):")
        print(tabulate(md2.to_numpy().tolist(), headers=list(md2.columns), tablefmt="simple", disable_numparse=True))
        jobs["e2"] = (lambda: fig_e2(d2, fdir / "ext_e2.png"), {"ext_e2_fan.csv": t2, "ext_e2_fan.md": md2})
    if "e3" in only:
        d3, t3 = e3_data(inp)
        md3 = fmt(t3)
        print(f"\nE3 (ext_e3_curves; S004 public {d3['public']}):")
        print(tabulate(md3.to_numpy().tolist(), headers=list(md3.columns), tablefmt="simple", disable_numparse=True))
        jobs["e3"] = (lambda: fig_e3(d3, fdir / "ext_e3.png"), {"ext_e3_curves.csv": t3, "ext_e3_curves.md": md3})
    if "e4" in only:
        d4, t4, s4 = e4_data(inp)
        md4 = fmt(s4)
        print("\nE4 (ext_e4_curves; summary, full curves in the CSV):")
        print(tabulate(md4.to_numpy().tolist(), headers=list(md4.columns), tablefmt="simple", disable_numparse=True))
        jobs["e4"] = (lambda: fig_e4(d4, fdir / "ext_e4.png"), {"ext_e4_curves.csv": t4, "ext_e4_curves.md": md4})
    if args.check:
        print("\n--check: nothing written")
        return 0

    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    notes = {
        "ext_e1_ridges.md": "One row per panel and scheme of reports/figures/ext_e1.png: 50 split seeds each; "
                            "MDD = 2 sd (archived paired_difference.csv) with its 95 % CI; 'resolves 0.01' = whole CI "
                            "below 0.01 (bold in the figure); band = +-0.01 around the pooled 5 x 20 mean.",
        "ext_e2_fan.md": "Rows 'subsets': CV AUC of the 10 subsets per effective size x (mean, 10-90 %, range). "
                         "Rows 'at 500 rows': power-law point, nominal 90 % bootstrap interval, envelope, replica "
                         "truth (R941) and coverage / RMSE / bias of the procedure on 100 replicas (R942).",
        "ext_e3_curves.md": "5-fold CV AUC per k_feat over 20 split seeds (real labels) or 30 permutations x 5 "
                            "seeds (permuted labels): mean, sd, 10-90 %. Rows 'gap': archived gap_mean (cells.csv), "
                            "drawn as the vertical segments; row 'S004 public LB': the open diamond.",
        "ext_e4_curves.md": "Summary per k; the CSV has every sorted position (feature, frequency, average rank "
                            "over ties, block membership from r320_feature_table.csv, rank-wise 95 % quantile of the "
                            "50 permuted-label curves = the perm-rule envelope; drawn for k = 20).",
    }
    written = []
    for key in only:
        draw, tables = jobs[key]
        for name, df in tables.items():
            p = tdir / name
            if name.endswith(".csv"):
                df.to_csv(p, index=False)
            else:
                write_md(df, p, "the archived files listed in reports/tables/ext_manifest.json", notes.get(name, ""))
            written.append(p)
        draw()
        written.append(fdir / f"ext_{key}.png")

    mpath = tdir / "ext_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    manifest.update({"generated_by": "scripts/make_ext_figures.py", "generated_at": now_iso()})
    figs = manifest.setdefault("figures", {})
    srcs = {"e1": [E1_RUN, E1Q_RUN], "e2": [E2_RUN, E2Q_RUN, E2G_RUN, R941_RUN, R942_RUN], "e3": [E3_RUN],
            "e4": [E4_RUN, R320_RUN]}
    outs = {"e1": ["ext_e1_ridges.csv", "ext_e1_ridges.md"], "e2": ["ext_e2_fan.csv", "ext_e2_fan.md"],
            "e3": ["ext_e3_curves.csv", "ext_e3_curves.md"], "e4": ["ext_e4_curves.csv", "ext_e4_curves.md"]}
    for key in only:
        figs[f"ext_{key}.png"] = {
            "source_runs": srcs[key],
            "inputs": {C.rel(p).replace("\\", "/"): sha256(p) for p in inp.used.get(key, [])},
            "table_views": [f"reports/tables/{n}" for n in outs[key]],
        }
    manifest["conventions"] = {
        "bands": "10-90 % of the plotted replicates (np.quantile, linear interpolation) unless a caption says "
                 "nominal 90 % (bootstrap 5-95 %)",
        "e1": "densities: Gaussian KDE (Scott bandwidth) evaluated on [min, max] of the 50 estimates only (cut = 0), "
              "each row scaled to the same height; fill lightness log-scaled by fits per decision (2 -> 200) within "
              "the entity hue (linear grey, QDA orange); ticks: the 50 estimates; bar: mean +- MDD, MDD = 2 sd "
              "(ddof 1) as archived, bold where mdd_hi < 0.01; band: pooled 5 x 20 mean +- 0.01",
        "e2": "x = effective training rows 4n/5; dashed: archived power-law parameters evaluated, fan = 5-95 % of "
              "the 2,000 archived bootstrap curves (outlined, to keep it apart from the filled 10-90 % data band); "
              "side panel: the 500-row box of (a) enlarged on its own y scale (0.845-0.99); panel (b) on "
              "-log10(1 - AUC), AUC 1.0 drawn at 1 - 1e-5 with an axis break between 0.9999 and 1.0",
        "e3": "official 5-fold splits (k_fold = 5); gap = vertical segment between the two means at each k_feat",
        "e4": "sorted order with ties broken by feature id for the curves (drawn as steps, square-root y scale); "
              "strip dots at the average rank of their tie group, tied block features stacked vertically; m* = perm "
              "rule (pre-registered primary), re-derived from the rank-wise 95 % null quantile as a check",
        "jitter": "subset dots of E2 jittered along x for visibility only (fixed seed 11)",
    }
    manifest["outputs"] = sorted({*manifest.get("outputs", []), *(C.rel(p).replace("\\", "/") for p in written)})
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwritten:")
    for p in written + [mpath]:
        print(f"  {C.rel(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
