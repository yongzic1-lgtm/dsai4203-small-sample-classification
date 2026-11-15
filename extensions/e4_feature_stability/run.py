"""E4 - stability of the |corr| feature ranking, and whether stable features are useful.

Question: rank the 150 features by |corr(feature, target)| on many bootstrap
resamples of the 250 rows. How often does each feature enter the top k? Is there
a sharp drop in the sorted frequencies, does its position depend on k, and are
the features above the drop the ones worth keeping?

Steps:
  1. bootstrap   B resamples (250 rows with replacement); per resample |corr| ranking,
                 no model. Selection frequency of every feature for every k
                 (ssc.features.bootstrap_topk_frequency; same resamples for all k).
  2. drop        objective rules for the drop position m*(k): largest gap between
                 consecutive sorted frequencies, frequency >= threshold, binomial
                 chance bound, and a permuted-label envelope of the sorted curve.
  3. overlap     stable sets compared across k (+ precision/recall vs truth.csv on
                 synthetic data; the code runs without truth.csv on real data).
  4. usefulness  repeated CV on IDENTICAL splits, paired against the model on all 150:
                 naive  = set chosen on all 250 rows, then FixedColumns (biased, cf. E3):
                          naive_stable (freq >= threshold), naive_drop (the set above the
                          drop m*(k) of the primary rule), naive_corr_top_m
                 honest = selection redone inside each fold: honest_stable (StabilityTopK),
                          honest_corr_top_m (CorrelationTopK, same m as the stable set),
                          honest_drop_top_m (CorrelationTopK(m*(k)); m*(k) comes from all rows,
                          a size-only leak, as for honest_corr_top_m)

Report outputs go to ssc.paths.tables_dir / figures_dir; with --quick to
<run>/report_preview/ instead (a smoke test never overwrites the report files).

Run from the project root (--data defaults to $SSC_DATA, else raw):
    .venv\\Scripts\\python.exe extensions\\e4_feature_stability\\run.py --data synthetic --quick
    .venv\\Scripts\\python.exe extensions\\e4_feature_stability\\run.py          # real data, full settings
    .venv\\Scripts\\python.exe extensions\\e4_feature_stability\\run.py --replot 20261022-071414_e4_feature_stability

--replot RUN_ID only redraws the report figures of an archived run from the tables in its run folder
(frequency.csv, null_sorted_curves.csv, drop.csv, usefulness.csv): no bootstrap, permutation or CV is
rerun, and no run folder, registry row or report table is written. The prefix (e4_) comes from the
reports/tables/e4*_manifest.json that names the run, so the figures keep their file names;
--replot-out DIR writes them elsewhere (e.g. a preview folder). e4_frequency_sorted.png (report
Figure 1) is drawn at its printed size, the full A4 text width, with a log rank axis.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from scipy.stats import binom  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.data import FEATURES, load  # noqa: E402
from ssc.features import StabilityTopK, abs_correlation, bootstrap_topk_frequency, top_k_indices  # noqa: E402
from ssc.models import build_pipeline  # noqa: E402
from ssc.tracking import Run, now_iso  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits, paired_compare, split_signature  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
PREFIX = "e4_"
P = len(FEATURES)
RULES = ("gap", "half", "binom", "perm")
VARIANTS = ("all150", "naive_stable", "naive_drop", "naive_corr_top_m",
            "honest_stable", "honest_drop_top_m", "honest_corr_top_m")

# Report palette (categorical slots in fixed order) and chart ink.
# ext34 fix: 4th slot (yellow) added and indexing by len(SERIES): with 4 k_fold values (E3) or
# 4 k values (E4) the old `i % 3` gave the 1st and 4th line the same color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Sorted-frequency figure (report Figure 1) is drawn at its printed size, the full text width of an
# A4 page with 2.54 cm margins, so the font sizes below are printed sizes in pt. (The earlier 1 x 4
# strip was 16.6 in wide and printed its text at 2-3 pt; research/reports/fix_round1.md.) One row of
# four panels: a 2 x 2 grid with the same fonts made the E4 draft 2.01-2.03 pages with equal-width
# tables; the strip keeps it at 1.86-1.94 pages as before (1.84 with scripts/build_report.py; 2026-11-01).
SORTED_WIDTH = 6.2        # in
SORTED_NCOLS = 4          # panels per row
SORTED_PANEL_H = 0.85     # in, plot height of one panel
FS = {"title": 7, "label": 7, "tick": 6.5, "legend": 6.3}
REPORT_DPI = {"frequency_sorted": 300, "frequency_overlay": 150, "usefulness": 150}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, val in (over or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: Path, quick: bool) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    quick_block = cfg.pop("quick", None)
    if quick:
        cfg = deep_merge(cfg, quick_block)
    if (cfg.get("pipeline") or {}).get("selector"):
        raise ValueError("config `pipeline` must not contain a selector: run.py adds it")
    ks = sorted({int(k) for k in cfg["bootstrap"]["ks"]})
    if any(k < 1 or k >= P for k in ks):
        raise ValueError(f"bootstrap.ks must be in 1..{P - 1}")
    cfg["bootstrap"]["ks"] = ks
    u = cfg["usefulness"]
    u["ks"] = ks if u.get("ks") is None else sorted({int(k) for k in u["ks"]})
    if cfg["drop"].get("null_n_boot") is None:
        cfg["drop"]["null_n_boot"] = int(cfg["bootstrap"]["n_boot"])
    if cfg["drop"]["primary_rule"] not in RULES:
        raise ValueError(f"drop.primary_rule must be one of {RULES}")
    cfg["quick_mode"] = bool(quick)
    try:
        cfg["config_file"] = path.resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        cfg["config_file"] = str(path)
    return cfg


def check_official_cv(cv: dict) -> None:
    try:
        off = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv") or {}
    except OSError:
        return
    mine = Scheme.from_dict(cv).to_dict()
    ref = Scheme.from_dict(off).to_dict() if off else mine
    diff = {k: (mine[k], ref[k]) for k in ("kind", "n_splits", "n_repeats", "seed") if mine[k] != ref[k]}
    if diff:
        print(f"note: cv differs from project.yaml official_cv (mine, official): {diff}")


def load_truth(source: str) -> set[int] | None:
    """Indices of the truly informative features (synthetic data only; None on real data)."""
    p = paths.data_dir(source) / "truth.csv"
    if not p.exists():
        return None
    return {FEATURES.index(n) for n in pd.read_csv(p)["feature"].astype(str)}


# ---------------------------------------------------------------------------
# bootstrap ranking
# ---------------------------------------------------------------------------
def boot_abs_corr(X: np.ndarray, y: np.ndarray, n_boot: int, seed: int, method: str) -> np.ndarray:
    """(n_boot, n_features) |corr| per resample. Same RNG convention as
    ssc.features.bootstrap_topk_frequency, so the k-frequencies agree exactly."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = abs_correlation(X[idx], y[idx], method)
    return out


def ranks_desc(A: np.ndarray) -> np.ndarray:
    """Rank 1 = largest |corr| in each row; ties broken by lower column index (as top_k_indices)."""
    order = np.argsort(-A, axis=1, kind="stable")
    R = np.empty(A.shape, dtype=int)
    np.put_along_axis(R, order, np.broadcast_to(np.arange(1, A.shape[1] + 1), A.shape), axis=1)
    return R


def sort_desc(freq: np.ndarray) -> np.ndarray:
    """Feature indices by descending frequency, ties by lower index."""
    return np.lexsort((np.arange(len(freq)), -freq))


def drop_rules(freq: np.ndarray, k: int, n_boot: int, null_sorted: np.ndarray, dcfg: dict) -> tuple[dict, dict]:
    """m*(k) under each rule, and the feature set above the drop for each rule."""
    order = sort_desc(freq)
    f = freq[order]
    steps = f[:-1] - f[1:]
    i_gap = int(np.argmax(steps))
    bound = float(binom.ppf(1 - float(dcfg["binom_alpha"]) / P, n_boot, k / P) / n_boot)
    q_null = np.quantile(null_sorted, float(dcfg["null_quantile"]), axis=0)
    above = f > q_null
    m_perm = P if above.all() else int(np.argmin(above))  # leading run above the null envelope
    sets = {
        "gap": order[: i_gap + 1],
        "half": np.flatnonzero(freq >= float(dcfg["threshold"])),
        "binom": np.flatnonzero(freq > bound),
        "perm": order[:m_perm],
    }
    row = {
        "k": k, "chance_rate": k / P,
        "m_gap": i_gap + 1, "gap_size": float(steps[i_gap]),
        "freq_above_gap": float(f[i_gap]), "freq_below_gap": float(f[i_gap + 1]),
        "m_half": len(sets["half"]), "binom_bound": bound, "m_binom": len(sets["binom"]),
        "m_perm": m_perm, "null_q_top1": float(q_null[0]),
        "top_freq": float(f[0]), "freq_at_rank_k": float(f[k - 1]),
    }
    row["m_star"] = row[f"m_{dcfg['primary_rule']}"]
    return row, {r: np.sort(s) for r, s in sets.items()}


def precision_recall(sel, truth: set[int]) -> tuple[float, float]:
    s = set(int(i) for i in sel)
    tp = len(s & truth)
    return (tp / len(s) if s else float("nan")), tp / len(truth)


# ---------------------------------------------------------------------------
# usefulness test
# ---------------------------------------------------------------------------
def with_selector(base: dict, selector: dict | None) -> dict:
    spec = copy.deepcopy(base)
    if selector:
        spec["selector"] = selector
    return spec


def stable_full(freq: np.ndarray, threshold: float) -> np.ndarray:
    """Stable set on all rows; mirrors StabilityTopK (never empty)."""
    s = np.flatnonzero(freq >= threshold)
    return s if len(s) else np.array([int(np.argmax(freq))])


def _fold_selection(X, y, split, k, n_boot, thr, method, seed):
    sel = StabilityTopK(k=k, n_boot=n_boot, threshold=thr, method=method, random_state=seed)
    sel.fit(X[split.train_idx], y[split.train_idx])
    return split.repeat, split.fold, sel.get_support(indices=True)


def jaccard(a, b) -> float:
    a, b = set(int(i) for i in a), set(int(i) for i in b)
    return len(a & b) / len(a | b) if a | b else 1.0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)


def legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=8, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)


def fig_sorted(freqs: dict, null_sorted: dict, drops: pd.DataFrame, truth: set[int] | None, dcfg: dict, B: int):
    """Report figure: one panel per k, drawn at its printed size (font sizes = printed pt sizes).

    The rank axis is logarithmic, so the drop (ranks 5-20) is not squeezed into the left edge. Panel
    titles give m* of the primary rule; with at most two panels per row also the other rules' m*.
    """
    ks = sorted(freqs)
    ncol = min(len(ks), SORTED_NCOLS)
    nrow = -(-len(ks) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(SORTED_WIDTH, nrow * (SORTED_PANEL_H + 0.2) + 0.5),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    x = np.arange(1, P + 1)
    q = float(dcfg["null_quantile"])
    rule = dcfg["primary_rule"]
    others = [r for r in RULES if r != rule]
    handles = {}
    for ax, k in zip(axes.flat, ks):
        order = sort_desc(freqs[k])
        f = freqs[k][order]
        lo, hi = np.quantile(null_sorted[k], [1 - q, q], axis=0)
        band = ax.fill_between(x, lo, hi, color=MUTED, alpha=0.22, lw=0)
        med = ax.plot(x, np.median(null_sorted[k], axis=0), color=MUTED, lw=0.8)[0]
        handles[f"permuted labels: median, {round((1 - q) * 100)}–{round(q * 100)}%"] = (band, med)
        handles["chance k/150"] = ax.axhline(k / P, color=INK2, lw=0.8, ls=(0, (4, 2)))
        d = drops[drops["k"] == k].iloc[0]
        handles["binomial bound"] = ax.axhline(d["binom_bound"], color=INK2, lw=0.8, ls=(0, (1, 1.5)))
        m = int(d["m_star"])
        handles[f"m* ({rule} rule)"] = ax.axvline(m + 0.5, color=SERIES[2], lw=1.2)
        handles["observed (sorted)"] = ax.plot(x, f, color=SERIES[0], lw=1.5)[0]
        if truth is not None:
            is_t = np.array([i in truth for i in order])
            handles["true informative (synthetic)"] = ax.plot(x[is_t], f[is_t], "o", color=SERIES[1], ms=3,
                                                              mec="white", mew=0.5)[0]
        ax.set_title(f"top-{k}", loc="left", color=INK, fontsize=FS["title"], pad=2)
        ax.set_title(f"m* = {m} ({rule})" + ("; " + ", ".join(f"{r} {int(d[f'm_{r}'])}" for r in others)
                                            if ncol <= 2 else ""),
                     loc="right", color=INK2, fontsize=FS["tick"], pad=2)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=INK2, labelsize=FS["tick"], width=0.6, length=2.5, pad=1.5)
    ax0 = axes[0][0]
    ax0.set_xscale("log")
    ax0.set_xlim(0.85, P * 1.15)
    ax0.set_xticks([1, 2, 5, 10, 20, 50, P])
    ax0.set_xticklabels(["1", "2", "5", "10", "20", "50", str(P)])
    ax0.xaxis.set_minor_locator(NullLocator())
    ax0.set_ylim(-0.03, 1.05)
    ax0.set_yticks([0, 0.5, 1])
    for ax in axes.flat[len(ks):]:
        ax.set_visible(False)
    fig.supylabel("selection frequency", color=INK2, fontsize=FS["label"])
    fig.supxlabel(f"feature rank by selection frequency (log scale); B = {B} resamples",
                  color=INK2, fontsize=FS["label"])
    order = ["observed (sorted)", "true informative (synthetic)"]
    labels = sorted(handles, key=lambda s: order.index(s) if s in order else len(order))
    leg = fig.legend([handles[s] for s in labels], labels, loc="outside upper center",
                     ncol=len(labels) if len(labels) <= 5 else -(-len(labels) // 2),
                     fontsize=FS["legend"], handlelength=1.8, handletextpad=0.5, columnspacing=1.4,
                     frameon=False, borderaxespad=0.1)
    for t in leg.get_texts():
        t.set_color(INK2)
    # Lay out once, then freeze: constrained layout reruns on every draw and depends on earlier draws
    # (the run folder copy is saved first at another dpi), so a --replot would differ by sub-pixel shifts.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig


def fig_overlay(freqs: dict, B: int):
    ks = sorted(freqs)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(1, P + 1)
    for i, k in enumerate(ks):
        f = np.sort(freqs[k])[::-1]
        ax.plot(x, f, color=SERIES[i % len(SERIES)], lw=2, label=f"top-{k}")
        ax.axhline(k / P, color=SERIES[i % len(SERIES)], lw=1, ls="--")
    ax.set_xscale("log")
    ticks = [1, 2, 5, 10, 20, 50, 100, 150]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    style(ax, f"Sorted selection frequency for each k (dashed: chance k/150; B = {B})",
          "rank of the feature by frequency (log scale)", "selection frequency")
    legend(ax, loc="upper right")
    fig.tight_layout()
    return fig


def fig_usefulness(summ: pd.DataFrame, n_rep: int):
    ks = sorted(summ["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.1), sharey=True, squeeze=False)
    colors = {"all150": MUTED, "naive_stable": SERIES[1], "naive_drop": SERIES[1], "naive_corr_top_m": SERIES[1],
              "honest_stable": SERIES[0], "honest_drop_top_m": SERIES[0], "honest_corr_top_m": SERIES[0]}
    marker = {"all150": "s", "naive_stable": "o", "naive_drop": "^", "naive_corr_top_m": "D",
              "honest_stable": "o", "honest_drop_top_m": "^", "honest_corr_top_m": "D"}
    for ax, k in zip(axes[0], ks):
        s = summ[summ["k"] == k].set_index("variant").loc[list(VARIANTS)]
        xs = np.arange(len(VARIANTS))
        for xi, (v, r) in zip(xs, s.iterrows()):
            ax.errorbar(xi, r["auc_mean"], yerr=r["auc_std_repeat"], color=colors[v], marker=marker[v],
                        ms=7, lw=2, capsize=3)
        ax.axhline(s.loc["all150", "auc_mean"], color=MUTED, lw=1)
        labels = [f"{v.replace('_', ' ')}\n({_nf(r['n_features_mean'])} feat.)" for v, r in s.iterrows()]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        style(ax, f"stable set from top-{k} frequencies", "", "CV AUC" if k == ks[0] else "")
    handles = [plt.Line2D([], [], ls="", marker="o", color=SERIES[1], label="naive: set chosen on all 250 rows"),
               plt.Line2D([], [], ls="", marker="o", color=SERIES[0], label="honest: selection inside each fold"),
               plt.Line2D([], [], ls="", marker="s", color=MUTED, label="all 150 features"),
               plt.Line2D([], [], ls="", marker="^", color=INK2, label="triangle = set above the drop m*(k)"),
               plt.Line2D([], [], ls="", marker="D", color=INK2, label="diamond = top-m by |corr| (m of the stable set)")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.suptitle(f"Model on stable features vs all {P} (mean +/- SD over {n_rep} split seeds, identical splits)",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def _nf(n: float) -> str:
    return f"{n:.0f}" if float(n).is_integer() else f"{n:.1f}"


# ---------------------------------------------------------------------------
# report figures: save them, or redraw them from an archived run (--replot)
# ---------------------------------------------------------------------------
def save_report_figures(figs: dict, fdir: Path, prefix: str, run_id: str) -> list[Path]:
    """Write <prefix><name>.png for each figure; the PNG metadata names the source run."""
    fdir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fig in figs.items():
        p = fdir / f"{prefix}{name}.png"
        fig.savefig(p, dpi=REPORT_DPI[name], bbox_inches="tight", pad_inches=0.03,
                    metadata={"Source": f"run {run_id}"})
        plt.close(fig)
        out.append(p)
    return out


def find_run_dir(ref: str, source: str) -> Path:
    """Archived run folder from an exact run_id or a unique prefix of one."""
    exp = paths.experiments_dir(source)
    if (exp / ref).is_dir():
        return exp / ref
    hits = sorted(p for p in exp.glob(f"{ref}*") if p.is_dir()) if exp.exists() else []
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise SystemExit(f"--replot: run id prefix {ref!r} is ambiguous: {', '.join(p.name for p in hits[:6])}")
    raise SystemExit(f"--replot: run {ref!r} not found in {exp}")


def report_prefix(run_id: str, source: str) -> str | None:
    """Prefix of the report files (e4_) whose manifest names this run; None if none does."""
    hits = []
    for m in sorted(paths.tables_dir(source).glob(f"{PREFIX.rstrip('_')}*manifest.json")):
        try:
            rid = json.loads(m.read_text(encoding="utf-8")).get("run_id")
        except (OSError, ValueError):
            continue
        if rid == run_id:
            hits.append(m.name[: -len("manifest.json")])
    if len(hits) > 1:
        raise SystemExit(f"--replot: several manifests name run {run_id}: {hits}")
    return hits[0] if hits else None


def load_archived(rd: Path, cfg: dict) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """freqs, null_sorted, drops and usefulness of an archived run, as main() held them, read from
    frequency.csv, null_sorted_curves.csv, drop.csv and usefulness.csv. Nothing is recomputed."""
    ks = sorted(int(k) for k in cfg["bootstrap"]["ks"])
    ft = pd.read_csv(rd / "frequency.csv")
    col = np.array([FEATURES.index(str(v)) for v in ft["feature"]])  # rows are sorted; back to feature order
    freqs = {}
    for k in ks:
        fk = np.full(P, np.nan)
        fk[col] = ft[f"freq_k{k}"].to_numpy(dtype=float)
        if np.isnan(fk).any():
            raise SystemExit(f"--replot: {rd.name}/frequency.csv does not cover all {P} features")
        freqs[k] = fk
    ns = pd.read_csv(rd / "null_sorted_curves.csv")
    pos = [f"pos{i + 1}" for i in range(P)]
    null_sorted = {k: ns[ns["k"] == k].sort_values("permutation")[pos].to_numpy(dtype=float) for k in ks}
    return freqs, null_sorted, pd.read_csv(rd / "drop.csv"), pd.read_csv(rd / "usefulness.csv")


def replot(ref: str, source: str, out_dir: Path | None) -> int:
    """Redraw the report figures of an archived E4 run from the tables saved in its run folder.

    No bootstrap, permutation or model fit is run; no run folder, registry row or report table is
    written. The figures replace <prefix>frequency_sorted.png, <prefix>frequency_overlay.png and
    <prefix>usefulness.png in reports/figures, where <prefix> comes from the reports/tables/e4*_manifest.json
    that names the run, so file names and the manifest stay valid. out_dir: write there instead.
    """
    rd = find_run_dir(ref, source)
    run_id = rd.name
    need = ["config.yaml", "meta.json", "summary.json", "frequency.csv", "null_sorted_curves.csv", "drop.csv",
            "usefulness.csv"]
    missing = [n for n in need if not (rd / n).exists()]
    if missing:
        raise SystemExit(f"--replot: {run_id} is not a finished E4 run (missing {', '.join(missing)})")
    cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    if meta.get("data_source", source) != source:
        raise SystemExit(f"--replot: {run_id} was made on {meta['data_source']!r} data; pass --data {meta['data_source']}")
    if summary.get("status") != "done":
        raise SystemExit(f"--replot: {run_id} did not finish (status {summary.get('status')!r})")
    freqs, null_sorted, drops, useful = load_archived(rd, cfg)
    B = int(cfg["bootstrap"]["n_boot"])
    figs = {"frequency_sorted": fig_sorted(freqs, null_sorted, drops, load_truth(source), cfg["drop"], B),
            "frequency_overlay": fig_overlay(freqs, B),
            "usefulness": fig_usefulness(useful, Scheme.from_dict(cfg["cv"]).n_repeats)}
    prefix = report_prefix(run_id, source)
    if out_dir is not None:
        fdir, prefix = Path(out_dir), prefix or PREFIX
    elif prefix is None:
        for fig in figs.values():
            plt.close(fig)
        raise SystemExit(f"--replot: no {PREFIX}*manifest.json in {paths.tables_dir(source)} names {run_id}, so its "
                         "figures are not report figures; use --replot-out DIR to draw them elsewhere")
    else:
        fdir = paths.figures_dir(source)
    written = save_report_figures(figs, fdir, prefix, run_id)
    print(f"E4 replot of {run_id} ({source} data) from its archived tables: nothing recomputed, no run written")
    for p in written:
        print(f"  wrote {p}")
    return 0


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def fmt_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
    out = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return out.to_string(index=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=paths.SOURCES, default=None,
                    help="raw (Kaggle, real archive) or synthetic (sandbox) (default: $SSC_DATA or raw)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--quick", action="store_true",
                    help="small settings for a smoke test (report tables/figures go to <run>/report_preview/)")
    ap.add_argument("--author", default=None)
    ap.add_argument("--n-jobs", type=int, default=1, help="joblib workers for CV fits and in-fold selection")
    ap.add_argument("--replot", metavar="RUN_ID", default=None,
                    help="only redraw the report figures of this archived run from its saved tables "
                         "(run_id or a unique prefix; nothing recomputed, no new run, no registry row)")
    ap.add_argument("--replot-out", type=Path, default=None, metavar="DIR",
                    help="with --replot: write the figures to DIR instead of reports/figures")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    source = paths.resolve_source(args.data)
    if args.replot_out is not None and not args.replot:
        ap.error("--replot-out needs --replot")
    if args.replot:
        if args.quick:
            ap.error("--replot redraws an archived run; --quick does not apply")
        return replot(args.replot, source, args.replot_out)
    if not args.config.exists() and not args.config.is_absolute() and (paths.ROOT / args.config).exists():
        args.config = paths.ROOT / args.config  # relative to the project root when not found from the cwd
    cfg = load_config(args.config, args.quick)
    cfg["data_source"] = source
    check_official_cv(cfg["cv"])
    ds = load(source)
    X, y = ds.X.to_numpy(dtype=float), ds.y
    truth = load_truth(source)
    bc, dc, uc = cfg["bootstrap"], cfg["drop"], cfg["usefulness"]
    B, ks, method, bseed = int(bc["n_boot"]), bc["ks"], bc["method"], int(bc["seed"])

    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    with Run(cfg["name"], cfg, kind="extension", author=args.author, source=source,
             hypothesis=cfg.get("hypothesis", "")) as run:
        print(f"E4 feature-ranking stability | data={source} | quick={args.quick} | run_id={run.run_id}")
        sanity: dict[str, bool] = {}

        # ---- 1. bootstrap frequencies -------------------------------------
        t0 = time.perf_counter()
        corr_full = abs_correlation(X, y, method)
        rank_full = np.empty(P, dtype=int)
        rank_full[sort_desc(corr_full)] = np.arange(1, P + 1)
        A = boot_abs_corr(X, y, B, bseed, method)
        R = ranks_desc(A)
        freqs = {}
        for k in ks:
            freqs[k] = bootstrap_topk_frequency(X, y, k, n_boot=B, random_state=bseed, method=method)
            sanity[f"ssc_matches_local_k{k}"] = bool(np.allclose(freqs[k], (R <= k).mean(axis=0)))
            sanity[f"freq_sum_equals_k{k}"] = bool(abs(freqs[k].sum() - k) < 1e-9)
        timings["bootstrap_seconds"] = time.perf_counter() - t0

        # ---- 2. permuted-label null curves --------------------------------
        t0 = time.perf_counter()
        n_perm, Bn = int(dc["null_permutations"]), int(dc["null_n_boot"])
        null_sorted = {k: np.empty((n_perm, P)) for k in ks}
        for p in range(n_perm):
            y_p = np.random.default_rng(int(dc["label_seed"]) + p).permutation(y)
            R_p = ranks_desc(boot_abs_corr(X, y_p, Bn, bseed + 1 + p, method))
            for k in ks:
                null_sorted[k][p] = np.sort((R_p <= k).mean(axis=0))[::-1]
        timings["null_seconds"] = time.perf_counter() - t0

        # ---- 3. drop position, stable sets, overlap, truth ---------------
        drop_rows, sets = [], {}
        for k in ks:
            row, s = drop_rules(freqs[k], k, B, null_sorted[k], dc)
            sets[k] = s
            if truth is not None:
                for r in RULES:
                    row[f"precision_{r}"], row[f"recall_{r}"] = precision_recall(s[r], truth)
                row["feature_auc_true_vs_noise"] = float(roc_auc_score(
                    [i in truth for i in range(P)], freqs[k]))
            drop_rows.append(row)
        drops = pd.DataFrame(drop_rows)

        freq_tab = pd.DataFrame({"feature": FEATURES, "abs_corr_full": corr_full, "rank_full": rank_full,
                                 "boot_rank_mean": R.mean(axis=0),
                                 "boot_rank_q05": np.quantile(R, 0.05, axis=0),
                                 "boot_rank_q95": np.quantile(R, 0.95, axis=0)})
        for k in ks:
            freq_tab[f"freq_k{k}"] = freqs[k]
            freq_tab[f"stable_{dc['primary_rule']}_k{k}"] = np.isin(np.arange(P), sets[k][dc["primary_rule"]])
        if truth is not None:
            freq_tab["is_true"] = [i in truth for i in range(P)]
        freq_tab = freq_tab.sort_values([f"freq_k{ks[-1]}", "abs_corr_full"], ascending=False).reset_index(drop=True)

        overlap_rows, set_rows = [], []
        for r in dict.fromkeys([dc["primary_rule"], *RULES]):
            for k in ks:
                set_rows.append({"rule": r, "k": k, "m": len(sets[k][r]),
                                 "features": " ".join(FEATURES[i] for i in sets[k][r])})
            for k1, k2 in combinations(ks, 2):
                a, b = set(sets[k1][r].tolist()), set(sets[k2][r].tolist())
                overlap_rows.append({"rule": r, "k1": k1, "k2": k2, "m1": len(a), "m2": len(b),
                                     "n_common": len(a & b), "jaccard": jaccard(a, b),
                                     "k1_set_in_k2_set": a <= b})
        overlap = pd.DataFrame(overlap_rows)
        stable_sets = pd.DataFrame(set_rows)

        # ---- 4. usefulness: stable vs all 150, identical splits ----------
        t0 = time.perf_counter()
        scheme = Scheme.from_dict(cfg["cv"])
        splits = make_splits(y, scheme)
        base, thr = cfg["pipeline"], float(uc["threshold"])
        nbf, fseed = int(uc["n_boot_fold"]), int(uc["fold_seed"])
        fold_all, summ_rows, pair_rows, diag_rows = [], [], [], []
        cv_cache: dict[str, pd.DataFrame] = {}  # keyed by the spec: identical variants are fitted once

        def evaluate(spec) -> pd.DataFrame:
            key = json.dumps(spec, sort_keys=True)
            if key not in cv_cache:
                cv = cross_validate(build_pipeline(spec), X, y, splits, scheme, test_pred="none", n_jobs=args.n_jobs)
                if cv.split_signature != sig:
                    raise RuntimeError("variants were not evaluated on identical splits")
                cv_cache[key] = cv.fold_scores
            return cv_cache[key]

        sig = split_signature(splits)  # every variant must be scored on exactly these splits
        fs_all = evaluate(with_selector(base, None))
        rule = dc["primary_rule"]
        for k in uc["ks"]:
            fk = freqs[k] if k in freqs else bootstrap_topk_frequency(X, y, k, B, bseed, method)
            S = stable_full(fk, thr)
            m = len(S)
            top_m = top_k_indices(corr_full, m)
            # the set above the drop m*(k) of the primary rule (threshold set if k has no drop analysis)
            D = sets[k][rule] if k in sets else S
            drop_fallback = len(D) == 0
            if drop_fallback:  # nothing above the drop: keep the single most frequent feature (as StabilityTopK)
                D = np.array([int(np.argmax(fk))])
            m_d = len(D)
            specs = {
                "naive_stable": with_selector(base, {"name": "fixed_columns", "columns": S.tolist()}),
                "naive_drop": with_selector(base, {"name": "fixed_columns", "columns": [int(i) for i in D]}),
                "naive_corr_top_m": with_selector(base, {"name": "fixed_columns", "columns": top_m.tolist()}),
                "honest_stable": with_selector(base, {"name": "stability_topk", "k": k, "n_boot": nbf,
                                                      "threshold": thr, "method": method, "random_state": fseed}),
                "honest_drop_top_m": with_selector(base, {"name": "corr_topk", "k": m_d, "method": method}),
                "honest_corr_top_m": with_selector(base, {"name": "corr_topk", "k": m, "method": method}),
            }
            fs = {"all150": fs_all}
            for v, spec in specs.items():
                fs[v] = evaluate(spec)

            # in-fold selections of the honest stable variant (diagnostics only)
            sel = Parallel(n_jobs=args.n_jobs)(delayed(_fold_selection)(X, y, s, k, nbf, thr, method, fseed)
                                               for s in splits)
            for rep, fold, idx in sel:
                d = {"k": k, "repeat": rep, "fold": fold, "n_selected": len(idx), "jaccard_with_full_set": jaccard(idx, S)}
                if truth is not None:
                    d["precision"], d["recall"] = precision_recall(idx, truth)
                diag_rows.append(d)
            diag_k = pd.DataFrame([d for d in diag_rows if d["k"] == k])

            n_feat = {"all150": float(P), "naive_stable": float(m), "naive_drop": float(m_d),
                      "naive_corr_top_m": float(m), "honest_stable": float(diag_k["n_selected"].mean()),
                      "honest_drop_top_m": float(m_d), "honest_corr_top_m": float(m)}
            # does the threshold set (as StabilityTopK uses it) equal the set above the drop m*(k)?
            j_drop = jaccard(S, sets[k][rule]) if k in sets else float("nan")
            for v in VARIANTS:
                rep = fs[v].groupby("repeat")["auc"].mean()
                summ_rows.append({"k": k, "variant": v, "auc_mean": float(rep.mean()),
                                  "auc_std_repeat": float(rep.std(ddof=1)) if len(rep) > 1 else float("nan"),
                                  "auc_std_fold": float(fs[v]["auc"].std(ddof=1)), "n_features_mean": n_feat[v],
                                  "stable_set_size_all_rows": m, "drop_set_size_all_rows": m_d,
                                  "drop_set_fallback_top1": drop_fallback,
                                  "jaccard_stable_vs_top_m": jaccard(S, top_m),
                                  "jaccard_stable_vs_drop_set": j_drop, "split_sig": sig})
                fold_all.append(fs[v].assign(k=k, variant=v))
            pairs = [(v, "all150") for v in VARIANTS[1:]] + [
                ("honest_stable", "honest_corr_top_m"), ("naive_stable", "honest_stable"),
                ("naive_stable", "naive_corr_top_m"), ("naive_drop", "honest_drop_top_m")]
            for a, b in pairs:
                pc = paired_compare(fs[a], fs[b])
                pair_rows.append({"k": k, "a": a, "b": b, **pc})
        fold_scores = pd.concat(fold_all, ignore_index=True)
        useful = pd.DataFrame(summ_rows)
        paired = pd.DataFrame(pair_rows)
        diag = pd.DataFrame(diag_rows)
        diag_summary = diag.groupby("k").agg(
            **{c: (c, "mean") for c in diag.columns if c not in ("k", "repeat", "fold")},
            n_selected_min=("n_selected", "min"), n_selected_max=("n_selected", "max")).reset_index()
        timings["usefulness_seconds"] = time.perf_counter() - t0

        # ---- 5. synthetic sanity: are the true features the stable ones? --
        if truth is not None:
            k_hi = ks[-1]
            sanity["truth_feature_auc_ge_0.9"] = bool(drops["feature_auc_true_vs_noise"].min() >= 0.9)
            sanity["truth_mean_freq_true_gt_3x_chance"] = bool(
                freq_tab.loc[freq_tab["is_true"], f"freq_k{k_hi}"].mean() > 3 * k_hi / P)
            sanity["truth_precision_primary_ge_0.8"] = bool(
                drops[f"precision_{dc['primary_rule']}"].fillna(0).min() >= 0.8)

        # ---- 6. save raw results, report tables, figures ------------------
        run.save_table("frequency", freq_tab)
        run.save_array("boot_abs_corr", A.astype(np.float32))
        run.save_table("null_sorted_curves", pd.concat(
            [pd.DataFrame(null_sorted[k], columns=[f"pos{i + 1}" for i in range(P)]).assign(k=k, permutation=range(n_perm))
             for k in ks], ignore_index=True))
        run.save_table("drop", drops)
        run.save_table("overlap", overlap)
        run.save_table("stable_sets", stable_sets)
        run.save_table("usefulness_fold_scores", fold_scores)
        run.save_table("usefulness", useful)
        run.save_table("paired", paired)
        run.save_table("infold_selection", diag)

        if args.quick:  # a smoke test never overwrites the report files
            tdir = fdir = run.dir / "report_preview"
        else:
            tdir, fdir = paths.tables_dir(source), paths.figures_dir(source)
        tdir.mkdir(parents=True, exist_ok=True)
        fdir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, df in (("frequency", freq_tab), ("drop", drops), ("overlap", overlap), ("stable_sets", stable_sets),
                         ("usefulness", useful), ("paired", paired), ("infold_selection", diag_summary)):
            df.to_csv(tdir / f"{PREFIX}{name}.csv", index=False)
            written.append(tdir / f"{PREFIX}{name}.csv")
        figs = {"frequency_sorted": fig_sorted(freqs, null_sorted, drops, truth, dc, B),
                "frequency_overlay": fig_overlay(freqs, B),
                "usefulness": fig_usefulness(useful, scheme.n_repeats)}
        for name, fig in figs.items():
            run.save_figure(name, fig)
        written += save_report_figures(figs, fdir, PREFIX, run.run_id)
        with open(tdir / f"{PREFIX}manifest.json", "w", encoding="utf-8") as fh:
            json.dump({"run_id": run.run_id, "data_source": source, "quick": args.quick, "created_at": now_iso(),
                       "files": [str(p) for p in written]}, fh, indent=2)

        total = time.perf_counter() - t_start
        timings["total"] = total
        run.summary.update({
            "scheme": scheme.to_dict(), "split_signature": sig,
            "m_star": {int(r["k"]): int(r["m_star"]) for _, r in drops.iterrows()},
            "primary_rule": dc["primary_rule"], "sanity": sanity,
            "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        })
        run.finish()

    # ---- console summary (ASCII only) -------------------------------------
    print()
    show = ["feature", "abs_corr_full"] + [f"freq_k{k}" for k in ks] + (["is_true"] if truth is not None else [])
    print(f"Top 12 features by bootstrap frequency (B={B}, sorted by top-{ks[-1]} frequency):")
    print(fmt_table(freq_tab.head(12), show, 3))
    print()
    print(f"Drop position m*(k) under each rule (primary = {dc['primary_rule']}):")
    dcols = ["k", "chance_rate", "m_gap", "gap_size", "m_half", "m_binom", "m_perm", "top_freq", "freq_at_rank_k"]
    if truth is not None:
        dcols += [f"precision_{dc['primary_rule']}", f"recall_{dc['primary_rule']}", "feature_auc_true_vs_noise"]
    print(fmt_table(drops, dcols, 3))
    print()
    print("Overlap of stable sets across k:")
    print(fmt_table(overlap[overlap["rule"].isin([dc["primary_rule"], "half"])],
                    ["rule", "k1", "k2", "m1", "m2", "n_common", "jaccard", "k1_set_in_k2_set"], 3))
    print()
    print(f"Usefulness ({scheme.label()}, identical splits, sig {sig}): mean CV AUC (SD over split seeds)")
    print(fmt_table(useful, ["k", "variant", "n_features_mean", "auc_mean", "auc_std_repeat"]))
    for k, g in useful.groupby("k"):
        r = g.iloc[0]
        print(f"  k={k}: stable set (freq >= {uc['threshold']}) on all rows has m={int(r['stable_set_size_all_rows'])};"
              f" set above the drop ({dc['primary_rule']} rule) has m*={int(r['drop_set_size_all_rows'])}"
              + (" (empty: top-1 feature used)" if r["drop_set_fallback_top1"] else "")
              + f"; Jaccard stable vs drop set = {r['jaccard_stable_vs_drop_set']:.3f},"
              f" stable vs top-m |corr| = {r['jaccard_stable_vs_top_m']:.3f}")
    print()
    print("Paired comparisons (a - b; split SE = std_diff_repeat / sqrt(repeats); p = Nadeau-Bengio corrected t-test):")
    print(fmt_table(paired, ["k", "a", "b", "mean_diff", "se_split", "win_rate_fold", "corrected_p"]))
    print()
    print("In-fold StabilityTopK selections (honest variant):")
    print(fmt_table(diag_summary, list(diag_summary.columns), 3))
    print()
    for key, ok in sanity.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {key}")
    print(f"time: bootstrap {timings['bootstrap_seconds']:.1f}s, null {timings['null_seconds']:.1f}s, "
          f"usefulness {timings['usefulness_seconds']:.1f}s, total {total:.1f}s")
    print(f"run folder: {run.dir}")
    print(f"tables:     {tdir}\\{PREFIX}*.csv" + ("   (--quick: report files untouched)" if args.quick else ""))
    print(f"figures:    {fdir}\\{PREFIX}*.png")
    return 0 if all(sanity.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
