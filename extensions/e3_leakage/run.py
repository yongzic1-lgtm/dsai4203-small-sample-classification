"""E3 - information leakage: feature selection outside vs inside the CV loop.

Question: the filter "keep the k features with the largest |corr(feature, target)|"
is applied (1) once on all 250 rows before k-fold CV (leaky) or (2) inside every
fold on that fold's training part only (honest). How much too high is the leaky
CV score, compared with the honest one and with the leaderboard, and how does
the gap change with the number of kept features (k_feat) and of folds (k_fold)?

Design (every grid cell uses IDENTICAL splits for both procedures):
  leaky   CorrelationTopK(k_feat) fitted once on all rows -> its columns frozen in
          Pipeline(scaler, FixedColumns(cols), model) -> repeated k-fold CV
  honest  Pipeline(scaler, CorrelationTopK(k_feat), model) -> repeated k-fold CV
  null    the same two procedures on permuted labels (no signal at all)
  LB      final model scaler + CorrelationTopK(k) + model fitted on all rows
          (identical for both procedures) -> test_pred.npy in the run folder

Report outputs (e3_grid/null/leaderboard.csv, e3_*.png, e3_manifest.json) go to
ssc.paths.tables_dir / figures_dir; with --quick to <run>/report_preview/ instead,
so a smoke test never overwrites the report files.

Run from the project root (--data defaults to $SSC_DATA, else raw):
    .venv\\Scripts\\python.exe extensions\\e3_leakage\\run.py --data synthetic --quick
    .venv\\Scripts\\python.exe extensions\\e3_leakage\\run.py                 # real data, full grid
    .venv\\Scripts\\python.exe extensions\\e3_leakage\\run.py --refresh-lb <run_id>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.data import FEATURES, load  # noqa: E402
from ssc.features import CorrelationTopK, abs_correlation, top_k_indices  # noqa: E402
from ssc.models import build_pipeline, predict_score  # noqa: E402
from ssc.tracking import Run, now_iso, read_registry  # noqa: E402
from ssc.validation import (Scheme, auc_standard_error, cross_validate, make_splits,  # noqa: E402
                            paired_compare, split_signature)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
PREFIX = "e3_"

# Report palette (categorical slots in fixed order) and chart ink.
# ext34 fix: 4th slot (yellow) added and indexing by len(SERIES): with 4 k_fold values (E3) or
# 4 k values (E4) the old `i % 3` gave the 1st and 4th line the same color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HONEST, LEAKY = SERIES[0], SERIES[1]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


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
    k_lb = int(cfg["leaderboard"]["k_feat"])
    grid = sorted({int(k) for k in cfg["grid"]["k_feat"]})
    if k_lb not in grid:
        print(f"note: leaderboard k_feat={k_lb} added to the k_feat grid")
        grid = sorted(set(grid) | {k_lb})
    if any(k < 1 or k > len(FEATURES) for k in grid):
        raise ValueError(f"k_feat values must be in 1..{len(FEATURES)}")
    cfg["grid"]["k_feat"] = grid
    cfg["grid"]["k_fold"] = sorted({int(k) for k in cfg["grid"]["k_fold"]})
    if cfg["null_experiment"].get("k_feat") is None:
        cfg["null_experiment"]["k_feat"] = list(grid)
    cfg["quick_mode"] = bool(quick)
    try:
        cfg["config_file"] = path.resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        cfg["config_file"] = str(path)
    return cfg


def public_fraction() -> float:
    try:
        cfg = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}
        return float(cfg["competition"]["public_fraction"])
    except Exception:  # noqa: BLE001 - fall back to the documented 2%
        return 0.02


def official_signature(y) -> str | None:
    """Split signature of project.yaml official_cv on these labels (None if it cannot be read).

    fix_ext34 (verify.md correction 22 / synthesis 20): the reference k_fold row is the official CV
    when its signature equals this one, and summary.json then says official_cv: true, so
    scripts/make_submission.py does not print its "not validated on the official CV" banner.
    """
    try:
        off = (yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}).get("official_cv")
        return split_signature(make_splits(y, Scheme.from_dict(off))) if off else None
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None


def load_truth(source: str) -> set[int] | None:
    """Indices of the truly informative features (synthetic data only)."""
    p = paths.data_dir(source) / "truth.csv"
    if not p.exists():
        return None
    names = pd.read_csv(p)["feature"].astype(str)
    return {FEATURES.index(n) for n in names}


# ---------------------------------------------------------------------------
# the two procedures
# ---------------------------------------------------------------------------
def with_selector(base: dict, selector: dict) -> dict:
    spec = copy.deepcopy(base)
    spec["selector"] = selector
    return spec


def leaky_columns(X: np.ndarray, y: np.ndarray, k_feat: int, method: str) -> np.ndarray:
    """Procedure 1: the filter sees ALL rows, including every later validation part."""
    return CorrelationTopK(k=k_feat, method=method).fit(X, y).get_support(indices=True)


def run_pair(X, y, splits, scheme: Scheme, base: dict, k_feat: int, method: str, n_jobs: int):
    """Leaky and honest CV of one (k_feat, k_fold) cell on the same splits."""
    cols = leaky_columns(X, y, k_feat, method)
    leaky = build_pipeline(with_selector(base, {"name": "fixed_columns", "columns": [int(c) for c in cols]}))
    honest = build_pipeline(with_selector(base, {"name": "corr_topk", "k": int(k_feat), "method": method}))
    cv_l = cross_validate(leaky, X, y, splits, scheme, test_pred="none", n_jobs=n_jobs)
    cv_h = cross_validate(honest, X, y, splits, scheme, test_pred="none", n_jobs=n_jobs)
    if cv_l.split_signature != cv_h.split_signature:
        raise RuntimeError("leaky and honest were not evaluated on identical splits")
    return cv_l.fold_scores, cv_h.fold_scores, cols, cv_l.split_signature


def repeat_means(fs: pd.DataFrame) -> pd.Series:
    """One CV estimate per split seed = mean fold AUC of that repeat."""
    return fs.groupby("repeat")["auc"].mean()


def summarise_pair(fs_l: pd.DataFrame, fs_h: pd.DataFrame) -> dict:
    rl, rh = repeat_means(fs_l), repeat_means(fs_h)
    gap = (rl - rh).to_numpy()
    m = fs_l.merge(fs_h, on=["repeat", "fold"], suffixes=("_l", "_h"), validate="one_to_one")
    pc = paired_compare(fs_l, fs_h)
    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")  # noqa: E731
    return {
        "leaky_mean": float(rl.mean()), "leaky_std": sd(rl.to_numpy()),
        "honest_mean": float(rh.mean()), "honest_std": sd(rh.to_numpy()),
        "gap_mean": float(gap.mean()), "gap_std": sd(gap),
        "gap_q05": float(np.quantile(gap, 0.05)), "gap_q95": float(np.quantile(gap, 0.95)),
        "gap_min": float(gap.min()), "gap_max": float(gap.max()),
        "frac_repeats_leaky_higher": float((gap > 0).mean()),
        "win_rate_fold": pc["win_rate_fold"],
        "corrected_t": pc["corrected_t"], "corrected_p": pc["corrected_p"],
        "max_abs_fold_diff": float(np.abs(m["auc_l"] - m["auc_h"]).max()),
        "n_repeats": int(len(rl)),
    }


def fold_overlap(X, y, splits, cols: np.ndarray, k_feat: int, method: str, truth: set[int] | None) -> dict:
    """How much of the all-rows feature set the honest in-fold selection reproduces."""
    full = set(int(c) for c in cols)
    overlap, n_true = [], []
    for s in splits:
        sel = set(int(c) for c in top_k_indices(abs_correlation(X[s.train_idx], y[s.train_idx], method), k_feat))
        overlap.append(len(full & sel) / k_feat)
        if truth is not None:
            n_true.append(len(truth & sel))
    out = {"fold_overlap_with_leaky_set": float(np.mean(overlap))}
    if truth is not None:
        out["n_true_in_leaky_set"] = len(truth & full)
        out["n_true_in_fold_set_mean"] = float(np.mean(n_true))
    return out


# ---------------------------------------------------------------------------
# leaderboard table
# ---------------------------------------------------------------------------
def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def lookup_lb(run_id: str, source: str) -> dict:
    """Latest submission of `run_id` in registry/submissions.csv (prefers one with a public score)."""
    reg = read_registry(paths.submissions_csv(source))
    if reg.empty or "run_id" not in reg.columns:
        return {}
    rows = reg[reg["run_id"] == run_id]
    if rows.empty:
        return {}
    filled = rows[rows["public_lb"].astype(str).str.strip() != ""]
    row = (filled if not filled.empty else rows).iloc[-1]
    return {"sub_id": row["sub_id"], "public_lb": _to_float(row.get("public_lb")),
            "private_lb": _to_float(row.get("private_lb"))}


def fill_lb(lb: pd.DataFrame, run_id: str, source: str, n_test: int, pos_rate: float) -> tuple[pd.DataFrame, bool]:
    lb = lb.copy()
    hit = lookup_lb(run_id, source)
    pub = hit.get("public_lb", float("nan"))
    lb["sub_id"] = hit.get("sub_id", "")
    lb["public_lb"] = pub
    lb["private_lb"] = hit.get("private_lb", float("nan"))
    if np.isfinite(pub):
        n_pub = int(round(public_fraction() * n_test))
        n_pos = max(1, int(round(n_pub * pos_rate)))
        lb["public_lb_se_approx"] = auc_standard_error(pub, n_pos, max(1, n_pub - n_pos))
    else:
        lb["public_lb_se_approx"] = float("nan")
    lb["leaky_minus_public"] = lb["leaky_cv_mean"] - lb["public_lb"]
    lb["honest_minus_public"] = lb["honest_cv_mean"] - lb["public_lb"]
    return lb, bool(np.isfinite(pub))


def add_synthetic_scores(lb: pd.DataFrame, ds, test_pred: np.ndarray) -> pd.DataFrame:
    """Synthetic data only: the hidden test labels are known, so score the final model directly."""
    from ssc.synthetic import score_submission
    s = score_submission(pd.DataFrame({"id": ds.ids_test, "target": test_pred}))
    return lb.assign(synthetic_public_auc=s["public_auc"], synthetic_private_auc=s["private_auc"])


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


def kfeat_axis(ax, kfeats) -> None:
    ax.set_xscale("log")
    ax.set_xticks(kfeats)
    ax.set_xticklabels([str(k) for k in kfeats])
    ax.xaxis.set_minor_locator(NullLocator())


def legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for t in leg.get_texts():
        t.set_color(INK2)


def fig_gap(cells: pd.DataFrame, n_rep: int):
    kfeats = sorted(cells["k_feat"].unique())
    kfolds = sorted(cells["k_fold"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    dodge = np.linspace(-0.05, 0.05, len(kfolds)) if len(kfolds) > 1 else [0.0]
    for i, kf in enumerate(kfolds):
        c = cells[cells["k_fold"] == kf].sort_values("k_feat")
        ax.errorbar(c["k_feat"] * np.exp(dodge[i]), c["gap_mean"], yerr=c["gap_std"], color=SERIES[i % len(SERIES)],
                    marker="o", ms=6, lw=2, capsize=3, label=f"{kf}-fold CV")
    ax.axhline(0.0, color=AXIS, lw=1)
    kfeat_axis(ax, kfeats)
    style(ax, f"Leakage gap = leaky CV AUC - honest CV AUC (mean +/- SD over {n_rep} split seeds)",
          "k_feat (number of features kept by the |corr| filter)", "AUC gap")
    legend(ax)
    fig.tight_layout()
    return fig


def fig_cv(cells: pd.DataFrame, n_rep: int):
    kfeats = sorted(cells["k_feat"].unique())
    kfolds = sorted(cells["k_fold"].unique())
    fig, axes = plt.subplots(1, len(kfolds), figsize=(3.2 * len(kfolds) + 0.6, 3.6), sharey=True, squeeze=False)
    for ax, kf in zip(axes[0], kfolds):
        c = cells[cells["k_fold"] == kf].sort_values("k_feat")
        for col, color, lab, d in (("leaky", LEAKY, "leaky (select on all rows)", 0.03),
                                   ("honest", HONEST, "honest (select inside folds)", -0.03)):
            ax.errorbar(c["k_feat"] * np.exp(d), c[f"{col}_mean"], yerr=c[f"{col}_std"], color=color,
                        marker="o", ms=6, lw=2, capsize=3, label=lab)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        kfeat_axis(ax, kfeats)
        style(ax, f"{kf}-fold CV", "k_feat", "CV AUC" if kf == kfolds[0] else "")
    legend(axes[0][0], loc="lower right")
    fig.suptitle(f"Leaky vs honest CV AUC (mean +/- SD over {n_rep} split seeds)", x=0.01, ha="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    return fig


def fig_null(null_cells: pd.DataFrame):
    kfeats = sorted(null_cells["k_feat"].unique())
    kfolds = sorted(null_cells["k_fold"].unique())
    fig, axes = plt.subplots(1, len(kfolds), figsize=(4.6 * len(kfolds) + 0.6, 3.6), sharey=True, squeeze=False)
    for ax, kf in zip(axes[0], kfolds):
        c = null_cells[null_cells["k_fold"] == kf].sort_values("k_feat")
        for col, color, lab, d in (("leaky", LEAKY, "leaky", 0.03), ("honest", HONEST, "honest", -0.03)):
            ax.errorbar(c["k_feat"] * np.exp(d), c[f"{col}_mean"], yerr=c[f"{col}_std"], color=color,
                        marker="o", ms=6, lw=2, capsize=3, label=lab)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        kfeat_axis(ax, kfeats)
        # ext34 fix: the long per-panel title overlapped the next panel when the null uses
        # several k_fold values; the shared part now goes into the suptitle.
        style(ax, f"{kf}-fold CV", "k_feat", "CV AUC" if kf == kfolds[0] else "")
    legend(axes[0][0], loc="upper right")
    fig.suptitle("Permuted labels: leaky vs honest CV AUC (mean +/- SD over permutations x split seeds)",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def fmt_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
    out = df[cols].copy()
    for c in cols:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return out.to_string(index=False)


def print_next_steps(run_id: str, source: str, k_lb: int | None = None) -> None:
    py, d = ".venv\\Scripts\\python.exe", (" --data synthetic" if source == "synthetic" else "")
    model = f"corr_topk({k_lb})" if k_lb else "corr_topk(k)"
    print("To fill the leaderboard column (one Kaggle submission of this run):")
    print(f"  1. {py} scripts\\make_submission.py --run {run_id} --desc \"E3 final model {model}\" "
          f"--expected-public <YOUR_GUESS>{d}")
    print("     (writes the CSV + registry row; expected-public = what you believe BEFORE uploading,")
    print("      e.g. the honest CV mean of the leaderboard table)")
    print(f"  2. upload the CSV (web page, or {py} scripts\\kaggle_submit.py --sub S###), then record the score:")
    print(f"     {py} scripts\\record_lb.py --sub S### --submitted-at now --public <SCORE>{d}")
    print("     (after kaggle_submit.py the time is already recorded: drop --submitted-at now)")
    print(f"  3. {py} extensions\\e3_leakage\\run.py --data {source} --refresh-lb {run_id}")


def report_dirs(run_dir: Path, source: str, quick: bool) -> tuple[Path, Path]:
    """Report tables/figures; a --quick (smoke-test) run never overwrites the real report files."""
    if quick:
        d = run_dir / "report_preview"
        return d, d
    return paths.tables_dir(source), paths.figures_dir(source)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def refresh_lb(run_id: str, source: str) -> int:
    run_dir = paths.experiments_dir(source) / run_id
    base = run_dir / "leaderboard.csv"
    if not base.exists():
        print(f"error: {base} not found (is --data correct?)")
        return 1
    lb = pd.read_csv(base)
    ds = load(source)
    lb, found = fill_lb(lb, run_id, source, len(ds.ids_test), float(ds.y.mean()))
    if source == "synthetic" and (run_dir / "test_pred.npy").exists():
        lb = add_synthetic_scores(lb, ds, np.load(run_dir / "test_pred.npy"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    tdir, _ = report_dirs(run_dir, source, bool(cfg.get("quick_mode")))
    out = tdir / f"{PREFIX}leaderboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out, index=False)
    extra = [c for c in ("synthetic_public_auc", "synthetic_private_auc") if c in lb]
    print(fmt_table(lb, ["k_feat", "k_fold", "leaky_cv_mean", "honest_cv_mean", "gap_mean", "sub_id",
                         "public_lb", "public_lb_se_approx", "private_lb"] + extra))
    print(f"written: {out}")
    if not found:
        print(f"no public score for {run_id} in {paths.submissions_csv(source)} yet.")
        print_next_steps(run_id, source, int(lb["k_feat"].iloc[0]) if len(lb) else None)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", choices=paths.SOURCES, default=None,
                    help="raw (Kaggle, real archive) or synthetic (sandbox) (default: $SSC_DATA or raw)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--quick", action="store_true",
                    help="small grid for a smoke test (report tables/figures go to <run>/report_preview/)")
    ap.add_argument("--author", default=None)
    ap.add_argument("--n-jobs", type=int, default=1, help="joblib workers for the CV fits")
    ap.add_argument("--refresh-lb", metavar="RUN_ID", default=None,
                    help="no computation: re-read the LB score of an earlier E3 run from the registry")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    source = paths.resolve_source(args.data)
    if args.refresh_lb:
        return refresh_lb(args.refresh_lb, source)

    if not args.config.exists() and not args.config.is_absolute() and (paths.ROOT / args.config).exists():
        args.config = paths.ROOT / args.config  # relative to the project root when not found from the cwd
    cfg = load_config(args.config, args.quick)
    cfg["data_source"] = source
    ds = load(source)
    X, y = ds.X.to_numpy(dtype=float), ds.y
    truth = load_truth(source)
    base, method = cfg["pipeline"], cfg["selection"]["method"]
    kfeats, kfolds = cfg["grid"]["k_feat"], cfg["grid"]["k_fold"]
    R, seed, strat = int(cfg["cv"]["n_repeats"]), int(cfg["cv"]["seed"]), bool(cfg["cv"].get("stratify", True))
    k_lb = int(cfg["leaderboard"]["k_feat"])
    k_ref = int(cfg["leaderboard"]["k_fold_reference"])
    if k_ref not in kfolds:
        print(f"note: k_fold_reference={k_ref} not in the grid; using {kfolds[0]}")
        k_ref = kfolds[0]

    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    with Run(cfg["name"], cfg, kind="extension", author=args.author, source=source,
             hypothesis=cfg.get("hypothesis", "")) as run:
        print(f"E3 information leakage | data={source} | quick={args.quick} | run_id={run.run_id}")

        # ---- 1. main grid ------------------------------------------------
        t0 = time.perf_counter()
        fold_rows, rep_rows, cells, schemes = [], [], [], {}
        for kf in kfolds:
            scheme = Scheme(kind="kfold", n_splits=kf, n_repeats=R, seed=seed, stratify=strat)
            splits = make_splits(y, scheme)
            schemes[kf] = scheme
            for kfeat in kfeats:
                fs_l, fs_h, cols, sig = run_pair(X, y, splits, scheme, base, kfeat, method, args.n_jobs)
                for proc, fs in (("leaky", fs_l), ("honest", fs_h)):
                    fold_rows.append(fs.assign(procedure=proc, k_feat=kfeat, k_fold=kf, split_sig=sig))
                rl, rh = repeat_means(fs_l), repeat_means(fs_h)
                rep_rows.append(pd.DataFrame({"k_feat": kfeat, "k_fold": kf, "repeat": rl.index,
                                              "leaky": rl.to_numpy(), "honest": rh.to_numpy(),
                                              "gap": (rl - rh).to_numpy()}))
                cells.append({"k_feat": kfeat, "k_fold": kf, "split_sig": sig, **summarise_pair(fs_l, fs_h),
                              **fold_overlap(X, y, splits, cols, kfeat, method, truth)})
            print(f"  grid: {kf}-fold done ({time.perf_counter() - t0:.1f}s)")
        fold_scores = pd.concat(fold_rows, ignore_index=True)
        repeat_scores = pd.concat(rep_rows, ignore_index=True)
        cells = pd.DataFrame(cells)
        timings["grid_seconds"] = time.perf_counter() - t0

        # ---- 2. null experiment (permuted labels) ------------------------
        t0 = time.perf_counter()
        nc = cfg["null_experiment"]
        null_rows, null_fold_rows = [], []
        for p in range(int(nc["n_permutations"])):
            y_p = np.random.default_rng(int(nc["label_seed"]) + p).permutation(y)
            for kf in nc["k_fold"]:
                scheme = Scheme(kind="kfold", n_splits=int(kf), n_repeats=int(nc["n_repeats"]), seed=seed, stratify=strat)
                splits = make_splits(y_p, scheme)
                for kfeat in nc["k_feat"]:
                    fs_l, fs_h, _, sig = run_pair(X, y_p, splits, scheme, base, int(kfeat), method, args.n_jobs)
                    for proc, fs in (("leaky", fs_l), ("honest", fs_h)):
                        null_fold_rows.append(fs.assign(procedure=proc, permutation=p, k_feat=kfeat, k_fold=kf))
                    rl, rh = repeat_means(fs_l), repeat_means(fs_h)
                    null_rows.append(pd.DataFrame({"permutation": p, "k_feat": kfeat, "k_fold": kf, "repeat": rl.index,
                                                   "leaky": rl.to_numpy(), "honest": rh.to_numpy(),
                                                   "gap": (rl - rh).to_numpy()}))
        null_rep = pd.concat(null_rows, ignore_index=True)
        null_folds = pd.concat(null_fold_rows, ignore_index=True)
        g = null_rep.groupby(["k_feat", "k_fold"])
        null_cells = g.agg(leaky_mean=("leaky", "mean"), leaky_std=("leaky", "std"),
                           honest_mean=("honest", "mean"), honest_std=("honest", "std"),
                           gap_mean=("gap", "mean"), gap_std=("gap", "std"), n=("gap", "size")).reset_index()
        perm_level = null_rep.groupby(["k_feat", "k_fold", "permutation"])["gap"].mean().reset_index()
        null_cells = null_cells.merge(
            perm_level.groupby(["k_feat", "k_fold"])["gap"].agg(
                frac_perm_leaky_higher=lambda s: float((s > 0).mean()), gap_std_perm="std").reset_index(),
            on=["k_feat", "k_fold"])
        timings["null_seconds"] = time.perf_counter() - t0

        # ---- 3. final model on all rows + leaderboard table --------------
        t0 = time.perf_counter()
        final_spec = with_selector(base, {"name": "corr_topk", "k": k_lb, "method": method})
        final = build_pipeline(final_spec).fit(X, y)
        test_pred = predict_score(final, ds.X_test.to_numpy(dtype=float))
        run.save_array("test_pred", test_pred)  # -> test_pred.npy, usable by scripts/make_submission.py
        sel = final.named_steps["select"]
        final_features = pd.DataFrame({"feature": FEATURES, "abs_corr": sel.scores_, "selected": sel.support_})
        if truth is not None:
            final_features["is_true"] = [i in truth for i in range(len(FEATURES))]
        final_features = final_features.sort_values("abs_corr", ascending=False).reset_index(drop=True)
        lb = cells[cells["k_feat"] == k_lb].sort_values("k_fold")
        lb = pd.DataFrame({
            "run_id": run.run_id, "data_source": source, "k_feat": k_lb, "k_fold": lb["k_fold"].to_numpy(),
            "n_repeats": lb["n_repeats"].to_numpy(),
            "leaky_cv_mean": lb["leaky_mean"].to_numpy(), "leaky_cv_std": lb["leaky_std"].to_numpy(),
            "honest_cv_mean": lb["honest_mean"].to_numpy(), "honest_cv_std": lb["honest_std"].to_numpy(),
            "gap_mean": lb["gap_mean"].to_numpy(), "is_reference_k_fold": (lb["k_fold"] == k_ref).to_numpy(),
        })
        run.save_table("leaderboard", lb)  # CV part only; --refresh-lb adds the LB columns later
        lb_full, lb_found = fill_lb(lb, run.run_id, source, len(ds.ids_test), float(y.mean()))
        if source == "synthetic":
            lb_full = add_synthetic_scores(lb_full, ds, test_pred)
        timings["final_seconds"] = time.perf_counter() - t0

        # ---- 4. save raw results, report tables, figures -----------------
        run.save_table("fold_scores_grid", fold_scores)
        run.save_table("repeat_scores_grid", repeat_scores)
        run.save_table("cells", cells)
        run.save_table("null_fold_scores", null_folds)
        run.save_table("null_repeat_scores", null_rep)
        run.save_table("null_cells", null_cells)
        run.save_table("final_features", final_features)

        tdir, fdir = report_dirs(run.dir, source, args.quick)
        tdir.mkdir(parents=True, exist_ok=True)
        fdir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, df in (("grid", cells), ("null", null_cells), ("leaderboard", lb_full)):
            df.to_csv(tdir / f"{PREFIX}{name}.csv", index=False)
            written.append(tdir / f"{PREFIX}{name}.csv")
        for name, fig in (("gap_vs_kfeat", fig_gap(cells, R)), ("cv_vs_kfeat", fig_cv(cells, R)),
                          ("null", fig_null(null_cells))):
            run.save_figure(name, fig)
            fig.savefig(fdir / f"{PREFIX}{name}.png", dpi=150, bbox_inches="tight")
            written.append(fdir / f"{PREFIX}{name}.png")
            plt.close(fig)
        with open(tdir / f"{PREFIX}manifest.json", "w", encoding="utf-8") as fh:
            json.dump({"run_id": run.run_id, "data_source": source, "quick": args.quick, "created_at": now_iso(),
                       "files": [str(p) for p in written]}, fh, indent=2)

        # ---- 5. sanity checks + registry summary -------------------------
        sanity = {}
        if len(FEATURES) in kfeats:
            z = cells[cells["k_feat"] == len(FEATURES)]["max_abs_fold_diff"]
            sanity["gap_zero_at_150"] = bool((z == 0.0).all())
        nk = null_cells[null_cells["k_feat"] < len(FEATURES)]
        if len(nk):
            sanity["null_leaky_above_honest"] = bool((nk["gap_mean"] > 0).all())
            sanity["null_honest_near_half"] = bool((nk["honest_mean"] - 0.5).abs().max() < 0.05)
            sanity["null_leaky_above_half"] = bool((nk["leaky_mean"] > 0.5).all())
        ref = cells[(cells["k_feat"] == k_lb) & (cells["k_fold"] == k_ref)].iloc[0]
        ref_h = fold_scores[(fold_scores["procedure"] == "honest") & (fold_scores["k_feat"] == k_lb)
                            & (fold_scores["k_fold"] == k_ref)]
        off_sig = official_signature(y)
        run.summary.update({
            "official_cv": bool(off_sig is not None and ref["split_sig"] == off_sig),
            "official_split_signature": off_sig,
            "auc_mean": float(ref["honest_mean"]), "auc_std_repeat": float(ref["honest_std"]),
            "auc_std_fold": float(ref_h["auc"].std(ddof=1)), "split_signature": ref["split_sig"],
            "scheme": schemes[k_ref].to_dict(), "leaky_auc_mean": float(ref["leaky_mean"]),
            "gap_mean_reference": float(ref["gap_mean"]), "leaderboard_k_feat": k_lb,
            "max_gap": {"k_feat": int(cells.loc[cells["gap_mean"].idxmax(), "k_feat"]),
                        "k_fold": int(cells.loc[cells["gap_mean"].idxmax(), "k_fold"]),
                        "gap_mean": float(cells["gap_mean"].max())},
            "sanity": sanity, "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        })
        if "synthetic_public_auc" in lb_full:
            run.summary["synthetic_test_auc"] = {"public": float(lb_full["synthetic_public_auc"].iloc[0]),
                                                 "private": float(lb_full["synthetic_private_auc"].iloc[0])}
        total = time.perf_counter() - t_start
        run.summary["timings_seconds"]["total"] = round(total, 2)
        # registry columns describe the model behind test_pred.npy (selector included);
        # config.yaml in the run folder keeps the base pipeline as configured
        run.finish(describe_spec=final_spec)

    # ---- console summary (ASCII only) -------------------------------------
    print()
    print(f"Grid: mean CV AUC over {R} split seeds (SD in *_std); gap = leaky - honest; "
          "p = Nadeau-Bengio corrected paired t-test")
    print(fmt_table(cells, ["k_feat", "k_fold", "leaky_mean", "leaky_std", "honest_mean", "honest_std",
                            "gap_mean", "gap_std", "frac_repeats_leaky_higher", "corrected_p",
                            "fold_overlap_with_leaky_set"]))
    print()
    print(f"Null (labels permuted {nc['n_permutations']}x, {nc['n_repeats']} split seeds each):")
    print(fmt_table(null_cells, ["k_feat", "k_fold", "leaky_mean", "leaky_std", "honest_mean", "honest_std",
                                 "gap_mean", "frac_perm_leaky_higher"]))
    print()
    print(f"Leaderboard comparison (final model: scaler + CorrelationTopK({k_lb}) + model on all rows):")
    lb_cols = ["k_fold", "leaky_cv_mean", "honest_cv_mean", "gap_mean", "public_lb"]
    lb_cols += [c for c in ("synthetic_public_auc", "synthetic_private_auc") if c in lb_full]
    print(fmt_table(lb_full, lb_cols))
    print()
    labels = {"gap_zero_at_150": "gap exactly 0 at k_feat=150 (every fold, every k_fold)",
              "null_leaky_above_honest": "null: leaky > honest for every k_feat < 150",
              "null_honest_near_half": "null: |honest - 0.5| < 0.05 for every k_feat < 150",
              "null_leaky_above_half": "null: leaky > 0.5 for every k_feat < 150"}
    for key, ok in sanity.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {labels[key]}")
    print(f"time: grid {timings['grid_seconds']:.1f}s, null {timings['null_seconds']:.1f}s, total {total:.1f}s")
    print(f"run folder: {run.dir}")
    print(f"tables:     {tdir}\\{PREFIX}*.csv" + ("   (--quick: report files untouched)" if args.quick else ""))
    print(f"figures:    {fdir}\\{PREFIX}*.png")
    if not lb_found:
        print_next_steps(run.run_id, source, k_lb)
    return 0 if all(sanity.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
