"""E2 Learning curve: CV AUC as a function of the number of training rows, with
subset-to-subset spread, and an extrapolation to 500 training rows with a
bootstrap interval.

For every size n in config.yaml, `n_subsets` stratified random subsets of the
training rows are drawn and each is scored by repeated stratified k-fold CV
inside the subset (ssc.validation). For n < N an out-of-subset estimate (fit on
the whole subset, score on the rows not in it) is added as a second view. At
n = N every subset is the full data and only the CV seed varies, so the spread
there is evaluation noise only.

An inverse power law AUC(x) = a - b * x^(-c) is fitted to the per-size means
(weighted by their standard error) and extrapolated to x = target_n, where x is
either the effective training size n*(k-1)/k or the nominal subset size n
(config fit.x_axis). A bootstrap over the subset scores at each size gives the
interval; a log-linear model AUC(x) = alpha + beta * ln(x) is the sensitivity check.

Outputs
-------
run folder (ssc.tracking.Run, kind="extension"):
    subset_scores.csv        one row per (size, subset): CV AUC, out-of-subset AUC, seeds
    fold_scores_all.csv      every fold AUC behind the CV scores
    subset_membership.npy    bool (rows of subset_scores, N): which rows each subset used
    bootstrap.csv            one row per bootstrap replicate: refitted parameters + predictions
    ground_truth.csv         synthetic only: "true" learning curve from the mock test labels
    *.png, learning_curve_table.csv, extrapolation.csv   copies of the report outputs
report outputs (ssc.paths.tables_dir / figures_dir; in --quick mode <run>/report_preview/):
    e2_learning_curve_table.{csv,md}   per size: mean / std / min / max (CV and out-of-subset)
    e2_extrapolation.{csv,md}          power-law and log-linear predictions with intervals
    e2_ground_truth.{csv,md}           synthetic only
    e2_learning_curve.png              points, mean +/- std, fit, extrapolation, nominal 90% band
                                       (drawn at its printed size, about 70% of the A4 text width)
    e2_bootstrap_target.png            bootstrap distribution of the predicted AUC at target_n
    e2_manifest.json                   which run produced these files

The 90% band and interval are "nominal": they resample the subset scores of one sample, and on
replica data they covered the true AUC(500) in only 21% of samples (R942, research/reports/verify_ext.md).

Usage (from the project root):
    .venv\\Scripts\\python.exe extensions\\e2_learning_curve\\run.py --data synthetic --quick
    .venv\\Scripts\\python.exe extensions\\e2_learning_curve\\run.py --n-jobs 8
    .venv\\Scripts\\python.exe extensions\\e2_learning_curve\\run.py --replot 20261022-070248_e2_learning_curve
--replot RUN_ID only redraws the report figures of an archived run from the tables in its run folder
(nothing is refitted; no run folder, registry row or report table is written). The output prefix
(e2_, e2_qda_, e2_gmm_) is taken from the reports/tables/e2*_manifest.json that names the run, so the
figures keep their file names; --replot-out DIR writes them elsewhere (e.g. a preview folder).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import textwrap
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import OptimizeWarning, curve_fit  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tabulate import tabulate  # noqa: E402

from ssc import data as ssc_data  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.models import build_pipeline, describe, predict_score  # noqa: E402
from ssc.tracking import Run, now_iso  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits, split_signature  # noqa: E402

HERE = Path(__file__).resolve().parent
PREFIX = "e2_"
# columns of the report-ready .md table (the .csv keeps every column)
TABLE_MD_COLS = ["size", "x", "n_subsets", "cv_mean", "cv_std", "cv_min", "cv_max",
                 "oos_mean", "oos_std", "oos_min", "oos_max", "n_oos_rows", "frac_constant_folds", "spread_source"]

# Chart tokens (reference palette of the dataviz guide; light surface).
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
# The learning-curve figure is drawn at the size it is printed: about 70% of the text width of an
# A4 page with 2.54 cm margins (the width at which the E2 draft fits 2 pages). Font sizes are
# therefore printed sizes in pt (the earlier 9 x 5.2 in figure printed its legend at about 3.5 pt).
CURVE_SIZE = (4.4, 2.5)
FS = {"title": 7, "label": 7, "tick": 6.5, "legend": 6.5}
REPORT_DPI = {"learning_curve": 300, "bootstrap_target": 200}  # report PNGs (pHYs = dpi, so Word keeps the size)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(path: Path, quick: bool) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if quick:
        q = cfg.get("quick") or {}
        cfg["sizes"] = list(q.get("sizes", cfg["sizes"]))
        cfg["n_subsets"] = int(q.get("n_subsets", cfg["n_subsets"]))
        cfg["cv"]["n_repeats"] = int(q.get("cv_n_repeats", cfg["cv"]["n_repeats"]))
        cfg["fit"]["n_boot"] = int(q.get("n_boot", cfg["fit"]["n_boot"]))
        if cfg["fit"].get("sensitivity") and "sens_n_boot" in q:
            cfg["fit"]["sensitivity"]["n_boot"] = int(q["sens_n_boot"])
        gt = cfg.get("ground_truth") or {}
        for key, qkey in (("n_draws", "gt_n_draws"), ("n_eval", "gt_n_eval")):
            if key in gt and qkey in q:
                gt[key] = int(q[qkey])
    if cfg["cv_seed_stride"] < cfg["cv"]["n_repeats"]:
        raise ValueError("cv_seed_stride must be >= cv.n_repeats (subsets would share CV seeds)")
    if cfg["fit"]["x_axis"] not in ("effective", "nominal"):
        raise ValueError("fit.x_axis must be 'effective' or 'nominal'")
    cfg["sizes"] = sorted(int(n) for n in cfg["sizes"])
    return cfg


def x_of(n: int, cfg: dict) -> float:
    """Position of nominal size n on the fit axis."""
    k = int(cfg["cv"]["n_splits"])
    return n * (k - 1) / k if cfg["fit"]["x_axis"] == "effective" else float(n)


def x_label(cfg: dict) -> str:
    if cfg["fit"]["x_axis"] == "effective":
        k = cfg["cv"]["n_splits"]
        return f"training rows behind each score (effective size n*{k - 1}/{k})"
    return "nominal subset size n"


# ---------------------------------------------------------------------------
# subsets and scoring
# ---------------------------------------------------------------------------
def stratified_subset(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n row indices drawn without replacement, class shares as in y (largest remainder)."""
    N = len(y)
    if n >= N:
        return np.arange(N)
    classes, counts = np.unique(y, return_counts=True)
    quota = counts * n / N
    take = np.floor(quota).astype(int)
    order = np.argsort(-(quota - take), kind="stable")
    take[order[: n - take.sum()]] += 1
    idx = [rng.choice(np.flatnonzero(y == c), size=t, replace=False) for c, t in zip(classes, take)]
    return np.sort(np.concatenate(idx))


def score_subset(n: int, i: int, cfg: dict, X: np.ndarray, y: np.ndarray) -> dict:
    N = len(y)
    idx = stratified_subset(y, n, np.random.default_rng([int(cfg["subset_seed"]), n, i]))
    rest = np.setdiff1d(np.arange(N), idx)
    cv_seed = int(cfg["cv"]["seed"]) + i * int(cfg["cv_seed_stride"])
    scheme = Scheme(kind="kfold", n_splits=int(cfg["cv"]["n_splits"]), n_repeats=int(cfg["cv"]["n_repeats"]),
                    seed=cv_seed)
    Xs, ys = X[idx], y[idx]
    splits = make_splits(ys, scheme)
    cv = cross_validate(build_pipeline(cfg["pipeline"]), Xs, ys, splits, scheme, test_pred="none", n_jobs=1)
    fs = cv.fold_scores
    # folds whose model predicts one constant score (e.g. L1 with every coefficient at 0) score AUC 0.5
    n_const = sum(np.ptp(cv.oof[sp.repeat, sp.val_idx]) < 1e-12 for sp in splits)
    oos, n_oos = np.nan, 0
    if cfg.get("out_of_subset", True) and len(rest) > 0 and len(np.unique(y[rest])) == 2:
        model = build_pipeline(cfg["pipeline"]).fit(Xs, ys)
        oos, n_oos = float(roc_auc_score(y[rest], predict_score(model, X[rest]))), int(len(rest))
    row = {
        "size": n, "subset": i, "x": x_of(n, cfg), "n_train_eff": float(fs["n_train"].mean()),
        "cv_auc": float(fs["auc"].mean()), "cv_fold_std": float(fs["auc"].std(ddof=1)),
        "cv_pooled_oof_auc": float(np.mean([roc_auc_score(ys, r) for r in cv.oof])),
        "frac_constant_folds": float(n_const / len(splits)),
        "oos_auc": oos, "n_oos": n_oos, "n_pos": int(ys.sum()),
        "subset_is_full": bool(n >= N), "cv_seed": cv_seed,
        "subset_hash": hashlib.sha1(idx.astype(np.int64).tobytes()).hexdigest()[:10],
        "split_signature": split_signature(splits),
    }
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return {"row": row, "folds": fs.assign(size=n, subset=i), "mask": mask}


def size_table(sub: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    m = cfg["n_subsets"]
    for n, g in sub.groupby("size", sort=True):
        o = g["oos_auc"].dropna()
        full = bool(g["subset_is_full"].all())
        rows.append({
            "size": n, "x": g["x"].iloc[0], "n_train_eff": g["n_train_eff"].mean(), "n_subsets": len(g),
            "cv_mean": g["cv_auc"].mean(), "cv_std": g["cv_auc"].std(ddof=1), "cv_sem": g["cv_auc"].std(ddof=1) / np.sqrt(m),
            "cv_min": g["cv_auc"].min(), "cv_max": g["cv_auc"].max(),
            "oos_mean": o.mean() if len(o) else np.nan, "oos_std": o.std(ddof=1) if len(o) > 1 else np.nan,
            "oos_min": o.min() if len(o) else np.nan, "oos_max": o.max() if len(o) else np.nan,
            "n_oos_rows": int(g["n_oos"].iloc[0]),
            "frac_constant_folds": g["frac_constant_folds"].mean(),
            "spread_source": "CV seed only (subset = full data)" if full else "subset draw + CV seed",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# curve fitting
# ---------------------------------------------------------------------------
def power_law(x, a, b, c):
    return a - b * np.power(x, -c)


def _sigma(std: np.ndarray, cfg: dict, weight: str) -> np.ndarray | None:
    floor = cfg["fit"]["sigma_floor"]
    if weight == "none":
        return None
    s = std / np.sqrt(cfg["n_subsets"]) if weight == "sem" else std
    return np.maximum(np.nan_to_num(s, nan=floor), floor)


def fit_power(x: np.ndarray, m: np.ndarray, sigma: np.ndarray | None, cfg: dict) -> np.ndarray | None:
    b = cfg["fit"]["bounds"]
    lo = np.array([b["a"][0], b["b"][0], b["c"][0]], float)
    hi = np.array([b["a"][1], b["b"][1], b["c"][1]], float)
    a0 = min(m.max() + 0.03, hi[0] - 1e-6)
    c0 = 0.5
    b0 = max((a0 - m.min()) * x.min() ** c0, 1e-3)
    p0 = np.clip([a0, b0, c0], lo + 1e-9, hi - 1e-9)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(power_law, x, m, p0=p0, sigma=sigma, absolute_sigma=True,
                                bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    return popt


def fit_loglin(x: np.ndarray, m: np.ndarray, sigma: np.ndarray | None) -> np.ndarray:
    w = None if sigma is None else 1.0 / sigma
    beta, alpha = np.polyfit(np.log(x), m, 1, w=w)
    return np.array([alpha, beta])


def loglin(x, alpha, beta):
    return alpha + beta * np.log(x)


def at_bounds(popt: np.ndarray, cfg: dict) -> list[str]:
    b = cfg["fit"]["bounds"]
    out = []
    for name, v in zip("abc", popt):
        lo, hi = b[name]
        if np.isclose(v, lo, atol=1e-6 * max(1, abs(lo))) or np.isclose(v, hi, atol=1e-6 * max(1, abs(hi))):
            out.append(name)
    return out


def _std(groups: list[np.ndarray]) -> np.ndarray:
    return np.array([g.std(ddof=1) if len(g) > 1 else np.nan for g in groups])


def _fit_variant(x, groups, cfg, weight, n_boot, seed, targets, loglin_too: bool):
    """Point fit on the per-size means + bootstrap (resample the subset scores within each size)."""
    m = np.array([g.mean() for g in groups])
    sig = _sigma(_std(groups), cfg, weight)
    pl = fit_power(x, m, sig, cfg)
    ll = fit_loglin(x, m, sig) if loglin_too else None
    rng = np.random.default_rng(int(seed))
    rows = []
    for bi in range(int(n_boot)):
        gb = [rng.choice(g, size=len(g), replace=True) for g in groups]
        mb, sb = np.array([g.mean() for g in gb]), _sigma(_std(gb), cfg, weight)
        pb = fit_power(x, mb, sb, cfg)
        row = {"b": bi, "pl_ok": pb is not None}
        row.update({f"pl_{k}": (pb[j] if pb is not None else np.nan) for j, k in enumerate("abc")})
        for t in targets:
            row[f"pl_pred_{t:g}"] = power_law(t, *pb) if pb is not None else np.nan
        if loglin_too:
            lb = fit_loglin(x, mb, sb)
            row.update({"ll_alpha": lb[0], "ll_beta": lb[1]})
            for t in targets:
                row[f"ll_pred_{t:g}"] = loglin(t, *lb)
        rows.append(row)
    return pl, ll, pd.DataFrame(rows)


def _ext_rows(name, tag, fn, params, boot, targets, cfg, note, lo_q, hi_q) -> list[dict]:
    rows = []
    for t in targets:
        col = boot[f"{tag}_pred_{t:g}"].dropna()
        rows.append({
            "model": name, "x_axis": cfg["fit"]["x_axis"], "x": t,
            "point": float(fn(t, *params)) if params is not None else np.nan,
            "boot_median": col.median(), f"boot_lo_{lo_q:.2f}": col.quantile(lo_q),
            f"boot_hi_{hi_q:.2f}": col.quantile(hi_q),
            "interval_width": col.quantile(hi_q) - col.quantile(lo_q),
            "frac_boot_above_1": float((col > 1).mean()) if len(col) else np.nan,
            "n_boot_ok": int(len(col)), "n_boot_failed": int(len(boot) - len(col)),
            "at_bounds": ",".join(at_bounds(params, cfg)) if (tag == "pl" and params is not None) else "",
            "note": note,
        })
    return rows


def fit_and_bootstrap(sub: pd.DataFrame, cfg: dict) -> dict:
    f = cfg["fit"]
    sizes = sorted(int(n) for n in sub["size"].unique())
    groups = [sub[sub["size"] == n]["cv_auc"].to_numpy() for n in sizes]
    x = np.array([x_of(n, cfg) for n in sizes])
    targets = sorted(set([float(t) for t in f.get("predict_at", [])] + [float(f["target_n"])]))
    lo_q, hi_q = (1 - f["interval"]) / 2, 1 - (1 - f["interval"]) / 2

    # main fit: power law (weighted as configured) + log-linear sensitivity check
    pl, ll, boot = _fit_variant(x, groups, cfg, f["weight"], f["n_boot"], f["boot_seed"], targets, True)
    rows = _ext_rows("power_law", "pl", power_law, pl, boot, targets, cfg,
                     f"main fit, weight={f['weight']}", lo_q, hi_q)
    rows += _ext_rows("log_linear", "ll", loglin, ll, boot, targets, cfg, "sensitivity: other curve form", lo_q, hi_q)

    # further sensitivity checks of the power law (weighting / smallest size)
    sens = f.get("sensitivity") or {}
    nb = int(sens.get("n_boot", f["n_boot"]))
    if sens.get("unweighted", False) and f["weight"] != "none":
        p2, _, b2 = _fit_variant(x, groups, cfg, "none", nb, int(f["boot_seed"]) + 1, targets, False)
        rows += _ext_rows("power_law_unweighted", "pl", power_law, p2, b2, targets, cfg,
                          "sensitivity: weight=none", lo_q, hi_q)
    if sens.get("drop_smallest", False) and len(sizes) >= 4:
        p3, _, b3 = _fit_variant(x[1:], groups[1:], cfg, f["weight"], nb, int(f["boot_seed"]) + 2, targets, False)
        rows += _ext_rows(f"power_law_drop_{sizes[0]}", "pl", power_law, p3, b3, targets, cfg,
                          f"sensitivity: size {sizes[0]} left out", lo_q, hi_q)
    ext = pd.DataFrame(rows)
    # Envelope = union of the main and sensitivity intervals. Every interval here resamples only the
    # subset scores; all subsets come from ONE 250-row sample, so parent-sample noise is still missing.
    lo_c, hi_c = f"boot_lo_{lo_q:.2f}", f"boot_hi_{hi_q:.2f}"
    env = []
    for t in targets:
        g = ext[ext["x"] == t]
        lo, hi = float(g[lo_c].min()), float(g[hi_c].max())
        env.append({"model": "envelope", "x_axis": f["x_axis"], "x": t,
                    "point": float(g.loc[g["model"] == "power_law", "point"].iloc[0]), "boot_median": np.nan,
                    lo_c: lo, hi_c: hi, "interval_width": hi - lo, "at_bounds": "",
                    "note": "union of all intervals above; still subset resampling only (no parent-sample noise)"})
    ext = pd.concat([ext, pd.DataFrame(env)], ignore_index=True)

    gains = {}  # predicted gain from the smallest reported x (default 250) to the target
    t0, t1 = min(targets), float(f["target_n"])
    for tag in ("pl", "ll"):
        g = (boot[f"{tag}_pred_{t1:g}"] - boot[f"{tag}_pred_{t0:g}"]).dropna()
        gains[tag] = {"from_x": t0, "to_x": t1, "median": float(g.median()) if len(g) else np.nan,
                      "lo": float(g.quantile(lo_q)) if len(g) else np.nan,
                      "hi": float(g.quantile(hi_q)) if len(g) else np.nan,
                      "frac_positive": float((g > 0).mean()) if len(g) else np.nan}
    return {"x": x, "power_law": pl, "log_linear": ll,
            "pl_at_bounds": at_bounds(pl, cfg) if pl is not None else ["fit failed"],
            "boot": boot, "extrapolation": ext, "gains": gains, "targets": targets,
            "interval_cols": (f"boot_lo_{lo_q:.2f}", f"boot_hi_{hi_q:.2f}")}


def train_rows_of(x: float, cfg: dict) -> int:
    """Training rows behind a point at position x of the fit axis."""
    k = int(cfg["cv"]["n_splits"])
    return int(round(x if cfg["fit"]["x_axis"] == "effective" else x * (k - 1) / k))


def add_truth(ext: pd.DataFrame, gt: pd.DataFrame | None, cfg: dict, lo_c: str, hi_c: str) -> pd.DataFrame:
    """Synthetic only: attach the true AUC at the same training size and whether the interval covers it."""
    if gt is None:
        return ext
    truth = dict(zip(gt["train_rows"], gt["true_auc_mean"]))
    out = ext.assign(true_auc_synthetic=ext["x"].map(lambda v: truth.get(train_rows_of(v, cfg), np.nan)))
    out["covered"] = (out["true_auc_synthetic"] >= out[lo_c]) & (out["true_auc_synthetic"] <= out[hi_c])
    return out


# ---------------------------------------------------------------------------
# synthetic ground truth
# ---------------------------------------------------------------------------
def _gt_one(t: int, d: int, spec: dict, Xp, yp, train_pool, eval_idx, seed: int) -> dict:
    rng = np.random.default_rng([seed, t, d])
    idx = train_pool[stratified_subset(yp[train_pool], t, rng)]
    model = build_pipeline(spec).fit(Xp[idx], yp[idx])
    return {"train_rows": t, "draw": d,
            "auc": float(roc_auc_score(yp[eval_idx], predict_score(model, Xp[eval_idx])))}


def ground_truth(ds, cfg: dict, n_jobs: int) -> pd.DataFrame | None:
    """Synthetic only: train on t fresh rows from the labelled mock test set, score on
    n_eval other rows. t runs over the training sizes behind each CV point and targets."""
    g = cfg.get("ground_truth") or {}
    if ds.source != "synthetic" or not g.get("enabled", False):
        return None
    labels = pd.read_csv(paths.DATA_SYNTHETIC / "test_labels.csv")
    yp = (pd.DataFrame({"id": ds.ids_test}).merge(labels, on="id", how="left", validate="one_to_one")
          ["target"].to_numpy().astype(int))
    Xp = ds.X_test.to_numpy(float)
    rng = np.random.default_rng(int(g["seed"]))
    perm = rng.permutation(len(yp))
    eval_idx, train_pool = perm[: int(g["n_eval"])], perm[int(g["n_eval"]):]
    k = int(cfg["cv"]["n_splits"])
    targets = [float(t) for t in cfg["fit"].get("predict_at", [])] + [float(cfg["fit"]["target_n"])]
    ts = sorted({int(round(n * (k - 1) / k)) for n in cfg["sizes"]} | set(cfg["sizes"])
                | {train_rows_of(t, cfg) for t in targets})
    res = Parallel(n_jobs=n_jobs)(delayed(_gt_one)(t, d, cfg["pipeline"], Xp, yp, train_pool, eval_idx, int(g["seed"]))
                                  for t in ts for d in range(int(g["n_draws"])))
    df = pd.DataFrame(res)
    return (df.groupby("train_rows")["auc"].agg(["mean", "std", "min", "max", "count"]).reset_index()
            .rename(columns={"mean": "true_auc_mean", "std": "true_auc_std", "min": "true_auc_min",
                             "max": "true_auc_max", "count": "n_draws"})
            .assign(n_eval=int(g["n_eval"])))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style(ax, labelsize: float = 8, lw: float = 0.8):
    ax.grid(color=GRID, linewidth=lw)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(colors=INK2, labelsize=labelsize, width=lw)


def fig_curve(sub: pd.DataFrame, tab: pd.DataFrame, fit: dict, cfg: dict, gt: pd.DataFrame | None, label: str):
    """Report figure, drawn at its printed size (CURVE_SIZE), so the font sizes are the printed sizes.

    Every interval here resamples the subset scores of ONE sample. R942 found that the 90% band
    covers the replica truth in only 21% of samples, so the legend calls it "nominal".
    """
    rng = np.random.default_rng(0)
    f = cfg["fit"]
    target = float(f["target_n"])
    lo_c, hi_c = fit["interval_cols"]
    nominal = f"nominal {f['interval']:.0%}"
    fig, ax = plt.subplots(figsize=CURVE_SIZE)
    span = target - fit["x"].min()
    jit = span * 0.006
    h = []  # legend entries in reading order
    h.append(ax.scatter(sub["x"] + rng.uniform(-jit, jit, len(sub)), sub["cv_auc"], s=5, color=BLUE, alpha=0.35,
                        edgecolor="none", label=f"CV AUC of each subset ({cfg['n_subsets']} per size)"))
    h.append(ax.errorbar(tab["x"], tab["cv_mean"], yerr=tab["cv_std"], fmt="o", color=BLUE, ms=3.6, capsize=1.8,
                         elinewidth=1.0, capthick=1.0, mec="white", mew=0.6, zorder=4,
                         label="CV mean ± SD per size"))
    o = tab.dropna(subset=["oos_mean"])
    if len(o):  # out-of-subset: model trained on all n rows of the subset -> x = n on either axis
        h.append(ax.errorbar(o["size"] + span * 0.012, o["oos_mean"], yerr=o["oos_std"], fmt="s", color=ORANGE,
                             ms=3.0, capsize=1.8, elinewidth=0.9, capthick=0.9, mec="white", mew=0.5, zorder=4,
                             label="out-of-subset mean ± SD"))
    grid_obs = np.linspace(fit["x"].min(), fit["x"].max(), 100)
    grid_ext = np.linspace(fit["x"].max(), target * 1.04, 100)
    grid_all = np.concatenate([grid_obs, grid_ext])
    boot = fit["boot"]
    ok = boot[boot["pl_ok"]]
    if fit["power_law"] is not None:
        a, b, c = fit["power_law"]
        h.append(ax.plot(grid_obs, power_law(grid_obs, a, b, c), color=BLUE, linewidth=1.4,
                         label="power law fit (dashed: extrapolation)")[0])
        ax.plot(grid_ext, power_law(grid_ext, a, b, c), color=BLUE, linewidth=1.4, linestyle=(0, (4, 2)))
    if len(ok) > 10:
        curves = np.array([power_law(grid_all, a, b, c) for a, b, c in ok[["pl_a", "pl_b", "pl_c"]].to_numpy()])
        q = (1 - f["interval"]) / 2
        h.append(ax.fill_between(grid_all, np.quantile(curves, q, axis=0), np.quantile(curves, 1 - q, axis=0),
                                 color=BLUE, alpha=0.12, linewidth=0, label=f"{nominal} bootstrap band"))
    al, be = fit["log_linear"]
    h.append(ax.plot(grid_all, loglin(grid_all, al, be), color=AQUA, linewidth=1.2, linestyle=":",
                     label="log-linear alternative")[0])
    ext = fit["extrapolation"]
    r = ext[(ext["model"] == "power_law") & (ext["x"] == target)].iloc[0]
    if np.isfinite(r["boot_median"]):
        h.append(ax.errorbar([target], [r["boot_median"]],
                             yerr=[[r["boot_median"] - r[lo_c]], [r[hi_c] - r["boot_median"]]],
                             fmt="D", color=INK, ms=3.6, capsize=2.2, elinewidth=1.0, capthick=1.0, zorder=5,
                             label=f"AUC({target:g}): bootstrap median {r['boot_median']:.3f},\n"
                                   f"{nominal} interval [{r[lo_c]:.3f}, {r[hi_c]:.3f}]"))
    e = ext[(ext["model"] == "envelope") & (ext["x"] == target)]
    if len(e) and np.isfinite(e[lo_c].iloc[0]):  # union of all intervals, drawn left of the target
        h.append(ax.plot([target - span * 0.015] * 2, [e[lo_c].iloc[0], e[hi_c].iloc[0]], color=MUTED,
                         linewidth=2.2, solid_capstyle="butt", zorder=4,
                         label=f"envelope of all fits [{e[lo_c].iloc[0]:.3f}, {e[hi_c].iloc[0]:.3f}]\n"
                               "(subset noise only)")[0])
    sens = ext[(ext["x"] == target) & ext["model"].str.startswith("power_law_")]
    if len(sens):  # point estimates of the sensitivity fits, offset to the right of the target
        h.append(ax.scatter(target + span * 0.015 * np.arange(1, len(sens) + 1), sens["point"], s=11,
                            facecolor="white", edgecolor=INK2, linewidth=0.9, zorder=5,
                            label="sensitivity fits: " + ", ".join(m.replace("power_law_", "").replace("_", " ")
                                                                   for m in sens["model"])))
    if gt is not None:
        h.append(ax.plot(gt["train_rows"], gt["true_auc_mean"], color=MUTED, linewidth=1.0, marker="^", ms=3,
                         label=f"synthetic ground truth ({int(gt['n_eval'].iloc[0])}-row test)")[0])
    lo, hi = ax.get_ylim()
    if hi - lo < 0.01:  # e.g. the GMM: AUC 1.0 everywhere; no offset axis in units of 1e-6, data above the legend
        ax.set_ylim(hi - 0.01, hi + 0.002)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_xlabel(x_label(cfg), color=INK2, fontsize=FS["label"], labelpad=2)
    ax.set_ylabel("ROC AUC", color=INK2, fontsize=FS["label"], labelpad=2)
    title = textwrap.fill(f"Learning curve of {label.replace(',', ', ')}", width=72)
    ax.set_title(title, fontsize=FS["title"], color=INK, loc="left", pad=4)
    _style(ax, labelsize=FS["tick"], lw=0.6)
    ax.tick_params(length=2.5, pad=2)
    ax.legend(handles=h, frameon=False, fontsize=FS["legend"], loc="lower right", handlelength=1.6,
              handletextpad=0.5, labelspacing=0.3, borderaxespad=0.2, borderpad=0.2)
    fig.tight_layout(pad=0.2)
    return fig


def fig_boot(fit: dict, cfg: dict, gt: pd.DataFrame | None):
    target = float(cfg["fit"]["target_n"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    series = [(tag, name, color, fit["boot"][f"{tag}_pred_{target:g}"].dropna().to_numpy())
              for tag, name, color in (("pl", "power law", BLUE), ("ll", "log-linear", AQUA))]
    allv = np.concatenate([v for *_, v in series if len(v)])
    bins = np.histogram_bin_edges(allv, bins=60)  # shared bins so the two shapes compare fairly
    for tag, name, color, v in series:
        if len(v) == 0:
            continue
        ax.hist(v, bins=bins, color=color, alpha=0.45, label=f"{name}: median {np.median(v):.3f}")
        ax.axvline(np.median(v), color=color, linewidth=1.5)
    if allv.max() > 1:
        ax.axvline(1.0, color=MUTED, linewidth=1.0, label="AUC = 1 (impossible beyond)")
    if gt is not None:
        tr = train_rows_of(target, cfg)
        g = gt[gt["train_rows"] == tr]
        if len(g):
            ax.axvline(float(g["true_auc_mean"].iloc[0]), color=INK, linewidth=1.5, linestyle="--",
                       label=f"synthetic ground truth {float(g['true_auc_mean'].iloc[0]):.3f}")
    ax.set_xlabel(f"predicted AUC at x = {target:g} ({x_label(cfg)})", color=INK2, fontsize=8)
    ax.set_ylabel("bootstrap replicates", color=INK2, fontsize=9)
    ax.set_title(f"Bootstrap distribution of the extrapolated AUC ({len(fit['boot'])} replicates)",
                 fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def write_table(df: pd.DataFrame, out_dir: Path, name: str, md_cols: list[str] | None = None) -> list[Path]:
    """Full table as .csv; report-ready .md with only `md_cols` (all columns if None)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p, md_p = out_dir / f"{name}.csv", out_dir / f"{name}.md"
    df.to_csv(csv_p, index=False)
    md = df[[c for c in md_cols if c in df.columns]] if md_cols else df
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(md.to_markdown(index=False, floatfmt=".4f") + "\n")
    return [csv_p, md_p]


def ascii_table(df: pd.DataFrame, cols: list[str]) -> str:
    rows = df[cols].astype(object).values.tolist()  # object rows keep ints as ints
    return tabulate(rows, headers=cols, tablefmt="simple", floatfmt=".4f")


def resolve_config(p: Path) -> Path:
    """A relative --config is taken from the current folder, else from the project root."""
    if p.exists() or p.is_absolute() or not (paths.ROOT / p).exists():
        return p
    return paths.ROOT / p


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
    """Prefix of the report files (e2_, e2_qda_, ...) whose manifest names this run; None if none does."""
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


def archived_fit(sub: pd.DataFrame, boot: pd.DataFrame, ext: pd.DataFrame, summary: dict, cfg: dict) -> dict:
    """The parts of fit_and_bootstrap's result the figures use, rebuilt from the archived run
    (bootstrap.csv, extrapolation.csv, summary.json). Nothing is refitted."""
    f = cfg["fit"]
    sizes = sorted(int(n) for n in sub["size"].unique())
    lo_q, hi_q = (1 - f["interval"]) / 2, 1 - (1 - f["interval"]) / 2
    pl, ll = summary.get("power_law_params"), summary["log_linear_params"]
    return {"x": np.array([x_of(n, cfg) for n in sizes]),
            "power_law": None if pl is None else np.array([pl[p] for p in "abc"], dtype=float),
            "log_linear": np.array([ll["alpha"], ll["beta"]], dtype=float),
            "boot": boot.assign(pl_ok=boot["pl_ok"].astype(str).str.lower().eq("true")),
            "extrapolation": ext,
            "interval_cols": (f"boot_lo_{lo_q:.2f}", f"boot_hi_{hi_q:.2f}")}


def replot(ref: str, source: str, out_dir: Path | None) -> int:
    """Redraw the report figures of an archived E2 run from the tables saved in its run folder.

    No model is fitted and no curve is refitted; no run folder, registry row or report table is
    written. The figures replace <prefix>learning_curve.png and <prefix>bootstrap_target.png in
    reports/figures, where <prefix> comes from the reports/tables/e2*_manifest.json that names the
    run (e2_, e2_qda_, e2_gmm_), so file names and manifests stay valid. out_dir: write there instead.
    """
    rd = find_run_dir(ref, source)
    run_id = rd.name
    need = ["config.yaml", "meta.json", "summary.json", "subset_scores.csv", "learning_curve_table.csv",
            "bootstrap.csv", "extrapolation.csv"]
    missing = [n for n in need if not (rd / n).exists()]
    if missing:
        raise SystemExit(f"--replot: {run_id} is not a finished E2 run (missing {', '.join(missing)})")
    cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    if meta.get("data_source", source) != source:
        raise SystemExit(f"--replot: {run_id} was made on {meta['data_source']!r} data; pass --data {meta['data_source']}")
    if summary.get("status") != "done" or "log_linear_params" not in summary:
        raise SystemExit(f"--replot: {run_id} has no finished E2 summary (status {summary.get('status')!r})")
    sub = pd.read_csv(rd / "subset_scores.csv")
    tab = pd.read_csv(rd / "learning_curve_table.csv")
    fit = archived_fit(sub, pd.read_csv(rd / "bootstrap.csv"), pd.read_csv(rd / "extrapolation.csv"), summary, cfg)
    gt = pd.read_csv(rd / "ground_truth.csv") if (rd / "ground_truth.csv").exists() else None
    label = summary.get("model") or describe(cfg["pipeline"])["model"]
    prefix = report_prefix(run_id, source)
    if out_dir is not None:
        fdir, prefix = Path(out_dir), prefix or PREFIX
    elif prefix is None:
        raise SystemExit(f"--replot: no {PREFIX}*manifest.json in {paths.tables_dir(source)} names {run_id}, so its "
                         "figures are not report figures; use --replot-out DIR to draw them elsewhere")
    else:
        fdir = paths.figures_dir(source)
    written = save_report_figures({"learning_curve": fig_curve(sub, tab, fit, cfg, gt, label),
                                   "bootstrap_target": fig_boot(fit, cfg, gt)}, fdir, prefix, run_id)
    print(f"[e2] replot of {run_id} ({source} data) from its archived tables: nothing refitted, no run written")
    for p in written:
        print(f"  wrote {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a character the console cannot show
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", choices=paths.SOURCES, default=None,
                    help="raw (Kaggle data, real archive) or synthetic (sandbox) (default: $SSC_DATA or raw)")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--author", default=None)
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel subsets (joblib); -1 = all cores")
    ap.add_argument("--replot", metavar="RUN_ID", default=None,
                    help="only redraw the report figures of this archived run from its saved tables "
                         "(run_id or a unique prefix; no fitting, no new run, no registry row)")
    ap.add_argument("--replot-out", type=Path, default=None, metavar="DIR",
                    help="with --replot: write the figures to DIR instead of reports/figures")
    args = ap.parse_args(argv)
    args.data = paths.resolve_source(args.data)  # $SSC_DATA or raw; the code below uses args.data
    if args.replot_out is not None and not args.replot:
        ap.error("--replot-out needs --replot")
    if args.replot:
        if args.quick:
            ap.error("--replot redraws an archived run; --quick does not apply")
        return replot(args.replot, args.data, args.replot_out)

    t0 = time.perf_counter()
    cfg = load_config(resolve_config(args.config), args.quick)
    ds = ssc_data.load(args.data)
    X, y = ds.X.to_numpy(float), ds.y
    if max(cfg["sizes"]) > len(y):
        raise ValueError(f"largest size {max(cfg['sizes'])} > {len(y)} training rows")
    label = describe(cfg["pipeline"])["model"]
    archived = copy.deepcopy(cfg)
    archived["_runtime"] = {"data": args.data, "quick": args.quick, "n_jobs": args.n_jobs,
                            "config_file": str(args.config)}

    run = Run(cfg.get("name", "e2_learning_curve"), archived, kind="extension", author=args.author, source=args.data)
    with run:
        tasks = [(n, i) for n in cfg["sizes"] for i in range(cfg["n_subsets"])]
        print(f"[e2] run {run.run_id}: {len(tasks)} subsets on {args.data} data "
              f"({'quick' if args.quick else 'full'} settings, n_jobs={args.n_jobs})")
        res = Parallel(n_jobs=args.n_jobs)(delayed(score_subset)(n, i, cfg, X, y) for n, i in tasks)
        sub = pd.DataFrame([r["row"] for r in res])
        folds = pd.concat([r["folds"] for r in res], ignore_index=True)
        membership = np.stack([r["mask"] for r in res])
        tab = size_table(sub, cfg)
        t_cv = time.perf_counter() - t0

        print(f"[e2] fitting learning curve + {cfg['fit']['n_boot']} bootstrap refits ...")
        fit = fit_and_bootstrap(sub, cfg)
        gt = ground_truth(ds, cfg, args.n_jobs)
        ext = add_truth(fit["extrapolation"], gt, cfg, *fit["interval_cols"])

        run.save_table("subset_scores", sub)
        run.save_table("fold_scores_all", folds)
        run.save_array("subset_membership", membership)
        run.save_table("bootstrap", fit["boot"])
        run.save_table("learning_curve_table", tab)
        run.save_table("extrapolation", ext)
        if gt is not None:
            run.save_table("ground_truth", gt)

        f1 = fig_curve(sub, tab, fit, cfg, gt, label)
        f2 = fig_boot(fit, cfg, gt)
        run.save_figure("learning_curve", f1)
        run.save_figure("bootstrap_target", f2)

        if args.quick:  # never overwrite report figures with smoke-test numbers
            tdir = fdir = run.dir / "report_preview"
        else:
            tdir, fdir = paths.tables_dir(args.data), paths.figures_dir(args.data)
        fdir.mkdir(parents=True, exist_ok=True)
        lo_c, hi_c = fit["interval_cols"]
        written = (write_table(tab, tdir, f"{PREFIX}learning_curve_table", TABLE_MD_COLS)
                   + write_table(ext, tdir, f"{PREFIX}extrapolation",
                                 ["model", "x_axis", "x", "point", "boot_median", lo_c, hi_c, "at_bounds",
                                  "true_auc_synthetic", "covered", "note"]))
        if gt is not None:
            written += write_table(gt, tdir, f"{PREFIX}ground_truth")
        written += save_report_figures({"learning_curve": f1, "bootstrap_target": f2}, fdir, PREFIX, run.run_id)
        runtime = time.perf_counter() - t0
        with open(tdir / f"{PREFIX}manifest.json", "w", encoding="utf-8") as fh:
            json.dump({"run_id": run.run_id, "data_source": args.data, "quick": args.quick, "created_at": now_iso(),
                       "runtime_seconds": round(runtime, 1), "files": [str(p) for p in written]}, fh, indent=2)

        pl = fit["power_law"]
        run.finish(extra_summary={
            "question": "learning curve: mean and spread per size; expected AUC at target_n training rows",
            "model": label, "x_axis": cfg["fit"]["x_axis"], "target_n": cfg["fit"]["target_n"],
            "per_size": tab.to_dict(orient="records"),
            "power_law_params": None if pl is None else dict(zip("abc", map(float, pl))),
            "power_law_at_bounds": fit["pl_at_bounds"],
            "log_linear_params": dict(zip(("alpha", "beta"), map(float, fit["log_linear"]))),
            "extrapolation": ext.to_dict(orient="records"), "gain": fit["gains"],
            "ground_truth_synthetic": None if gt is None else gt.to_dict(orient="records"),
            "quick": args.quick, "n_jobs": args.n_jobs, "runtime_seconds": round(runtime, 1),
            "cv_seconds": round(t_cv, 1), "total_model_fits": int(len(folds) + sub["n_oos"].gt(0).sum()),
            "report_outputs": [str(p) for p in written],
        })

    # ---- console summary (ASCII only) ----
    lo_c, hi_c = fit["interval_cols"]
    print()
    print(f"E2 learning curve  run_id={run.run_id}  data={args.data}  quick={args.quick}")
    print(f"model: {label}  inner CV: {cfg['cv']['n_splits']}-fold x {cfg['cv']['n_repeats']}  "
          f"x axis: {cfg['fit']['x_axis']} ({x_label(cfg)})")
    print("\nPer size (CV inside subset; out-of-subset = fit on subset, score on the other rows):")
    print(ascii_table(tab, ["size", "x", "n_subsets", "cv_mean", "cv_std", "cv_min", "cv_max",
                            "oos_mean", "oos_std", "n_oos_rows", "frac_constant_folds"]))
    full = tab[tab["spread_source"].str.startswith("CV seed")]
    if len(full):
        print(f"  note: size {int(full['size'].iloc[0])} = full data; its spread is CV-seed (evaluation) noise only")
    print("\nExtrapolation (bootstrap over subset scores):")
    cols = ["model", "x", "point", "boot_median", lo_c, hi_c, "n_boot_failed", "at_bounds"]
    print(ascii_table(ext, cols + (["true_auc_synthetic", "covered"] if "covered" in ext else [])))
    print("  CAVEAT: every interval above resamples only the subset scores. All subsets come from the same\n"
          "  250 rows, so the noise of that parent sample (another 250 rows from the same population) is\n"
          "  missing, and the SEM weighting pins the curve to the n = 250 point. Treat even the 'envelope'\n"
          "  row as a lower bound on the uncertainty, at x = 250 as well as at the target"
          + (" (on synthetic data compare 'covered')." if "covered" in ext else "."))
    if pl is not None:
        print(f"power law: a={pl[0]:.4f} b={pl[1]:.4f} c={pl[2]:.4f}"
              + (f"  (at bound: {','.join(fit['pl_at_bounds'])})" if fit["pl_at_bounds"] else ""))
    for tag, g in fit["gains"].items():
        print(f"gain {g['from_x']:g} -> {g['to_x']:g} ({'power law' if tag == 'pl' else 'log-linear'}): "
              f"median {g['median']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}], P(gain > 0) = {g['frac_positive']:.2f}")
    if gt is not None:
        print("\nSynthetic ground truth (train on fresh rows, score on held-out mock test rows):")
        print(ascii_table(gt, ["train_rows", "true_auc_mean", "true_auc_std", "n_draws"]))
    print(f"\nrun folder: {run.dir}")
    print(f"report outputs: {tdir}" + ("" if tdir == fdir else f" | {fdir}"))
    print(f"runtime: {runtime:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
