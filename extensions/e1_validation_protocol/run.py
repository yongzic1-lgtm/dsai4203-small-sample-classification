"""E1 Validation protocol design: how much does each validation scheme's AUC move
when only the split seed changes, and which scheme can resolve a 0.01 AUC difference?

For every scheme in config.yaml (hold-out, k-fold for several k, repeated k-fold,
and the non-shuffled k-fold caveat) the scheme is executed `n_split_seeds` times.
Between executions only the split seed changes; the model seed stays fixed. Each
execution yields one "estimate" (what a practitioner would report from it). Two
models A and B are scored on the identical splits of every execution, so the
paired difference A - B and its spread can be compared with a large repeated-CV
reference.

Outputs
-------
run folder (ssc.tracking.Run, kind="extension"):
    estimates.csv            one row per (scheme, execution, model)
    paired_estimates.csv     one row per (scheme, execution): A - B on identical splits
    fold_scores_all.csv      every fold AUC behind the estimates
    reference_fold_scores.csv
    scheme_spread.csv, paired_difference.csv, *.png   copies of the report outputs
report outputs (ssc.paths.tables_dir / figures_dir; in --quick mode <run>/report_preview/):
    e1_scheme_spread.{csv,md}      spread of single-model estimates per scheme
    e1_paired_difference.{csv,md}  spread / sign agreement / MDD of A - B per scheme
    e1_estimates_by_scheme.png     box + strip plot of estimates (A) and of A - B
    e1_diff_std_by_scheme.png      std of the paired difference with the 0.01 line
    e1_manifest.json               which run produced these files

Usage (from the project root):
    .venv\\Scripts\\python.exe extensions\\e1_validation_protocol\\run.py --data synthetic --quick
    .venv\\Scripts\\python.exe extensions\\e1_validation_protocol\\run.py --n-jobs 8
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
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tabulate import tabulate  # noqa: E402

from ssc import data as ssc_data  # noqa: E402
from ssc import paths  # noqa: E402
from ssc.models import build_pipeline, describe, predict_score  # noqa: E402
from ssc.tracking import Run, now_iso  # noqa: E402
from ssc.validation import Scheme, cross_validate, make_splits, paired_compare, split_signature  # noqa: E402

HERE = Path(__file__).resolve().parent
PREFIX = "e1_"
SCHEME_FIELDS = ("kind", "n_splits", "n_repeats", "test_size", "shuffle", "stratify")
# columns of the report-ready .md tables (the .csv keeps every column)
SPREAD_MD_COLS = ["scheme", "label", "model", "n_estimates", "mean", "std", "iqr", "min", "max",
                  "mean_within_rep_se", "fits_per_estimate", "n_unique_splits"]
PAIRED_MD_COLS = ["scheme", "label", "mean_diff", "std_diff", "std_diff_unpaired", "mdd_2sd", "mdd_lo", "mdd_hi",
                  "sign_agree_ref", "p_correct_at_target", "frac_nb_p_lt_0.05", "fits_per_decision",
                  "n_unique_splits", "note"]
MDD_CI = 0.95  # chi-square interval of std_diff (hence MDD) from n_split_seeds executions

# Chart tokens (reference palette of the dataviz guide; light surface).
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(path: Path, quick: bool) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["schemes"] = [s for s in cfg["schemes"] if s.get("enabled", True)]
    if quick:
        q = cfg.get("quick") or {}
        cfg["n_split_seeds"] = int(q.get("n_split_seeds", cfg["n_split_seeds"]))
        if "max_repeats" in q:
            for s in cfg["schemes"]:
                s["n_repeats"] = min(int(s.get("n_repeats", 1)), int(q["max_repeats"]))
        if "reference_repeats" in q:
            cfg["reference"]["n_repeats"] = int(q["reference_repeats"])
    max_r = max(int(s.get("n_repeats", 1)) for s in cfg["schemes"])
    if cfg["seed_stride"] < max_r:
        raise ValueError(f"seed_stride={cfg['seed_stride']} < largest n_repeats={max_r}: executions would share splits")
    lo = cfg["seed_base"]
    hi = cfg["seed_base"] + (cfg["n_split_seeds"] - 1) * cfg["seed_stride"] + max_r
    ref = cfg["reference"]
    if ref["seed"] < hi and ref["seed"] + ref["n_repeats"] > lo:
        raise ValueError("reference seeds overlap the execution seeds; move reference.seed")
    return cfg


def scheme_for(entry: dict, seed: int) -> Scheme:
    return Scheme(**{k: entry[k] for k in SCHEME_FIELDS if k in entry}, seed=int(seed))


def fits_per_estimate(entry: dict) -> int:
    reps = int(entry.get("n_repeats", 1))
    return reps if entry["kind"] == "holdout" else reps * int(entry["n_splits"])


# ---------------------------------------------------------------------------
# one execution = one estimate per model, on identical splits
# ---------------------------------------------------------------------------
def run_execution(entry: dict, s: int, split_seed: int, specs: dict, X: np.ndarray, y: np.ndarray) -> dict:
    scheme = scheme_for(entry, split_seed)
    splits = make_splits(y, scheme)
    sig = split_signature(splits)
    est_rows, fold_rows, fold_scores = [], [], {}
    for key, spec in specs.items():
        cv = cross_validate(build_pipeline(spec), X, y, splits, scheme, test_pred="none", n_jobs=1)
        fs = cv.fold_scores
        fold_scores[key] = fs
        pooled = (float(np.mean([roc_auc_score(y, row) for row in cv.oof]))
                  if scheme.kind == "kfold" else np.nan)
        n_folds = len(fs)
        rep = fs.groupby("repeat")["auc"].mean()
        est_rows.append({
            "scheme": entry["name"], "label": scheme.label(), "execution": s, "split_seed": split_seed,
            "model": key, "estimate": float(fs["auc"].mean()), "pooled_oof_auc": pooled,
            "fold_std": float(fs["auc"].std(ddof=1)) if n_folds > 1 else np.nan,
            # error bars a practitioner could compute from this ONE execution:
            "within_fold_se": float(fs["auc"].std(ddof=1) / np.sqrt(n_folds)) if n_folds > 1 else np.nan,
            "within_rep_se": float(rep.std(ddof=1) / np.sqrt(len(rep))) if len(rep) > 1 else np.nan,
            "n_fits": n_folds, "split_signature": sig,
        })
        fold_rows.append(fs.assign(scheme=entry["name"], execution=s, split_seed=split_seed, model=key))
    a, b = fold_scores["A"], fold_scores["B"]
    pc = paired_compare(a, b) if len(a) > 1 else {}
    paired = {
        "scheme": entry["name"], "label": scheme.label(), "execution": s, "split_seed": split_seed,
        "diff": float(a["auc"].mean() - b["auc"].mean()),
        "nb_p": pc.get("corrected_p", np.nan), "nb_t": pc.get("corrected_t", np.nan),
        "win_rate_fold": pc.get("win_rate_fold", float(a["auc"].iloc[0] > b["auc"].iloc[0])),
        "within_rep_se": (pc["std_diff_repeat"] / np.sqrt(scheme.n_repeats)
                          if pc and scheme.n_repeats > 1 else np.nan),
        "split_signature": sig,
    }
    return {"estimates": est_rows, "folds": fold_rows, "paired": paired}


def run_reference(cfg: dict, specs: dict, X: np.ndarray, y: np.ndarray, n_jobs: int) -> dict:
    ref = cfg["reference"]
    scheme = Scheme(kind=ref.get("kind", "kfold"), n_splits=int(ref["n_splits"]),
                    n_repeats=int(ref["n_repeats"]), seed=int(ref["seed"]))
    splits = make_splits(y, scheme)
    fs = {k: cross_validate(build_pipeline(spec), X, y, splits, scheme, test_pred="none", n_jobs=n_jobs).fold_scores
          for k, spec in specs.items()}
    pc = paired_compare(fs["A"], fs["B"])
    rep_diff = (fs["A"].groupby("repeat")["auc"].mean() - fs["B"].groupby("repeat")["auc"].mean()).to_numpy()
    se = float(rep_diff.std(ddof=1) / np.sqrt(len(rep_diff))) if len(rep_diff) > 1 else np.nan
    diff = float(fs["A"]["auc"].mean() - fs["B"]["auc"].mean())
    folds = pd.concat([f.assign(model=k) for k, f in fs.items()], ignore_index=True)
    return {
        "label": scheme.label(), "seed": scheme.seed, "n_fits_per_model": len(splits),
        "auc_A": float(fs["A"]["auc"].mean()), "auc_B": float(fs["B"]["auc"].mean()),
        "diff": diff, "diff_se": se, "sign": int(np.sign(diff)),
        "sign_reliable": bool(np.isfinite(se) and abs(diff) > 2 * se),
        "nb_p": pc["corrected_p"], "split_signature": split_signature(splits), "folds": folds,
    }


def ground_truth(ds, specs: dict) -> dict | None:
    """Synthetic data only: fit on all 250 rows, score on the 19,750 hidden test labels."""
    if ds.source != "synthetic":
        return None
    labels = pd.read_csv(paths.DATA_SYNTHETIC / "test_labels.csv")
    yt = (pd.DataFrame({"id": ds.ids_test}).merge(labels, on="id", how="left", validate="one_to_one")
          ["target"].to_numpy())
    X, Xt = ds.X.to_numpy(float), ds.X_test.to_numpy(float)
    out = {k: float(roc_auc_score(yt, predict_score(build_pipeline(spec).fit(X, ds.y), Xt)))
           for k, spec in specs.items()}
    out["diff"] = out["A"] - out["B"]
    out["n_test"] = int(len(yt))
    return out


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def _iqr(x: pd.Series) -> float:
    return float(x.quantile(0.75) - x.quantile(0.25))


def spread_table(est: pd.DataFrame, cfg: dict, ref: dict) -> pd.DataFrame:
    rows = []
    for entry in cfg["schemes"]:
        for key in ("A", "B"):
            e = est[(est["scheme"] == entry["name"]) & (est["model"] == key)]["estimate"]
            sig = est[est["scheme"] == entry["name"]]["split_signature"]
            sel = est[(est["scheme"] == entry["name"]) & (est["model"] == key)]
            rows.append({
                "scheme": entry["name"], "label": est[est["scheme"] == entry["name"]]["label"].iloc[0],
                "model": key, "n_estimates": len(e),
                "mean": e.mean(), "std": e.std(ddof=1), "iqr": _iqr(e), "min": e.min(), "max": e.max(),
                "bias_vs_ref": e.mean() - ref[f"auc_{key}"],
                "mean_within_fold_se": sel["within_fold_se"].mean() if sel["within_fold_se"].notna().any() else np.nan,
                "mean_within_rep_se": sel["within_rep_se"].mean() if sel["within_rep_se"].notna().any() else np.nan,
                "fits_per_estimate": fits_per_estimate(entry),
                "n_unique_splits": int(sig.nunique()),
            })
    for key in ("A", "B"):
        rows.append({"scheme": "reference", "label": ref["label"], "model": key, "n_estimates": 1,
                     "mean": ref[f"auc_{key}"], "std": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan,
                     "bias_vs_ref": 0.0, "mean_within_fold_se": np.nan, "mean_within_rep_se": np.nan,
                     "fits_per_estimate": ref["n_fits_per_model"], "n_unique_splits": 1})
    return pd.DataFrame(rows)


def sd_interval(sd: float, n: int, level: float = MDD_CI) -> tuple[float, float]:
    """Chi-square confidence interval of a standard deviation estimated from n draws."""
    if not np.isfinite(sd) or n < 2:
        return np.nan, np.nan
    a, df = 1 - level, n - 1
    return (sd * np.sqrt(df / stats.chi2.ppf(1 - a / 2, df)), sd * np.sqrt(df / stats.chi2.ppf(a / 2, df)))


def paired_table(pair: pd.DataFrame, est: pd.DataFrame, cfg: dict, ref: dict) -> pd.DataFrame:
    target = float(cfg["target_difference"])
    rows = []
    for entry in cfg["schemes"]:
        d = pair[pair["scheme"] == entry["name"]]
        x = d["diff"]
        sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        sd_lo, sd_hi = sd_interval(sd, len(x))
        degenerate = int(d["split_signature"].nunique()) == 1 and len(d) > 1
        if degenerate:
            note = "identical split for every seed (shuffle=False): zero spread is not precision"
        elif not np.isfinite(sd):
            note = "too few executions"
        elif 2 * sd_hi < target:  # the whole CI of the MDD must be on one side of the target
            note = "MDD < target"
        elif 2 * sd_lo > target:
            note = "MDD > target"
        else:
            note = f"inconclusive ({MDD_CI:.0%} CI of MDD contains the target)"
        nb = d["nb_p"]
        # what pairing buys: std of A - B if A and B had been scored on independent splits
        w = est[est["scheme"] == entry["name"]].pivot(index="execution", columns="model", values="estimate")
        sa, sb = float(w["A"].std(ddof=1)), float(w["B"].std(ddof=1))
        rho = float(np.corrcoef(w["A"], w["B"])[0, 1]) if len(w) > 2 and sa > 0 and sb > 0 else np.nan
        rows.append({
            "scheme": entry["name"], "label": d["label"].iloc[0], "n_estimates": len(d),
            "mean_diff": x.mean(), "std_diff": sd, "iqr_diff": _iqr(x), "min_diff": x.min(), "max_diff": x.max(),
            "mdd_2sd": 2 * sd if np.isfinite(sd) else np.nan,
            "mdd_lo": 2 * sd_lo if not degenerate else np.nan,
            "mdd_hi": 2 * sd_hi if not degenerate else np.nan,
            "sign_agree_ref": float((np.sign(x) == ref["sign"]).mean()),
            "p_correct_at_target": (float(stats.norm.cdf(target / sd)) if np.isfinite(sd) and sd > 0 and not degenerate
                                    else np.nan),
            "frac_nb_p_lt_0.05": float((nb < 0.05).mean()) if nb.notna().any() else np.nan,
            "mean_within_rep_se": d["within_rep_se"].mean() if d["within_rep_se"].notna().any() else np.nan,
            "std_diff_unpaired": float(np.hypot(sa, sb)) if np.isfinite(sa) else np.nan,
            "corr_AB": rho,
            "fits_per_decision": 2 * fits_per_estimate(entry),
            "n_unique_splits": int(d["split_signature"].nunique()),
            "note": note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=8)


def _xlabels(cfg: dict, est: pd.DataFrame | None = None) -> list[str]:
    """Compact tick labels: scheme, then cost in model fits per estimate."""
    out = []
    for entry in cfg["schemes"]:
        reps = int(entry.get("n_repeats", 1))
        if entry["kind"] == "holdout":
            lab = f"hold-out {int(round(100 * entry.get('test_size', 0.2)))}%"
        else:
            lab = f"{entry['n_splits']}-fold" + (f" x{reps}" if reps > 1 else "")
            if not entry.get("shuffle", True):
                lab += "\nno shuffle"
        n = fits_per_estimate(entry)
        out.append(f"{lab}\n{n} fit{'s' if n > 1 else ''}")
    return out


def _box_strip(ax, groups: list[np.ndarray], colors: list[str], rng: np.random.Generator):
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": MUTED},
                    capprops={"color": MUTED}, boxprops={"linewidth": 0})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.18)
    for i, (g, c) in enumerate(zip(groups, colors), start=1):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(g)), g, s=14, color=c, edgecolor="white",
                   linewidth=0.6, zorder=3)


def fig_estimates(est: pd.DataFrame, pair: pd.DataFrame, cfg: dict, ref: dict, labels: dict):
    rng = np.random.default_rng(0)
    names = [e["name"] for e in cfg["schemes"]]
    colors = [GRAY if not e.get("shuffle", True) else BLUE for e in cfg["schemes"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ga = [est[(est["scheme"] == n) & (est["model"] == "A")]["estimate"].to_numpy() for n in names]
    _box_strip(axes[0], ga, colors, rng)
    axes[0].axhline(ref["auc_A"], color=ORANGE, linewidth=1.5, linestyle="--",
                    label=f"reference {ref['label']}: {ref['auc_A']:.3f}")
    axes[0].set_title(f"(a) AUC estimate of model A ({labels['A']})", fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("AUC estimate (one per split seed)", color=INK2, fontsize=9)
    gd = [pair[pair["scheme"] == n]["diff"].to_numpy() for n in names]
    _box_strip(axes[1], gd, colors, rng)
    axes[1].axhline(ref["diff"], color=ORANGE, linewidth=1.5, linestyle="--",
                    label=f"reference A - B: {ref['diff']:+.3f}")
    axes[1].axhline(0.0, color=MUTED, linewidth=1.0)
    axes[1].set_title(f"(b) paired difference A - B ({labels['A']} vs {labels['B']})",
                      fontsize=10, color=INK, loc="left")
    axes[1].set_ylabel("AUC(A) - AUC(B), identical splits", color=INK2, fontsize=9)
    xl = _xlabels(cfg, est)
    for ax in axes:
        _style(ax)
        ax.set_xticks(range(1, len(names) + 1), xl, fontsize=7.5)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Spread of validation estimates over {cfg['n_split_seeds']} split seeds "
                 "(model seed fixed; gray = shuffle=False)", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def fig_diff_std(ptab: pd.DataFrame, cfg: dict, est: pd.DataFrame):
    target = float(cfg["target_difference"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sd = ptab["std_diff"].fillna(0).to_numpy()
    lo = (ptab["mdd_lo"] / 2).fillna(pd.Series(sd)).to_numpy()
    hi = (ptab["mdd_hi"] / 2).fillna(pd.Series(sd)).to_numpy()
    colors = [GRAY if "identical" in n else BLUE for n in ptab["note"]]
    x = np.arange(len(ptab))
    ax.bar(x, sd, width=0.55, color=colors, zorder=2)
    ax.errorbar(x, sd, yerr=[sd - lo, hi - sd], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3,
                label=f"{MDD_CI:.0%} CI of the std ({cfg['n_split_seeds']} split seeds)")
    for xi, v, top, note in zip(x, sd, hi, ptab["note"]):
        ax.text(xi, top + target * 0.03, "0 (degenerate)" if "identical" in note else f"{v:.4f}",
                ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.axhline(target, color=ORANGE, linewidth=1.5, label=f"difference to detect = {target:g}")
    ax.axhline(target / 2, color=ORANGE, linewidth=1.2, linestyle="--",
               label=f"std needed for MDD = 2 x std <= {target:g}")
    ax.set_xticks(x, _xlabels(cfg, est), fontsize=7.5)
    ax.set_ylabel("std of paired-difference estimate", color=INK2, fontsize=9)
    ax.set_title("Split-seed noise of AUC(A) - AUC(B) by scheme", fontsize=10, color=INK, loc="left")
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
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel executions (joblib); -1 = all cores")
    args = ap.parse_args(argv)
    args.data = source = paths.resolve_source(args.data)  # the code below uses args.data

    t0 = time.perf_counter()
    cfg = load_config(resolve_config(args.config), args.quick)
    ds = ssc_data.load(source)
    X, y = ds.X.to_numpy(float), ds.y
    specs = {k: v["pipeline"] for k, v in cfg["models"].items()}
    labels = {k: v.get("label", describe(v["pipeline"])["model"]) for k, v in cfg["models"].items()}
    if set(specs) != {"A", "B"}:
        raise ValueError("config.models must define exactly A and B")
    archived = copy.deepcopy(cfg)
    archived["_runtime"] = {"data": args.data, "quick": args.quick, "n_jobs": args.n_jobs,
                            "config_file": str(args.config)}

    run = Run(cfg.get("name", "e1_validation_protocol"), archived, kind="extension",
              author=args.author, source=args.data)
    with run:
        tasks = [(entry, s, cfg["seed_base"] + s * cfg["seed_stride"])
                 for entry in cfg["schemes"] for s in range(cfg["n_split_seeds"])]
        print(f"[e1] run {run.run_id}: {len(tasks)} executions on {args.data} data "
              f"({'quick' if args.quick else 'full'} settings, n_jobs={args.n_jobs})")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_execution)(entry, s, seed, specs, X, y) for entry, s, seed in tasks)
        est = pd.DataFrame([r for res in results for r in res["estimates"]])
        pair = pd.DataFrame([res["paired"] for res in results])
        folds = pd.concat([f for res in results for f in res["folds"]], ignore_index=True)
        t_exec = time.perf_counter() - t0

        print(f"[e1] reference {cfg['reference']['n_splits']}fold x {cfg['reference']['n_repeats']} ...")
        ref = run_reference(cfg, specs, X, y, args.n_jobs)
        gt = ground_truth(ds, specs) if cfg.get("ground_truth", True) else None

        stab = spread_table(est, cfg, ref)
        ptab = paired_table(pair, est, cfg, ref)

        # raw per-repetition results live in the run folder
        run.save_table("estimates", est)
        run.save_table("paired_estimates", pair)
        run.save_table("fold_scores_all", folds)
        run.save_table("reference_fold_scores", ref["folds"])
        run.save_table("scheme_spread", stab)
        run.save_table("paired_difference", ptab)

        f1 = fig_estimates(est, pair, cfg, ref, labels)
        f2 = fig_diff_std(ptab, cfg, est)
        run.save_figure("estimates_by_scheme", f1)
        run.save_figure("diff_std_by_scheme", f2)

        if args.quick:  # never overwrite report figures with smoke-test numbers
            tdir = fdir = run.dir / "report_preview"
        else:
            tdir, fdir = paths.tables_dir(args.data), paths.figures_dir(args.data)
        fdir.mkdir(parents=True, exist_ok=True)
        written = (write_table(stab, tdir, f"{PREFIX}scheme_spread", SPREAD_MD_COLS)
                   + write_table(ptab, tdir, f"{PREFIX}paired_difference", PAIRED_MD_COLS))
        for name, fig in ((f"{PREFIX}estimates_by_scheme", f1), (f"{PREFIX}diff_std_by_scheme", f2)):
            p = fdir / f"{name}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            written.append(p)
            plt.close(fig)
        runtime = time.perf_counter() - t0
        manifest = {"run_id": run.run_id, "data_source": args.data, "quick": args.quick,
                    "created_at": now_iso(), "runtime_seconds": round(runtime, 1),
                    "files": [str(p) for p in written]}
        with open(tdir / f"{PREFIX}manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        ref_public = {k: v for k, v in ref.items() if k != "folds"}
        run.finish(extra_summary={
            "question": "which scheme can resolve a 0.01 AUC difference between two models?",
            "models": labels, "reference": ref_public, "ground_truth_synthetic": gt,
            "paired_difference": ptab.to_dict(orient="records"),
            "scheme_spread": stab.to_dict(orient="records"),
            "n_split_seeds": cfg["n_split_seeds"], "quick": args.quick, "n_jobs": args.n_jobs,
            "runtime_seconds": round(runtime, 1), "executions_seconds": round(t_exec, 1),
            "total_model_fits": int(est["n_fits"].sum() + 2 * ref["n_fits_per_model"]),
            "report_outputs": [str(p) for p in written],
        })

    # ---- console summary (ASCII only) ----
    print()
    print(f"E1 validation protocol  run_id={run.run_id}  data={args.data}  quick={args.quick}")
    print(f"models: A = {labels['A']} | B = {labels['B']}  (model seeds fixed)")
    print(f"reference {ref['label']}: A={ref['auc_A']:.4f}  B={ref['auc_B']:.4f}  "
          f"A-B={ref['diff']:+.4f} (SE {ref['diff_se']:.4f}, sign {'reliable' if ref['sign_reliable'] else 'NOT reliable'})")
    if gt:
        print(f"synthetic ground truth (fit on 250, scored on {gt['n_test']} hidden labels): "
              f"A={gt['A']:.4f}  B={gt['B']:.4f}  A-B={gt['diff']:+.4f}")
    print("\nSpread of model-A estimates across split seeds:")
    print(ascii_table(stab[stab["model"] == "A"],
                      ["scheme", "label", "n_estimates", "mean", "std", "iqr", "min", "max",
                       "mean_within_rep_se", "fits_per_estimate", "n_unique_splits"]))
    print(f"\nPaired difference A - B (target difference {cfg['target_difference']}):")
    print(ascii_table(ptab, ["scheme", "mean_diff", "std_diff", "mdd_2sd", "mdd_lo", "mdd_hi", "sign_agree_ref",
                             "p_correct_at_target", "frac_nb_p_lt_0.05", "mean_within_rep_se",
                             "fits_per_decision"]))
    print(f"  (mdd_lo/mdd_hi: {MDD_CI:.0%} chi-square CI of the MDD from {cfg['n_split_seeds']} split seeds)")
    for _, r in ptab.iterrows():
        print(f"  {r['scheme']}: {r['note']}")
    print(f"\nrun folder: {run.dir}")
    print(f"report outputs: {tdir}" + ("" if tdir == fdir else f" | {fdir}"))
    print(f"runtime: {runtime:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
