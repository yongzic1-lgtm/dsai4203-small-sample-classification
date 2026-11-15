"""Run one experiment from a YAML config and archive it.

    .venv\\Scripts\\python.exe scripts\\run_experiment.py --config configs\\baseline_l2_logreg.yaml
    .venv\\Scripts\\python.exe scripts\\run_experiment.py --config configs\\baseline_l1_logreg.yaml ^
        --data synthetic --override cv.n_repeats=2 --override pipeline.model.params.C=0.5

Steps: load the config (format: configs/_template.yaml), apply --override
key=value items (dotted keys, values parsed as YAML), fill a missing cv block
with project.yaml official_cv, validate every key, pre-flight fit the pipeline
once (catches bad parameters before anything is archived), run repeated CV with
ssc.validation.run_cv (model seed fixed, only the split changes between repeats),
predict the test set (test_pred: fold_mean | refit | none) and archive everything
as an ssc.tracking.Run (kind "experiment"): experiments/<run_id>/ + one row in
registry/experiments.csv.

Only runs on the official CV share a split signature, so only they can be
compared (scripts/compare_runs.py) or ranked against each other. A run on any
other scheme is archived with "official_cv": false and a loud warning.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from pathlib import Path

import yaml
from sklearn.base import clone

import _common as C
from ssc import data, paths
from ssc.models import ESTIMATORS, SCALERS, build_pipeline, describe, predict_score
from ssc.tracking import Run
from ssc.validation import Scheme, make_splits, run_cv, split_signature

TOP_KEYS = ("name", "author", "hypothesis", "pipeline", "cv", "test_pred")
PIPELINE_KEYS = ("scaler", "selector", "model", "random_state")
MODEL_KEYS = ("name", "params")
SELECTORS = ("corr_topk", "stability_topk", "fixed_columns")
CV_KEYS = tuple(Scheme().to_dict()) + ("scheme",)
TEST_PRED = ("fold_mean", "refit", "none")


def _unknown(where: str, d: dict, allowed) -> list[str]:
    bad = sorted(set(d) - set(allowed))
    return [f"unknown key(s) in {where}: {bad}; allowed: {list(allowed)}"] if bad else []


def validate(cfg: dict) -> list[str]:
    """Structural check of the config; parameter values are checked by the pre-flight fit."""
    p = _unknown("config", cfg, TOP_KEYS)
    pipe = cfg.get("pipeline")
    if not isinstance(pipe, dict):
        return p + ["config needs a 'pipeline' mapping (see configs/_template.yaml)"]
    p += _unknown("pipeline", pipe, PIPELINE_KEYS)
    scaler = pipe.get("scaler", "standard")
    if scaler not in (None, "none", *SCALERS):
        p.append(f"pipeline.scaler {scaler!r} not in {['none', *SCALERS]}")
    model = pipe.get("model")
    if not isinstance(model, dict) or "name" not in model:
        p.append("pipeline.model needs a 'name' (and optional 'params')")
    else:
        p += _unknown("pipeline.model", model, MODEL_KEYS)
        if model["name"] not in ESTIMATORS:
            p.append(f"pipeline.model.name {model['name']!r} unknown; available: {sorted(ESTIMATORS)}")
        if not isinstance(model.get("params") or {}, dict):
            p.append("pipeline.model.params must be a mapping")
    sel = pipe.get("selector")
    if sel is not None:
        if not isinstance(sel, dict) or sel.get("name") not in SELECTORS:
            p.append(f"pipeline.selector needs name in {list(SELECTORS)}")
    if not isinstance(pipe.get("random_state", 0), int):
        p.append("pipeline.random_state must be an integer")
    cv = cfg.get("cv")
    if not isinstance(cv, dict):
        p.append("cv must be a mapping (or omitted to use project.yaml official_cv)")
    else:
        p += _unknown("cv", cv, CV_KEYS)
        if cv.get("kind", cv.get("scheme", "kfold")) not in ("kfold", "holdout"):
            p.append("cv.kind must be kfold or holdout")
        for k, lo in (("n_splits", 2), ("n_repeats", 1)):
            if k in cv and (not isinstance(cv[k], int) or cv[k] < lo):
                p.append(f"cv.{k} must be an integer >= {lo}")
    if cfg.get("test_pred", "fold_mean") not in TEST_PRED:
        p.append(f"test_pred {cfg.get('test_pred')!r} not in {list(TEST_PRED)}")
    return p


def preflight(pipe, ds) -> list[str]:
    """Fit once on all rows (result discarded) to fail fast and surface warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = clone(pipe).fit(ds.X.to_numpy(), ds.y)
            predict_score(model, ds.X.to_numpy()[:5])
        except Exception as e:  # noqa: BLE001 - report any estimator error cleanly
            C.die(f"pipeline failed in the pre-flight fit: {type(e).__name__}: {e}", 2)
    seen = []
    for w in caught:
        msg = f"{w.category.__name__}: {str(w.message).splitlines()[0][:200]}"
        if msg not in seen:
            seen.append(msg)
    return seen


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="experiment YAML (see configs/_template.yaml)")
    C.add_data_arg(ap)
    ap.add_argument("--author", help="member id (default: config author, $SSC_AUTHOR, project.yaml default_author)")
    ap.add_argument("--name", help="run name (default: config name, else the file name)")
    ap.add_argument("--n-jobs", type=int, default=4, help="parallel fits (default 4; -1 = all cores)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config key, repeatable, e.g. cv.n_repeats=2 or pipeline.model.params.C=0.5")
    ap.add_argument("--allow-deprecated", action="store_true",
                    help="run even if the pre-flight fit raises FutureWarning/DeprecationWarning")
    args = ap.parse_args()
    source = paths.resolve_source(args.data)

    # ---- config ------------------------------------------------------------
    args.config = C.resolve_path(args.config, "config")
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = C.coerce_numbers(yaml.safe_load(f) or {})
    except yaml.YAMLError as e:
        C.die(f"cannot parse {args.config}: {e}", 2)
    if not isinstance(cfg, dict):
        C.die(f"{args.config} must contain a YAML mapping", 2)
    official = C.official_cv_dict()
    cv_from = "config"
    if cfg.get("cv") is None:
        cfg["cv"] = copy.deepcopy(official)
        cv_from = "project.yaml official_cv"
    overrides = []
    for item in args.override:
        try:
            overrides.append("%s=%r" % C.apply_override(cfg, item))
        except ValueError as e:
            C.die(str(e), 2)
    if overrides and any(o.startswith("cv.") or o.startswith("cv=") for o in overrides):
        cv_from += " + overrides"
    cfg["name"] = args.name or cfg.get("name") or args.config.stem
    if args.author:
        cfg["author"] = args.author
    cfg.setdefault("test_pred", "fold_mean")
    problems = validate(cfg)
    if problems:
        print(f"Config {args.config} is invalid:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    try:
        scheme = Scheme.from_dict(cfg["cv"])
        pipe = build_pipeline(cfg["pipeline"], source=source)  # transductive models read this source's features
    except (TypeError, ValueError, KeyError) as e:
        C.die(f"cannot build the pipeline/scheme: {type(e).__name__}: {e}", 2)

    # ---- data, pre-flight, official check -------------------------------------
    try:
        ds = data.load(source)
    except (FileNotFoundError, ValueError) as e:
        C.die(str(e), 2)
    off_scheme = Scheme.from_dict(official)
    off_sig = split_signature(make_splits(ds.y, off_scheme))
    splits = make_splits(ds.y, scheme)
    sig = split_signature(splits)
    is_official = sig == off_sig

    desc = describe(cfg["pipeline"])
    C.print_archive(source)
    print(f"config : {C.rel(args.config)}   name: {cfg['name']}")
    print(f"model  : {desc['model']}   selector: {desc['selector']}   scaler: {desc['scaler']}")
    print(f"cv     : {scheme.label()} seed {scheme.seed} ({cv_from}), {len(splits)} fits, "
          f"test_pred={cfg['test_pred']}")
    for o in overrides:
        print(f"override: {o}")
    if not cfg.get("hypothesis"):
        print("note: no hypothesis in the config; write one BEFORE running (archive rule)")
    if not is_official:
        C.banner([f"NOT THE OFFICIAL CV: this run uses {scheme.label()} seed {scheme.seed},",
                  f"official is {off_scheme.label()} seed {off_scheme.seed}. Its split signature",
                  "differs, so it cannot be paired with official runs (compare_runs refuses)",
                  "and its score must not be ranked against them. Fine for a quick test."])

    pf_warnings = preflight(pipe, ds)
    for w in pf_warnings:
        print(f"WARNING (pre-flight fit): {w}")
    if any(w.startswith(("FutureWarning", "DeprecationWarning")) for w in pf_warnings):
        C.banner(["Deprecated parameter in the config (FutureWarnings are treated as bugs):",
                  "fix the config, e.g. LogisticRegression penalty -> l1_ratio (1.0 = L1, 0.0 = L2)."
                  + ("" if args.allow_deprecated else " Nothing archived.")], "!")
        if not args.allow_deprecated:
            return 2

    # ---- run + archive -------------------------------------------------------
    t0 = time.perf_counter()
    run = Run(cfg["name"], cfg, kind="experiment", author=cfg.get("author"), source=source,
              hypothesis=str(cfg.get("hypothesis") or ""))
    print(f"run_id : {run.run_id}  (running...)")
    with run:
        tp = cfg["test_pred"]
        cv = run_cv(pipe, ds.X, ds.y, scheme, X_test=None if tp == "none" else ds.X_test,
                    test_pred=tp, n_jobs=args.n_jobs)
        run.save_cv(cv)
        summary = run.finish(extra_summary={
            "official_cv": bool(is_official),
            "official_cv_label": f"{off_scheme.label()} seed {off_scheme.seed}",
            "official_split_signature": off_sig,
            "cv_from": cv_from,
            "test_pred_mode": tp,
            "has_test_pred": cv.test_pred is not None,
            "overrides": overrides,
            "config_file": C.rel(args.config),
            "preflight_warnings": pf_warnings,
            "n_jobs": args.n_jobs,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
        })

    # ---- report ----------------------------------------------------------------
    rid = run.run_id
    print()
    print(f"run_id       : {rid}")
    print(f"auc_mean     : {C.fnum(summary['auc_mean'])} +/- {C.fnum(summary['auc_std_repeat'])} "
          f"(std across {scheme.n_repeats} repeats; fold std {C.fnum(summary['auc_std_fold'])})")
    if "oof_auc_mean" in summary:
        print(f"oof_auc_mean : {C.fnum(summary['oof_auc_mean'])}")
    print(f"split sig    : {summary['split_signature']} ({scheme.label()} seed {scheme.seed})")
    print(f"official_cv  : {'yes' if is_official else 'NO'}")
    print(f"folder       : {run.dir}")
    print(f"time         : {summary['elapsed_seconds']:.1f} s")
    if not is_official:
        C.banner([f"WARNING: {rid} is NOT on the official CV ({off_scheme.label()}).",
                  "Re-run without cv overrides before comparing or submitting."])
    d = C.data_flag(source)
    print("\nNext steps:")
    print("  write the conclusion in " + C.rel(run.dir / "notes.md"))
    print("  compare : " + C.hint("compare_runs.py", f"{rid} <OTHER_RUN_ID>{d}"))
    print("            " + C.hint("compare_runs.py", f"--ref {rid} <RUN_ID> <RUN_ID> ...{d}"))
    print("  list    : " + C.hint("list_runs.py", f"--top 10{d}"))
    if summary.get("has_test_pred"):
        print("  submit  : " + C.hint("make_submission.py",
                                      f"--run {rid} --desc \"...\" --expected-public <YOUR_GUESS>{d}"))
    return 0


if __name__ == "__main__":
    sys.exit(C.run_main(main))
