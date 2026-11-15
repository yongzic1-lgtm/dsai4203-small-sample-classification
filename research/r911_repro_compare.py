"""R911 - reproducibility and submission-integrity checks (verifier `repro`, 2026-10-22). Pure file reading:
runs locally, no model is fitted (archive rule: local runs only for file checks).

  A  archived vs re-run (research/r910_repro_rerun.py on a fresh Colab VM, today's code):
       per-fold AUC (matched by val_id), oof.npy, test_pred.npy (max |diff|, #differ, Spearman, bit-identity),
       r201 chosen reg_param / in-fold block per fold, the E3 leaky cell, the BLAS-thread repeats of r910,
       the archived E3 config fields, and which code bundle each run records.
  B  submissions S001-S003 (registry/submissions.csv): file present, name = S###_<run_id>.csv, columns, 19,750
       rows, ids = sample_submission ids in the same order, no NaN/inf, in [0, 1], sha256 prefix, values =
       "%.8f" of the run's test_pred.npy after the recorded score_transform, cv_auc_mean = run summary, data
       fingerprint of the run = today's data; plus the tie structure the %.8f format creates.
  C  dry run of scripts/make_submission.py for the E3 final model and r201_qda_block WITHOUT writing into
       submissions/ or the registry: the same steps as ssc.submission.create (to_unit_interval, sample ids,
       %.8f, validate_file) into a temporary folder, plus the checks make_submission.py does first.
  D  environment: packages recorded in every Colab run's meta.json vs requirements-lock.txt and
       colab/requirements-colab.txt; env_hash vs sha256(requirements-colab.txt); local venv vs the lock.

    .venv\\Scripts\\python.exe research\\r911_repro_compare.py --rerun-prefix r910_
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from ssc import paths
from ssc.data import load_sample_submission, sha256
from ssc.submission import next_sub_id, to_unit_interval, validate_file
from ssc.tracking import Run, read_registry

ARCHIVED = {"s003": "20261022-050708_r003_gmm2_hv", "e3final": "20261022-063908_e3_leakage",
            "r201qda": "20261022-071234_r201_qda_block"}
RERUN_NAME = {"s003": "r910_s003_gmm2_hv", "e3final": "r910_e3_final_k20", "r201qda": "r910_r201_qda_block"}
E3_ARCHIVED = {"pipeline": {"scaler": "standard",
                            "model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 0.0, "solver": "liblinear"}},
                            "random_state": 0},
               "selection": {"method": "pearson"}, "leaderboard": {"k_feat": 20, "k_fold_reference": 5}}
CORE = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "joblib", "threadpoolctl"]
META_NAME = {"sklearn": "scikit-learn"}

HYP = ("A: every re-run fold AUC, oof and test_pred value equals the archived one (bit-identical, or max |diff| "
       "< 1e-12 for the BLAS-heavy GMM). B: S001-S003 files are valid, their sha256 prefixes match the registry "
       "and their values are exactly '%.8f' of test_pred.npy. C: make_submission would accept both runs (E3 "
       "with a non-official-CV warning, because its summary has no official_cv flag). D: every Colab run "
       "recorded numpy 2.4.6 / pandas 3.0.6 / scipy 1.17.1 / sklearn 1.9.1 / lightgbm 4.7.0 / xgboost 3.2.0 = "
       "the lock; refuted by any mismatch.")


def exp_dir(run_id: str) -> Path:
    return paths.experiments_dir("raw") / run_id


def latest_run(name: str) -> str:
    """Newest COLAB run of that name in the real archive."""
    exp = paths.experiments_dir("raw")
    cands = sorted(p.name for pat in (f"*_{name}", f"*_{name}-*") for p in exp.glob(pat)
                   if (p / "summary.json").exists()
                   and (jload(p / "meta.json").get("compute") or {}).get("host") == "colab")
    if not cands:
        raise SystemExit(f"no Colab run named {name} in the archive")
    return cands[-1]


def jload(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def arr_cmp(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return {"what": label, "shape_a": str(a.shape), "shape_b": str(b.shape), "comparable": False}
    m = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[m] - b[m])
    return {"what": label, "n": int(a.size), "nan_pattern_equal": bool(np.array_equal(np.isnan(a), np.isnan(b))),
            "bit_identical": bool(np.array_equal(a, b, equal_nan=True)), "max_abs_diff": float(d.max()) if d.size else 0.0,
            "n_differ": int((d > 0).sum()),
            "spearman": float(spearmanr(a[m].ravel(), b[m].ravel()).statistic) if d.size > 2 else float("nan"),
            "sha_a": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16],
            "sha_b": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()[:16]}


def fold_cmp(fa: pd.DataFrame, fb: pd.DataFrame, label: str) -> dict:
    m = fa.merge(fb, on=["repeat", "fold"], suffixes=("_a", "_b"), validate="one_to_one")
    d = (m["auc_a"] - m["auc_b"]).abs()
    return {"what": label, "n": int(len(m)), "n_a": int(len(fa)), "n_b": int(len(fb)),
            "val_id_all_equal": bool((m["val_id_a"] == m["val_id_b"]).all()),
            "bit_identical": bool((m["auc_a"] == m["auc_b"]).all()), "max_abs_diff": float(d.max()),
            "n_differ": int((d > 0).sum()), "mean_a": float(m["auc_a"].mean()), "mean_b": float(m["auc_b"].mean())}


def bundle_of(run_id: str) -> str:
    c = jload(exp_dir(run_id) / "meta.json").get("compute") or {}
    return c.get("code_bundle", "none recorded (pre-D005 run)")


def part_a(prefix: str, run: Run) -> pd.DataFrame:
    rows, extra = [], {}
    reruns = {k: latest_run(v) for k, v in RERUN_NAME.items()}
    extra["reruns"] = reruns
    for key, arch in ARCHIVED.items():
        new = reruns[key]
        A, B = exp_dir(arch), exp_dir(new)
        base = {"model": key, "archived": arch, "rerun": new, "bundle_archived": bundle_of(arch),
                "bundle_rerun": bundle_of(new)}
        if key == "e3final":
            g = pd.read_csv(A / "fold_scores_grid.csv")
            fa = g[(g["procedure"] == "honest") & (g["k_feat"] == 20) & (g["k_fold"] == 5)].drop(
                columns=["procedure", "k_feat", "k_fold", "split_sig"])
        else:
            fa = pd.read_csv(A / "fold_scores.csv")
        fb = pd.read_csv(B / "fold_scores.csv")
        rows.append({**base, **fold_cmp(fa, fb, "fold_auc")})
        if (A / "oof.npy").exists():
            rows.append({**base, **arr_cmp(np.load(A / "oof.npy"), np.load(B / "oof.npy"), "oof")})
        rows.append({**base, **arr_cmp(np.load(A / "test_pred.npy"), np.load(B / "test_pred.npy"), "test_pred")})
        sa, sb = jload(A / "summary.json"), jload(B / "summary.json")
        rows.append({**base, "what": "summary auc_mean", "max_abs_diff": abs(sa["auc_mean"] - sb["auc_mean"]),
                     "bit_identical": sa["auc_mean"] == sb["auc_mean"], "mean_a": sa["auc_mean"], "mean_b": sb["auc_mean"]})
        if key == "r201qda":
            ia = pd.read_csv(A / "fold_fit_info.csv")
            ib = pd.read_csv(B / "fold_fit_info.csv")
            m = ia.merge(ib, on=["repeat", "fold"], suffixes=("_a", "_b"))
            rows.append({**base, "what": "fold_fit_info best_params", "n": int(len(m)),
                         "n_differ": int((m["best_params_a"] != m["best_params_b"]).sum()),
                         "bit_identical": bool((m["best_params_a"] == m["best_params_b"]).all())})
            rows.append({**base, "what": "fold_fit_info support (in-fold block)", "n": int(len(m)),
                         "n_differ": int((m["support_a"] != m["support_b"]).sum()),
                         "bit_identical": bool((m["support_a"] == m["support_b"]).all())})
            d = (m["best_inner_auc_a"] - m["best_inner_auc_b"]).abs()
            rows.append({**base, "what": "fold_fit_info best_inner_auc", "n": int(len(m)),
                         "max_abs_diff": float(d.max()), "n_differ": int((d > 0).sum()),
                         "bit_identical": bool((d == 0).all())})
    # E3 leaky cell + thread sensitivity, from the r910 analysis run
    env_run = latest_run("r910_repro_env")
    extra["env_run"] = env_run
    E = exp_dir(env_run)
    g = pd.read_csv(exp_dir(ARCHIVED["e3final"]) / "fold_scores_grid.csv")
    fl = g[(g["procedure"] == "leaky") & (g["k_feat"] == 20) & (g["k_fold"] == 5)].drop(
        columns=["procedure", "k_feat", "k_fold", "split_sig"])
    rows.append({"model": "e3leaky", "archived": ARCHIVED["e3final"], "rerun": env_run,
                 **fold_cmp(fl, pd.read_csv(E / "e3_leaky_fold_scores.csv"), "fold_auc (leaky k20, 5-fold)")})
    # archived E3 config fields
    cfg = yaml.safe_load((exp_dir(ARCHIVED["e3final"]) / "config.yaml").read_text(encoding="utf-8"))
    for k, v in E3_ARCHIVED.items():
        rows.append({"model": "e3final", "archived": ARCHIVED["e3final"], "what": f"archived config `{k}` == r910 literal",
                     "bit_identical": cfg.get(k) == v})
    res = pd.DataFrame(rows)
    run.save_table("A_archived_vs_rerun", res)
    th = E / "thread_sensitivity.csv"
    if th.exists():
        run.save_table("A_thread_sensitivity_r910", pd.read_csv(th))
    run.summary["A"] = extra
    return res


def part_b(run: Run) -> pd.DataFrame:
    reg = read_registry(paths.submissions_csv("raw"))
    sample = load_sample_submission("raw")
    rows, ties = [], []
    for _, r in reg.iterrows():
        f = paths.archive_root("raw") / r["file"]
        row = {"sub_id": r["sub_id"], "run_id": r["run_id"], "file": r["file"], "exists": f.exists()}
        if not f.exists():
            rows.append(row)
            continue
        sub = pd.read_csv(f)
        tp = np.load(exp_dir(r["run_id"]) / "test_pred.npy")
        vals, transform = to_unit_interval(tp)
        v = sub["target"].to_numpy(float) if "target" in sub else np.full(len(sub), np.nan)
        expect = np.array([float("%.8f" % x) for x in vals])
        summ = jload(exp_dir(r["run_id"]) / "summary.json")
        meta = jload(exp_dir(r["run_id"]) / "meta.json")
        today_fp = {n: sha256(paths.data_dir("raw") / n)[:16] for n in ("train.csv", "test.csv")}
        row.update({
            "name_matches": f.name == f"{r['sub_id']}_{r['run_id']}.csv",
            "columns": "|".join(sub.columns), "columns_ok": list(sub.columns) == ["id", "target"],
            "n_rows": int(len(sub)), "n_rows_ok": len(sub) == len(sample) == 19750,
            "ids_equal_sample_in_order": bool(np.array_equal(sub["id"].to_numpy(), sample["id"].to_numpy())),
            "ids_unique": bool(sub["id"].is_unique),
            "n_nan": int(np.isnan(v).sum()), "n_inf": int(np.isinf(v).sum()),
            "in_unit_interval": bool((v >= 0).all() and (v <= 1).all()),
            "validate_file_problems": "; ".join(validate_file(f, "raw")) or "none",
            "sha256_16": sha256(f)[:16], "sha256_registry": r["sha256"], "sha_ok": sha256(f)[:16] == r["sha256"],
            "transform_recomputed": transform, "transform_registry": r["score_transform"],
            "transform_ok": transform == r["score_transform"],
            "values_equal_pct8f_of_test_pred": bool(np.array_equal(v, expect)),
            "max_abs_diff_vs_test_pred": float(np.abs(v - vals).max()),
            "spearman_csv_vs_test_pred": float(spearmanr(v, vals).statistic),
            "n_unique_test_pred": int(len(np.unique(vals))), "n_unique_csv": int(len(np.unique(v))),
            "largest_tie_csv": int(pd.Series(v).value_counts().iloc[0]),
            "rows_in_ties_csv": int(pd.Series(v).value_counts().loc[lambda s: s > 1].sum()),
            "rows_in_ties_test_pred": int(pd.Series(vals).value_counts().loc[lambda s: s > 1].sum()),
            "cv_auc_registry": r["cv_auc_mean"], "cv_auc_run": summ.get("auc_mean"),
            "cv_auc_ok": abs(float(r["cv_auc_mean"]) - float(summ.get("auc_mean"))) < 5e-6,
            "run_status": summ.get("status"),
            "run_data_fp_equals_today": meta.get("data_fingerprint") == today_fp,
            "run_created_at": meta.get("created_at"), "sub_created_at": r["created_at"],
        })
        rows.append(row)
        vc = pd.Series(v).value_counts()
        for val, n in vc.head(5).items():
            ties.append({"sub_id": r["sub_id"], "value": val, "n_rows": int(n)})
    res = pd.DataFrame(rows)
    run.save_table("B_submissions", res)
    run.save_table("B_largest_ties", pd.DataFrame(ties))
    return res


def part_c(run: Run) -> pd.DataFrame:
    sample = load_sample_submission("raw")
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="r911_dryrun_"))
    nxt = next_sub_id("raw")  # read-only: registry + glob of submissions/
    for key in ("e3final", "r201qda"):
        rid = ARCHIVED[key]
        d = exp_dir(rid)
        summ = jload(d / "summary.json")
        tp = np.load(d / "test_pred.npy")
        vals, transform = to_unit_interval(tp)
        out = tmp / f"{nxt}_{rid}.csv"  # the name create() would use; written to a temp dir only
        pd.DataFrame({"id": sample["id"].to_numpy(), "target": vals}).to_csv(out, index=False, float_format="%.8f")
        v = pd.read_csv(out)["target"].to_numpy(float)
        rows.append({"model": key, "run_id": rid, "status": summ.get("status"),
                     "summary_official_cv": summ.get("official_cv"),
                     "make_submission_warns_not_official": summ.get("official_cv") is not True,
                     "split_signature": summ.get("split_signature"), "cv_auc_mean": summ.get("auc_mean"),
                     "has_test_pred": True, "n_test_pred": int(len(tp)), "len_ok": len(tp) == len(sample),
                     "test_pred_finite": bool(np.isfinite(tp).all()), "transform": transform,
                     "test_pred_min": float(tp.min()), "test_pred_max": float(tp.max()),
                     "validate_file_problems": "; ".join(validate_file(out, "raw")) or "none",
                     "would_be_sub_id": nxt, "would_be_file": f"submissions/{nxt}_{rid}.csv",
                     "dryrun_sha256_16": sha256(out)[:16],
                     "n_unique_test_pred": int(len(np.unique(vals))), "n_unique_csv": int(len(np.unique(v))),
                     "largest_tie_csv": int(pd.Series(v).value_counts().iloc[0]),
                     "spearman_csv_vs_test_pred": float(spearmanr(v, vals).statistic),
                     "max_abs_diff_vs_test_pred": float(np.abs(v - vals).max())})
    res = pd.DataFrame(rows)
    run.save_table("C_make_submission_dryrun", res)
    for f in tmp.glob("*.csv"):
        f.unlink()
    tmp.rmdir()
    return res


def read_pins(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            n, v = line.split("==", 1)
            out[n.strip().lower()] = v.strip()
    return out


def part_d(run: Run) -> pd.DataFrame:
    lock = read_pins(paths.ROOT / "requirements-lock.txt")
    colab = read_pins(paths.ROOT / "colab" / "requirements-colab.txt")
    env_hash_now = hashlib.sha256((paths.ROOT / "colab" / "requirements-colab.txt").read_bytes()).hexdigest()[:16]
    rows = []
    for m in sorted(paths.experiments_dir("raw").glob("*/meta.json")):
        meta = jload(m)
        comp = meta.get("compute") or {}
        pk = meta.get("packages") or {}
        row = {"run_id": meta["run_id"], "host": comp.get("host", "none recorded"), "hardware": comp.get("hardware"),
               "session": comp.get("session"), "env_hash": comp.get("env_hash"), "code_bundle": comp.get("code_bundle"),
               "env_hash_matches_today": comp.get("env_hash") == env_hash_now, "python": pk.get("python")}
        mism = []
        for k, v in pk.items():
            if k == "python":
                continue
            n = META_NAME.get(k, k)
            row[k] = v
            for src, pins in (("lock", lock), ("colab", colab)):
                want = pins.get(n) or (pins.get("xgboost-cpu") if n == "xgboost" else None)
                if want is not None and v != want:
                    mism.append(f"{k} {v} != {src} {want}")
        row["mismatches"] = "; ".join(mism)
        rows.append(row)
    runs = pd.DataFrame(rows)
    run.save_table("D_run_packages", runs)
    pins = []
    for n in sorted(set(lock) | set(colab)):
        if n not in colab and n not in CORE:
            continue
        try:
            local = metadata.version(n)
        except metadata.PackageNotFoundError:
            local = None
        pins.append({"package": n, "lock": lock.get(n), "colab": colab.get(n), "local_venv": local,
                     "lock_eq_colab": lock.get(n) == colab.get(n), "local_eq_lock": local == lock.get(n)})
    pins = pd.DataFrame(pins)
    run.save_table("D_pins", pins)
    run.summary["D"] = {"env_hash_today": env_hash_now, "n_runs": int(len(runs)),
                        "n_runs_with_mismatch": int((runs["mismatches"] != "").sum()),
                        "python_versions": runs["python"].value_counts().to_dict(),
                        "env_hash_values": runs["env_hash"].fillna("<none>").value_counts().to_dict(),
                        "code_bundle_unsynced": int((runs["code_bundle"] == "unsynced (--no-sync)").sum()),
                        "code_bundle_missing": int(runs["code_bundle"].isna().sum())}
    return pins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--author", default=None)
    ap.add_argument("--skip-a", action="store_true", help="skip part A (no r910 re-run merged yet)")
    a = ap.parse_args()
    cfg = {"name": "r911_repro_compare", "description": "archived vs re-run arrays, submission files, make_submission "
           "dry run, environment versions (pure file checks, local)", "archived": ARCHIVED, "reruns_named": RERUN_NAME}
    with Run("r911_repro_compare", cfg, kind="analysis", source="raw", author=a.author, hypothesis=HYP) as run:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 40)
        if not a.skip_a:
            A = part_a("r910_", run)
            print("A archived vs re-run\n" + A.drop(columns=[c for c in ("sha_a", "sha_b", "archived") if c in A]).to_string(index=False))
        B = part_b(run)
        print("\nB submissions\n" + B.T.to_string())
        C = part_c(run)
        print("\nC make_submission dry run\n" + C.T.to_string())
        D = part_d(run)
        print("\nD pins\n" + D.to_string(index=False))
        print("\nD summary " + json.dumps(run.summary["D"], indent=1))
        out = paths.tables_dir("raw")
        out.mkdir(parents=True, exist_ok=True)
        for name in ("A_archived_vs_rerun", "B_submissions", "C_make_submission_dryrun", "D_pins"):
            p = run.dir / f"{name}.csv"
            if p.exists():
                pd.read_csv(p).to_csv(out / f"r911_{name}.csv", index=False)
        run.finish(extra_summary={"B_all_ok": bool(B[["columns_ok", "n_rows_ok", "ids_equal_sample_in_order", "sha_ok",
                                                      "transform_ok", "values_equal_pct8f_of_test_pred"]].all().all()),
                                  "C_problems": C["validate_file_problems"].tolist()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
