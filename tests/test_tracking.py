"""Tests for ssc.tracking: run folders, the experiments registry, BOM-safe CSV appends,
the archive isolation that keeps tests out of the real records, and the machine/model
metadata in meta.json (BLAS thread pools, CPU model, estimator get_params)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ssc import paths, tracking
from ssc.models import build_pipeline
from ssc.tracking import EXPERIMENT_COLUMNS, Run, append_csv_row, load_run, read_registry, slugify
from ssc.validation import Scheme, run_cv

BOM = b"\xef\xbb\xbf"

CONFIG = {
    "name": "unit-test",
    "hypothesis": "from config",
    "pipeline": {"scaler": "standard", "selector": {"name": "corr_topk", "k": 10},
                 "model": {"name": "logreg", "params": {"C": 0.5}}, "random_state": 0},
    "cv": {"kind": "kfold", "n_splits": 5, "n_repeats": 2, "seed": 3},
}


def _cv(small_xy, cfg=CONFIG):
    X, y, Xt = small_xy
    return run_cv(build_pipeline(cfg["pipeline"]), X, y, Scheme.from_dict(cfg["cv"]), X_test=Xt)


# ---------------------------------------------------------------- isolation
def test_archive_is_redirected_to_tmp(isolated_archive):
    for source in paths.SOURCES:
        for d in (paths.experiments_dir(source), paths.registry_dir(source), paths.submissions_dir(source),
                  paths.figures_dir(source), paths.tables_dir(source)):
            assert d.is_relative_to(isolated_archive)
            assert not d.is_relative_to(paths.ROOT)
    assert paths.resolve_source() == "synthetic"


def test_raw_source_run_also_lands_in_tmp(isolated_archive):
    run = Run("raw-source-smoke", {}, source="raw")
    run.finish()
    assert run.dir.is_relative_to(isolated_archive)
    assert paths.experiments_csv("raw").is_relative_to(isolated_archive)


# ---------------------------------------------------------------- Run lifecycle
def test_run_creates_files_and_registry_row(small_xy, isolated_archive):
    run = Run("Unit Test: v1", CONFIG, source="synthetic", hypothesis="corr top-10 helps")
    for f in ("config.yaml", "meta.json", "notes.md"):
        assert (run.dir / f).exists()
    assert not (run.dir / "summary.json").exists()  # only written by finish()
    assert run.run_id.endswith("_unit-test-v1")
    assert run.dir.parent == paths.experiments_dir("synthetic")

    cv = _cv(small_xy)
    summary = run.save_cv(cv)
    for f in ("fold_scores.csv", "oof.npy", "test_pred.npy"):
        assert (run.dir / f).exists()
    run.finish()
    saved = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done" and saved["scheme"] == cv.scheme
    assert saved["auc_mean"] == pytest.approx(summary["auc_mean"]) and "finished_at" in saved

    reg = read_registry(paths.experiments_csv("synthetic"))
    assert list(reg.columns) == EXPERIMENT_COLUMNS
    assert len(reg) == 1
    row = reg.iloc[0]
    assert row["run_id"] == run.run_id and row["kind"] == "experiment"
    assert row["author"] == "pytest" and row["name"] == "Unit Test: v1"
    assert row["data_source"] == "synthetic" and row["status"] == "done"
    assert row["model"] == "logreg(C=0.5)" and row["selector"] == "corr_topk(k=10)"
    assert row["scaler"] == "standard"
    assert row["cv"] == "5foldx2" and row["cv_seed"] == "3"
    assert row["auc_mean"] == f"{summary['auc_mean']:.5f}"
    assert row["oof_auc_mean"] == f"{summary['oof_auc_mean']:.5f}"
    assert row["split_sig"] == cv.split_signature
    assert row["hypothesis"] == "corr top-10 helps"


def test_run_metadata_contents(isolated_archive):
    cfg = {"a": 1, "nested": {"list": [1, 2, 3], "text": "unicode ok: é"}}
    run = Run("meta", cfg, source="synthetic")
    assert yaml.safe_load((run.dir / "config.yaml").read_text(encoding="utf-8")) == cfg
    meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == run.run_id and meta["data_source"] == "synthetic"
    assert {"commit", "dirty", "branch"} <= set(meta["git"])
    assert meta["packages"]["sklearn"] and meta["packages"]["python"]
    assert set(meta["data_fingerprint"]) <= {"train.csv", "test.csv"}
    notes = (run.dir / "notes.md").read_text(encoding="utf-8")
    assert run.run_id in notes and "(none given)" in notes and "author: pytest" in notes


def test_registry_row_without_pipeline_or_cv(isolated_archive):
    run = Run("bare", {}, kind="analysis", source="synthetic",
              hypothesis="line one\nline two" + "x" * 400)
    run.finish(extra_summary={"note": 1})
    row = read_registry(paths.experiments_csv("synthetic")).iloc[0]
    assert row["kind"] == "analysis" and row["model"] == "" and row["cv"] == "" and row["auc_mean"] == ""
    assert "\n" not in row["hypothesis"] and len(row["hypothesis"]) == 300


def test_same_second_runs_get_unique_ids(monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 22, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(tracking, "datetime", FrozenDatetime)
    runs = [Run("dup", {}, source="synthetic") for _ in range(3)]
    ids = [r.run_id for r in runs]
    assert ids == ["20261022-120000_dup", "20261022-120000_dup-2", "20261022-120000_dup-3"]
    assert all(r.dir.is_dir() for r in runs)


def test_context_manager_records_failure():
    with pytest.raises(RuntimeError, match="boom"):
        with Run("crash", {}, source="synthetic") as run:
            raise RuntimeError("boom")
    saved = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed: RuntimeError"
    reg = read_registry(paths.experiments_csv("synthetic"))
    assert reg["status"].tolist() == ["failed: RuntimeError"]


def test_context_manager_success_writes_one_row():
    with Run("auto", {}, source="synthetic"):
        pass
    with Run("explicit", {}, source="synthetic") as run:
        run.finish(status="done", extra_summary={"auc_mean": 0.61})
    reg = read_registry(paths.experiments_csv("synthetic"))
    assert reg["status"].tolist() == ["done", "done"]  # explicit finish() is not repeated by __exit__
    assert reg["auc_mean"].tolist() == ["", "0.61000"]


def test_load_run_roundtrip(small_xy):
    cv = _cv(small_xy)
    with Run("roundtrip", CONFIG, source="synthetic") as run:
        run.save_cv(cv)
    out = load_run(run.run_id, "synthetic")
    assert out["dir"] == run.dir and out["config"] == CONFIG
    assert out["meta"]["run_id"] == run.run_id
    assert out["summary"]["auc_mean"] == pytest.approx(cv.summary()["auc_mean"])
    pd.testing.assert_frame_equal(out["fold_scores"], cv.fold_scores, check_dtype=False)
    np.testing.assert_array_equal(out["test_pred"], cv.test_pred)
    np.testing.assert_array_equal(np.load(run.dir / "oof.npy"), cv.oof)
    with pytest.raises(FileNotFoundError):
        load_run("19990101-000000_missing", "synthetic")


def test_save_helpers_write_into_run_dir():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Run("helpers", {}, source="synthetic")
    t = run.save_table("tbl", pd.DataFrame({"a": [1, 2]}))
    a = run.save_array("arr", [1.0, 2.0])
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    f = run.save_figure("fig", fig)
    plt.close(fig)
    assert t == run.dir / "tbl.csv" and a == run.dir / "arr.npy" and f == run.dir / "fig.png"
    assert all(p.exists() for p in (t, a, f))
    np.testing.assert_array_equal(np.load(a), [1.0, 2.0])


# ---------------------------------------------------------------- CSV helpers
def test_append_csv_row_writes_bom_only_once(tmp_path):
    path = tmp_path / "reg" / "x.csv"
    cols = ["a", "b", "c"]
    append_csv_row(path, cols, {"a": 1, "b": "x, with comma", "c": None})
    append_csv_row(path, cols, {"a": 2, "b": 'quote "q"', "extra": "ignored"})
    append_csv_row(path, cols, {"a": 3, "b": "unicode é", "c": 0.5})
    raw = path.read_bytes()
    assert raw.startswith(BOM) and raw.count(BOM) == 1
    reg = read_registry(path)
    assert list(reg.columns) == cols  # no BOM glued to the first column name
    assert reg["a"].tolist() == ["1", "2", "3"]
    assert reg["b"].tolist() == ["x, with comma", 'quote "q"', "unicode é"]
    assert reg["c"].tolist() == ["", "", "0.5"]


def test_append_csv_row_treats_empty_file_as_new(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    append_csv_row(path, ["a"], {"a": "v"})
    assert path.read_bytes().startswith(BOM)
    assert read_registry(path)["a"].tolist() == ["v"]


def test_read_registry_missing_file_is_empty(tmp_path):
    assert read_registry(tmp_path / "nope.csv").empty


def test_append_csv_row_repairs_a_missing_final_newline(tmp_path):
    path = tmp_path / "reg.csv"
    append_csv_row(path, ["a", "b"], {"a": 1, "b": 2})
    path.write_bytes(path.read_bytes().rstrip(b"\r\n"))  # e.g. after a hand edit
    append_csv_row(path, ["a", "b"], {"a": 3, "b": 4})
    assert read_registry(path)["a"].tolist() == ["1", "3"]


def test_read_registry_names_the_file_on_parse_errors(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("a,b\n1,2\n3,4,5,6,7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.csv"):
        read_registry(path)


def test_locked_registry_keeps_run_done_and_saves_the_row(monkeypatch):
    monkeypatch.setattr(tracking, "LOCK_RETRY_SECONDS", 0.0)
    real = tracking.append_csv_row

    def locked(path, columns, row):  # what Windows does while Excel has the file open
        tracking.retry_locked(lambda: (_ for _ in ()).throw(PermissionError(13, "Permission denied")), path)

    monkeypatch.setattr(tracking, "append_csv_row", locked)
    with pytest.raises(tracking.RegistryLockedError, match="append_saved_row"):
        with Run("locked", {}, source="synthetic") as run:
            run.finish()
    saved = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done"  # not "failed: ...": the run itself completed
    assert (run.dir / "registry_row.json").exists()
    monkeypatch.setattr(tracking, "append_csv_row", real)
    tracking.append_saved_row(run.dir)
    assert read_registry(paths.experiments_csv("synthetic"))["run_id"].tolist() == [run.run_id]
    assert not (run.dir / "registry_row.json").exists()
    with pytest.raises(FileNotFoundError):
        tracking.append_saved_row(run.dir)


def test_run_id_and_created_at_use_archive_clock():
    run = Run("tz", {}, source="synthetic")
    assert run.meta["created_at"].endswith("+09:00")
    stamp = datetime.strptime(run.run_id.split("_")[0], "%Y%m%d-%H%M%S")
    archive_now = datetime.now(tracking.ARCHIVE_TZ).replace(tzinfo=None)
    assert abs((archive_now - stamp).total_seconds()) < 120


def test_compute_info_from_launcher_env(monkeypatch):
    monkeypatch.setenv("SSC_COMPUTE", '{"host": "colab", "account": "pro1", "session": "ssc"}')
    c = tracking.compute_info()
    assert c["host"] == "colab" and c["account"] == "pro1" and "node" in c
    monkeypatch.delenv("SSC_COMPUTE")
    assert tracking.compute_info()["host"] == "local"


def test_git_info_prefers_launcher_env(monkeypatch):
    monkeypatch.setenv("SSC_GIT_COMMIT", "abc1234567")
    monkeypatch.setenv("SSC_GIT_DIRTY", "True")
    g = tracking.git_info()
    assert g["commit"] == "abc1234567" and g["dirty"] is True and g["from_env"] is True
    monkeypatch.setenv("SSC_GIT_DIRTY", "")
    assert tracking.git_info()["dirty"] is None
    monkeypatch.delenv("SSC_GIT_COMMIT")
    assert "from_env" not in tracking.git_info()


def test_slugify():
    assert slugify("Unit Test: v1!") == "unit-test-v1"
    assert slugify("  ///  ") == "run"
    assert slugify("a" * 100) == "a" * 40
    assert slugify("keep_these.chars-ok") == "keep_these.chars-ok"


# ---------------------------------------------------------------- machine and model metadata
def test_meta_records_blas_cpu_and_parallel_package_versions():
    run = Run("env-meta", {}, source="synthetic")
    meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["packages"]["joblib"] and meta["packages"]["threadpoolctl"]
    assert isinstance(meta["compute"]["cpu_model"], str) and meta["compute"]["cpu_model"]
    pools = meta["threadpools"]
    assert isinstance(pools, list) and any(p.get("user_api") == "blas" for p in pools)  # numpy's BLAS is loaded
    blas = next(p for p in pools if p.get("user_api") == "blas")
    assert {"internal_api", "version", "architecture", "num_threads", "library"} <= set(blas)
    assert "\\" not in blas["library"] and "/" not in blas["library"]  # file name only, no local path


def test_cpu_model_on_colab_env_too(monkeypatch):
    monkeypatch.setenv("SSC_COMPUTE", '{"host": "colab", "code_bundle": "abc"}')
    c = tracking.compute_info()
    assert c["code_bundle"] == "abc" and c["cpu_model"] == tracking.cpu_model() and c["cpu_model"]


def test_save_cv_records_estimator_params(small_xy):
    run = Run("est-params", CONFIG, source="synthetic")
    run.save_cv(_cv(small_xy))  # run_cv keeps the unfitted pipeline in cv.extra["estimator"]
    run.finish()
    est = load_run(run.run_id, "synthetic")["meta"]["estimator"]
    assert est["class"] == "sklearn.pipeline.Pipeline"
    p = est["params"]
    assert p["model__C"] == 0.5 and p["model__random_state"] == 0 and p["model__solver"] == "liblinear"
    assert p["select__k"] == 10 and p["model__max_iter"] == 5000  # defaults the config never named
    assert isinstance(p["steps"], list) and p["steps"][0][0] == "scale"
    # an explicit estimator wins over the one kept in cv.extra
    run2 = Run("est-explicit", CONFIG, source="synthetic")
    run2.save_cv(_cv(small_xy), estimator=build_pipeline({"model": {"name": "gnb"}}))
    assert "model__var_smoothing" in run2.meta["estimator"]["params"]
    run2.finish()


def test_estimator_params_are_json_safe():
    class Odd:
        def get_params(self, deep=True):
            return {"a": np.int64(3), "b": np.float32(0.5), "c": float("nan"), "d": (1, np.inf),
                    "e": np.array([1.0, 2.0]), "f": np.bool_(True), "g": build_pipeline({"model": {"name": "gnb"}}),
                    "h": {"k": None}, "i": object()}

    out = tracking.estimator_params(Odd())
    json.dumps(out, allow_nan=False)  # strict JSON: no NaN/inf, no numpy types
    p = out["params"]
    assert p["a"] == 3 and p["b"] == 0.5 and p["c"] == "nan" and p["d"] == [1, "inf"]
    assert p["e"] == [1.0, 2.0] and p["f"] is True and p["h"] == {"k": None}
    assert p["g"].startswith("Pipeline(") and p["i"].startswith("<object")
    no_params = tracking.estimator_params(object())
    assert no_params["class"] == "builtins.object" and "error" in no_params and "params" not in no_params
