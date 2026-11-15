"""Tests for ssc.submission: score mapping to [0, 1], submission files built from a run,
the submissions registry (ids, record(), daily count on the Kaggle UTC day),
validate_file(), the test-id checks, create(dry_run=True), the team daily limit, and
scripts/make_submission.py (--dry-run, stop at the team limit unless --force-limit)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

from ssc import paths, submission
from ssc.data import load_sample_submission, sha256
from ssc.submission import (SUBMISSION_COLUMNS, create, next_sub_id, record, submissions_today,
                            to_unit_interval, validate_file)
from ssc.tracking import Run, append_csv_row, load_run, read_registry

BOM = b"\xef\xbb\xbf"
N_TEST = 19750


# ---------------------------------------------------------------- to_unit_interval
def test_to_unit_interval_passes_in_range_scores_through():
    s = np.array([0.0, 0.2, 0.2, 0.9, 1.0])
    out, how = to_unit_interval(s)
    assert how == "none" and np.array_equal(out, s)


def test_to_unit_interval_rank_transforms_out_of_range_scores():
    s = np.array([-2.0, 0.5, 3.0, 0.5, 10.0])
    out, how = to_unit_interval(s)
    assert how == "rank"
    np.testing.assert_allclose(out, rankdata(s) / len(s))   # ties get the average rank
    assert out.min() > 0 and out.max() == 1.0
    assert spearmanr(out, s).statistic == pytest.approx(1.0)  # ordering (hence AUC) unchanged
    assert to_unit_interval(np.array([-0.1, 0.5]))[1] == "rank"
    assert to_unit_interval(np.array([0.5, 1.0001]))[1] == "rank"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_to_unit_interval_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="NaN or inf"):
        to_unit_interval(np.array([0.1, bad, 0.3]))


# ---------------------------------------------------------------- create()
def _run_with_test_pred(pred, auc_mean=0.7123, auc_std_repeat=0.0123) -> str:
    run = Run("sub-source", {}, source="synthetic")
    if pred is not None:
        run.save_array("test_pred", pred)
    run.finish(extra_summary={"auc_mean": auc_mean, "auc_std_repeat": auc_std_repeat})
    return run.run_id


@pytest.mark.synthetic_data
def test_create_writes_valid_file_and_registry_rows(isolated_archive):
    pred = np.random.default_rng(0).normal(size=N_TEST)  # decision-function-like, out of [0, 1]
    run_id = _run_with_test_pred(pred)
    row = create(run_id, "first try", expected_public=0.70, source="synthetic", notes="n1")
    assert row["sub_id"] == "S001"
    path = isolated_archive / row["file"]
    assert row["file"] == f"submissions/S001_{run_id}.csv" and path.exists()
    assert validate_file(path, "synthetic") == []
    assert row["score_transform"] == "rank" and row["sha256"] == sha256(path)[:16]
    assert row["author"] == "pytest" and row["run_id"] == run_id

    sub = pd.read_csv(path)
    sample = load_sample_submission("synthetic")
    assert list(sub.columns) == ["id", "target"]
    assert np.array_equal(sub["id"].to_numpy(), sample["id"].to_numpy())
    np.testing.assert_allclose(sub["target"], rankdata(pred) / N_TEST, atol=1e-8)

    row2 = create(_run_with_test_pred(np.linspace(0, 1, N_TEST)), "second", source="synthetic")
    assert row2["sub_id"] == "S002" and row2["score_transform"] == "none"

    reg = read_registry(paths.submissions_csv("synthetic"))
    assert list(reg.columns) == SUBMISSION_COLUMNS
    assert reg["sub_id"].tolist() == ["S001", "S002"]
    r1 = reg.iloc[0]
    assert r1["description"] == "first try" and r1["notes"] == "n1"
    assert float(r1["cv_auc_mean"]) == pytest.approx(0.7123)
    assert float(r1["cv_auc_std_repeat"]) == pytest.approx(0.0123)
    assert float(r1["expected_public"]) == pytest.approx(0.70)
    assert r1["public_lb"] == "" and r1["submitted_at_utc"] == ""
    assert reg.iloc[1]["expected_public"] == ""


@pytest.mark.synthetic_data
def test_create_rejects_bad_runs():
    with pytest.raises(FileNotFoundError, match="test_pred"):
        create(_run_with_test_pred(None), "no preds", source="synthetic")
    with pytest.raises(ValueError, match="sample_submission"):
        create(_run_with_test_pred(np.linspace(0, 1, 100)), "wrong length", source="synthetic")
    assert read_registry(paths.submissions_csv("synthetic")).empty
    assert list(paths.submissions_dir("synthetic").glob("*.csv")) == []


# ---------------------------------------------------------------- record()
@pytest.mark.synthetic_data
def test_record_updates_only_allowed_fields():
    run_id = _run_with_test_pred(np.linspace(0, 1, N_TEST))
    create(run_id, "a", source="synthetic")
    create(run_id, "b", source="synthetic")
    reg_path = paths.submissions_csv("synthetic")
    before = read_registry(reg_path)

    out = record("S001", source="synthetic", public_lb=0.71234, submitted_at_utc="2026-10-22T01:00:00+00:00",
                 notes=None)  # None = leave unchanged
    assert out["public_lb"] == "0.71234" and out["sub_id"] == "S001"
    after = read_registry(reg_path)
    assert list(after.columns) == SUBMISSION_COLUMNS
    changed = {c for c in SUBMISSION_COLUMNS if (before[c] != after[c]).any()}
    assert changed == {"public_lb", "submitted_at_utc"}
    assert after.loc[after["sub_id"] == "S002", "public_lb"].item() == ""  # other rows untouched

    with pytest.raises(ValueError, match="may be recorded"):
        record("S001", source="synthetic", description="rewrite history", public_lb=0.9)
    pd.testing.assert_frame_equal(read_registry(reg_path), after)  # rejected call changed nothing

    with pytest.raises(KeyError):
        record("S999", source="synthetic", public_lb=0.5)

    # record() rewrites the file with a BOM; a later create() must append without a second one
    assert create(run_id, "c", source="synthetic")["sub_id"] == "S003"
    raw = reg_path.read_bytes()
    assert raw.startswith(BOM) and raw.count(BOM) == 1
    assert read_registry(reg_path)["sub_id"].tolist() == ["S001", "S002", "S003"]


def test_record_on_empty_registry_raises_keyerror():
    with pytest.raises(KeyError):
        record("S001", source="synthetic", public_lb=0.5)


# ---------------------------------------------------------------- ids and daily count
def _add_rows(rows: list[dict]) -> None:
    for r in rows:
        append_csv_row(paths.submissions_csv("synthetic"), SUBMISSION_COLUMNS, r)


def test_next_sub_id():
    assert next_sub_id("synthetic") == "S001"
    _add_rows([{"sub_id": "S001"}, {"sub_id": "S007"}, {"sub_id": "manual-1"}])
    assert next_sub_id("synthetic") == "S008"


def test_submissions_today_counts_the_utc_kaggle_day(monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # 2026-10-22 09:30 HKT
            return datetime(2026, 10, 22, 1, 30, 0, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(submission, "datetime", FrozenDatetime)
    assert submissions_today("synthetic") == 0
    _add_rows([
        {"sub_id": "S001", "submitted_at_utc": "2026-10-21T23:59:59+00:00"},  # 07:59 HKT: previous Kaggle day
        {"sub_id": "S002", "submitted_at_utc": "2026-10-22T00:00:00+00:00"},  # 08:00 HKT: new day starts
        {"sub_id": "S003", "submitted_at_utc": "2026-10-22T01:15:00+00:00"},
        {"sub_id": "S004", "submitted_at_utc": ""},                           # created, not uploaded
        {"sub_id": "S005", "submitted_at_utc": "2026-10-20T03:00:00+00:00"},
    ])
    assert submissions_today("synthetic") == 2
    # times stored with another zone are counted on the right UTC day (not by string prefix)
    _add_rows([{"sub_id": "S006", "submitted_at_utc": "2026-10-22T09:00:00+08:00"},   # 01:00 UTC on the 22nd
               {"sub_id": "S007", "submitted_at_utc": "2026-10-22T07:30:00+08:00"}])  # 23:30 UTC on the 21st
    assert submissions_today("synthetic") == 3


def test_next_sub_id_skips_ids_whose_file_exists():
    _add_rows([{"sub_id": "S001"}])
    paths.submissions_dir("synthetic").mkdir(parents=True, exist_ok=True)
    (paths.submissions_dir("synthetic") / "S002_made-elsewhere.csv").write_text("id,target\n", encoding="utf-8")
    assert next_sub_id("synthetic") == "S003"


@pytest.mark.synthetic_data
def test_record_normalises_submitted_at_to_utc():
    create(_run_with_test_pred(np.linspace(0, 1, N_TEST)), "a", source="synthetic")
    out = record("S001", source="synthetic", submitted_at_utc="2026-10-22T07:30+08:00")
    assert out["submitted_at_utc"] == "2026-10-21T23:30:00+00:00"
    with pytest.raises(ValueError, match="ISO"):
        record("S001", source="synthetic", submitted_at_utc="yesterday")


@pytest.mark.synthetic_data
def test_create_author_precedence_and_cv_format(monkeypatch):
    monkeypatch.delenv("SSC_AUTHOR")
    run = Run("by-a1", {}, author="A1", source="synthetic")
    run.save_array("test_pred", np.linspace(0, 1, N_TEST))
    run.finish(extra_summary={"auc_mean": 0.712345678})
    row = create(run.run_id, "a", source="synthetic")
    assert row["author"] == "A1"            # the run's author, not project.yaml default_author
    assert row["cv_auc_mean"] == "0.71235"  # same 5-decimal format as experiments.csv
    assert create(run.run_id, "b", author="B2", source="synthetic")["author"] == "B2"
    monkeypatch.setenv("SSC_AUTHOR", "C3")
    assert create(run.run_id, "c", source="synthetic")["author"] == "C3"


@pytest.mark.synthetic_data
def test_create_refuses_run_of_other_source():
    run = Run("raw-run", {}, source="raw")  # same flat archive under SSC_ARCHIVE_ROOT
    run.save_array("test_pred", np.linspace(0, 1, N_TEST))
    run.finish()
    with pytest.raises(ValueError, match="made on 'raw' data"):
        create(run.run_id, "mix", source="synthetic")


# ---------------------------------------------------------------- validate_file()
@pytest.fixture
def sample_frame():
    sample = load_sample_submission("synthetic")
    return pd.DataFrame({"id": sample["id"], "target": np.linspace(0, 1, len(sample))})


def _check(df: pd.DataFrame, tmp_path) -> list[str]:
    p = tmp_path / "candidate.csv"
    df.to_csv(p, index=False)
    return validate_file(p, "synthetic")


@pytest.mark.synthetic_data
def test_validate_file_accepts_good_file(sample_frame, tmp_path):
    assert _check(sample_frame, tmp_path) == []


@pytest.mark.synthetic_data
@pytest.mark.parametrize("mutation, expected", [
    ("wrong_names", "columns"),
    ("wrong_order", "columns"),
    ("extra_column", "columns"),
    ("nan", "NaN in target"),
    ("constant", "constant predictions"),
    ("out_of_range", "outside [0, 1]"),
    ("missing_row", "rows, expected"),
    ("shuffled_ids", "ids differ"),
])
def test_validate_file_detects_problems(sample_frame, tmp_path, mutation, expected):
    df = sample_frame.copy()
    if mutation == "wrong_names":
        df.columns = ["Id", "prediction"]
    elif mutation == "wrong_order":
        df = df[["target", "id"]]
    elif mutation == "extra_column":
        df["extra"] = 1
    elif mutation == "nan":
        df.loc[5, "target"] = np.nan
    elif mutation == "constant":
        df["target"] = 0.5
    elif mutation == "out_of_range":
        df.loc[0, "target"] = 1.5
    elif mutation == "missing_row":
        df = df.iloc[:-1]
    elif mutation == "shuffled_ids":
        df = df.sample(frac=1.0, random_state=0)
    problems = _check(df, tmp_path)
    assert problems, mutation
    assert any(expected in p for p in problems), problems


# ---------------------------------------------------------------- test-id checks, dry run
@pytest.mark.synthetic_data
def test_create_dry_run_writes_nothing_and_matches_the_real_file(isolated_archive):
    run_id = _run_with_test_pred(np.random.default_rng(1).normal(size=N_TEST))
    reg_path = paths.submissions_csv("synthetic")
    dry = create(run_id, "try", expected_public=0.7, source="synthetic", dry_run=True)
    assert dry["sub_id"] == "S001" and dry["file"] == f"submissions/S001_{run_id}.csv"
    assert not reg_path.exists() and list(paths.submissions_dir("synthetic").glob("*.csv")) == []
    real = create(run_id, "try", expected_public=0.7, source="synthetic")
    assert real["sub_id"] == dry["sub_id"] and real["sha256"] == dry["sha256"]  # same bytes as the dry run
    assert real["sha256"] == sha256(isolated_archive / real["file"])[:16]
    assert create(run_id, "next", source="synthetic", dry_run=True)["sub_id"] == "S002"
    assert read_registry(reg_path)["sub_id"].tolist() == ["S001"]


@pytest.mark.synthetic_data
def test_check_test_ids_and_create_refuse_misaligned_ids(monkeypatch):
    run_id = _run_with_test_pred(np.linspace(0, 1, N_TEST))
    run = load_run(run_id, "synthetic")
    assert submission.check_test_ids(run, "synthetic") == []

    sample = load_sample_submission("synthetic")
    swapped = sample.copy()
    swapped.loc[[0, 1], "id"] = swapped.loc[[1, 0], "id"].to_numpy()  # same ids, different order
    monkeypatch.setattr(submission, "load_sample_submission", lambda source=None: swapped)
    assert any("ids differ" in p for p in submission.check_test_ids(run, "synthetic"))
    with pytest.raises(ValueError, match="sample_submission"):
        create(run_id, "misaligned", source="synthetic")
    assert list(paths.submissions_dir("synthetic").glob("*.csv")) == []
    assert not paths.submissions_csv("synthetic").exists()


@pytest.mark.synthetic_data
def test_create_refuses_a_run_made_on_another_test_file():
    run = Run("old-test-file", {}, source="synthetic")
    run.save_array("test_pred", np.linspace(0, 1, N_TEST))
    run.meta["data_fingerprint"]["test.csv"] = "0123456789abcdef"  # as if test.csv had changed since
    run._write_json("meta.json", run.meta)
    run.finish()
    with pytest.raises(ValueError, match="test.csv changed since the run"):
        create(run.run_id, "stale", source="synthetic")


# ---------------------------------------------------------------- team daily limit
def test_team_daily_limit_from_project_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "project.yaml"
    monkeypatch.setattr(paths, "PROJECT_YAML", cfg)
    assert submission.team_daily_limit() == submission.TEAM_DAILY_LIMIT == 3  # file missing
    cfg.write_text("competition:\n  daily_submission_limit: 5\n", encoding="utf-8")
    assert submission.team_daily_limit() == 3  # Kaggle's limit is not the team's
    cfg.write_text("competition:\n  team_daily_limit: 2\n", encoding="utf-8")
    assert submission.team_daily_limit() == 2
    cfg.write_text("competition:\n  team_daily_limit: lots\n", encoding="utf-8")
    assert submission.team_daily_limit() == 3


def _make_submission(monkeypatch, *argv) -> int:
    """Run scripts/make_submission.py main() in-process; returns its exit code."""
    monkeypatch.syspath_prepend(str(paths.ROOT / "scripts"))
    import make_submission
    monkeypatch.setattr(sys, "argv", ["make_submission.py", *argv])
    try:
        return make_submission.main()
    except SystemExit as e:
        return e.code


@pytest.mark.synthetic_data
def test_make_submission_stops_at_the_team_limit(monkeypatch, capsys):
    run_id = _run_with_test_pred(np.linspace(0, 1, N_TEST))
    limit = submission.team_daily_limit()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _add_rows([{"sub_id": f"S{i:03d}", "submitted_at_utc": now} for i in range(1, limit + 1)])
    args = ["--run", run_id, "--desc", "one more", "--expected-public", "0.7", "--data", "synthetic"]
    reg_path = paths.submissions_csv("synthetic")

    assert _make_submission(monkeypatch, *args) == 1
    assert "team daily limit reached" in capsys.readouterr().err
    assert _make_submission(monkeypatch, *args, "--dry-run") == 1
    assert "would stop" in capsys.readouterr().out
    assert len(read_registry(reg_path)) == limit
    assert list(paths.submissions_dir("synthetic").glob("*.csv")) == []

    assert _make_submission(monkeypatch, *args, "--force-limit") == 0
    assert "--force-limit" in capsys.readouterr().out
    assert read_registry(reg_path)["sub_id"].tolist()[-1] == f"S{limit + 1:03d}"
    assert len(list(paths.submissions_dir("synthetic").glob("*.csv"))) == 1


@pytest.mark.synthetic_data
def test_make_submission_dry_run_prints_and_writes_nothing(monkeypatch, capsys):
    run_id = _run_with_test_pred(np.random.default_rng(2).normal(size=N_TEST))
    args = ["--run", run_id, "--desc", "dry", "--expected-public", "0.7", "--data", "synthetic"]
    assert _make_submission(monkeypatch, *args, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "S001" in out and "(not written)" in out
    assert not paths.submissions_csv("synthetic").exists()
    assert list(paths.submissions_dir("synthetic").glob("*.csv")) == []
    sha_dry = next(line for line in out.splitlines() if line.startswith("sha256")).split(":")[1].split()[0]
    assert _make_submission(monkeypatch, *args) == 0
    row = read_registry(paths.submissions_csv("synthetic")).iloc[0]
    assert row["sub_id"] == "S001" and row["sha256"] == sha_dry
