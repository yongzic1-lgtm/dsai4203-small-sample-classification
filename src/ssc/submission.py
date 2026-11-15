"""Kaggle submissions: build the CSV from a run, and keep the submission log.

Flow (see docs/ARCHIVE_RULES.md, section "Submission"):
  1. create(run_id, description, expected_public=...)  -> submissions/S###_<run_id>.csv + registry row
     The expected public score is written BEFORE uploading (pre-registration).
     create(..., dry_run=True) runs every check and returns the row it would write, writing nothing.
     The team uploads at most team_daily_limit() files per Kaggle day (scripts/make_submission.py stops there).
  2. upload the file to Kaggle (web page or `kaggle competitions submit`)
  3. record(sub_id, public=..., submitted_at=...)        -> fills the LB fields of that row
  4. after the deadline: record(sub_id, private=...) for every submission

registry/submissions.csv rows are never deleted; only the LB/selection fields
are filled in afterwards by record().
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from . import paths
from .data import ID_COL, data_fingerprint, load_sample_submission, sha256
from .tracking import _fmt, append_csv_row, load_run, now_iso, project_author, read_registry, retry_locked

SUBMISSION_COLUMNS = [
    "sub_id", "created_at", "author", "run_id", "file", "sha256", "description",
    "cv_auc_mean", "cv_auc_std_repeat", "expected_public", "score_transform",
    "submitted_at_utc", "public_lb", "private_lb", "selected_final", "notes",
]
RECORDABLE = ("submitted_at_utc", "public_lb", "private_lb", "selected_final", "notes")
DAILY_LIMIT = 5  # competition rule 3; Kaggle's day resets at 00:00 UTC (08:00 HKT)
TEAM_DAILY_LIMIT = 3  # team rule (README golden rule 4): at most 3 uploads per Kaggle day
COMPETITION = "small-sample-classification-challenge"


def team_daily_limit() -> int:
    """Uploads the team allows per Kaggle day: project.yaml competition.team_daily_limit if set,
    else TEAM_DAILY_LIMIT. Stricter than Kaggle's own limit (competition.daily_submission_limit)."""
    try:
        cfg = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}
        value = (cfg.get("competition") or {}).get("team_daily_limit")
        return TEAM_DAILY_LIMIT if value in (None, "") else int(value)
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return TEAM_DAILY_LIMIT


def check_test_ids(run: dict, source: str | None = None) -> list[str]:
    """The rows of a run's test_pred.npy are the rows of test.csv in file order (ssc.data.load), and
    create() labels them with sample_submission's ids. That is only right if test.csv is the file the
    run used (data fingerprint) and its ids equal sample_submission's ids in the same order.
    Returns the problems found (empty list = consistent)."""
    problems = []
    used = (run["meta"].get("data_fingerprint") or {}).get("test.csv")
    now = data_fingerprint(source).get("test.csv")
    if used and now and used != now:
        problems.append(f"test.csv changed since the run (sha256 prefix {used} at run time, {now} now)")
    ids_test = pd.read_csv(paths.data_dir(source) / "test.csv", usecols=[ID_COL])[ID_COL].to_numpy()
    ids_sample = load_sample_submission(source)[ID_COL].to_numpy()
    if len(ids_test) != len(ids_sample) or not np.array_equal(ids_test, ids_sample):
        problems.append("test.csv ids differ from sample_submission ids (values or order)")
    if "test_pred" in run and len(run["test_pred"]) != len(ids_test):
        problems.append(f"{len(run['test_pred'])} predictions, test.csv has {len(ids_test)} rows")
    return problems


def to_unit_interval(scores: np.ndarray) -> tuple[np.ndarray, str]:
    """Kaggle wants values in [0, 1]. AUC depends only on the ordering, so
    out-of-range scores (decision functions) are mapped to rank/n."""
    s = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("predictions contain NaN or inf")
    if s.min() >= 0.0 and s.max() <= 1.0:
        return s, "none"
    return rankdata(s) / len(s), "rank"


def next_sub_id(source: str | None = None) -> str:
    """Next free S### in the registry; ids whose file already exists in submissions/
    (e.g. created on another machine, registry not yet pulled) are skipped too."""
    reg = read_registry(paths.submissions_csv(source))
    nums = [] if reg.empty else [int(s[1:]) for s in reg["sub_id"] if s.startswith("S") and s[1:].isdigit()]
    n = max(nums) + 1 if nums else 1
    sub_dir = paths.submissions_dir(source)
    while sub_dir.exists() and any(sub_dir.glob(f"S{n:03d}_*")):
        n += 1
    return f"S{n:03d}"


def to_utc_iso(value) -> str:
    """An ISO time with or without zone (no zone = UTC) -> 'YYYY-MM-DDTHH:MM:SS+00:00'."""
    dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def validate_file(path, source: str | None = None) -> list[str]:
    """Check a submission CSV against sample_submission.csv. Empty list = valid."""
    problems = []
    sub = pd.read_csv(path)
    sample = load_sample_submission(source)
    if list(sub.columns) != ["id", "target"]:
        problems.append(f"columns {list(sub.columns)} != ['id', 'target']")
        return problems
    if len(sub) != len(sample):
        problems.append(f"{len(sub)} rows, expected {len(sample)}")
    elif not np.array_equal(sub["id"].to_numpy(), sample["id"].to_numpy()):
        problems.append("ids differ from sample_submission (values or order)")
    if sub["target"].isna().any():
        problems.append("NaN in target")
    elif sub["target"].min() < 0 or sub["target"].max() > 1:
        problems.append("target outside [0, 1]")
    if sub["target"].nunique() < 2:
        problems.append("constant predictions (AUC would be 0.5)")
    return problems


def create(run_id: str, description: str, expected_public: float | None = None,
           author: str | None = None, source: str | None = None, notes: str = "",
           dry_run: bool = False) -> dict:
    """Write submissions/S###_<run_id>.csv and its registry row; returns the row.
    dry_run=True: the same checks on a temporary copy of the file (same bytes, so the same sha256),
    then nothing is kept: no file in submissions/, no registry row, no archive folders."""
    source = paths.resolve_source(source)
    if not dry_run:
        paths.ensure_archive(source)
    run = load_run(run_id, source)
    run_source = run["meta"].get("data_source")
    if run_source not in (None, source):  # possible when SSC_ARCHIVE_ROOT holds both sources
        raise ValueError(f"run {run_id} was made on {run_source!r} data, not {source!r}")
    if "test_pred" not in run:
        raise FileNotFoundError(f"run {run_id} has no test_pred.npy (was test_pred: none?)")
    problems = check_test_ids(run, source)
    if problems:
        raise ValueError(f"run {run_id} does not line up with sample_submission: {problems}")
    sample = load_sample_submission(source)
    values, transform = to_unit_interval(run["test_pred"])
    if len(values) != len(sample):
        raise ValueError(f"{len(values)} predictions, sample_submission has {len(sample)} rows")

    sub_id = next_sub_id(source)
    out = paths.submissions_dir(source) / f"{sub_id}_{run_id}.csv"
    if out.exists():
        raise FileExistsError(out)
    frame = pd.DataFrame({"id": sample["id"].to_numpy(), "target": values})
    with tempfile.TemporaryDirectory(prefix="ssc_dry_run_") as tmp:
        target = Path(tmp) / out.name if dry_run else out
        frame.to_csv(target, index=False, float_format="%.8f")
        problems = validate_file(target, source)
        if problems:
            target.unlink()
            raise ValueError(f"invalid submission: {problems}")
        digest = sha256(target)[:16]

    summary = run.get("summary", {})
    row = {
        "sub_id": sub_id, "created_at": now_iso(),
        # explicit author > $SSC_AUTHOR > the run's author > project.yaml default_author
        "author": (author or os.environ.get("SSC_AUTHOR") or run["meta"].get("author")
                   or project_author() or "unknown"),
        "run_id": run_id, "file": out.relative_to(paths.archive_root(source)).as_posix(),
        "sha256": digest, "description": description,
        "cv_auc_mean": _fmt(summary.get("auc_mean")), "cv_auc_std_repeat": _fmt(summary.get("auc_std_repeat")),
        "expected_public": expected_public, "score_transform": transform, "notes": notes,
    }
    if not dry_run:
        append_csv_row(paths.submissions_csv(source), SUBMISSION_COLUMNS, row)
    return row


def record(sub_id: str, source: str | None = None, **fields) -> dict:
    """Fill leaderboard / selection fields of an existing submission row.
    submitted_at_utc is normalised to UTC (a value without zone is taken as UTC)."""
    bad = set(fields) - set(RECORDABLE)
    if bad:
        raise ValueError(f"only {RECORDABLE} may be recorded after creation, got {sorted(bad)}")
    if fields.get("submitted_at_utc"):
        try:
            fields["submitted_at_utc"] = to_utc_iso(fields["submitted_at_utc"])
        except ValueError:
            raise ValueError(f"submitted_at_utc {fields['submitted_at_utc']!r} is not an ISO time") from None
    path = paths.submissions_csv(source)
    reg = read_registry(path)
    if reg.empty or sub_id not in set(reg["sub_id"]):
        raise KeyError(f"{sub_id} not in {path}")
    for k, v in fields.items():
        if v is not None:
            reg.loc[reg["sub_id"] == sub_id, k] = str(v)
    tmp = path.with_suffix(".tmp")
    reg[SUBMISSION_COLUMNS].to_csv(tmp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    retry_locked(lambda: tmp.replace(path), path)  # fails while Excel holds the registry open
    return reg.loc[reg["sub_id"] == sub_id].iloc[0].to_dict()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def submissions_today(source: str | None = None) -> int:
    """Uploads recorded for the current Kaggle day (UTC)."""
    reg = read_registry(paths.submissions_csv(source))
    if reg.empty:
        return 0
    # parse instead of matching the string prefix, so a time stored with another zone still counts right
    ts = pd.to_datetime(reg["submitted_at_utc"], utc=True, errors="coerce", format="ISO8601")
    return int((ts.dt.date == datetime.now(timezone.utc).date()).sum())
