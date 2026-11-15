"""Experiment archive.

Every evaluated idea becomes one immutable run folder plus one row in the
registry CSV:

    experiments/<run_id>/
        config.yaml        exact config used (copied, never edited afterwards)
        meta.json          who/when/git commit/package versions/data checksums/command, the machine
                           (compute, incl. CPU model), the BLAS/OpenMP libraries (threadpools) and,
                           once save_cv() or record_estimator() sees the model, its full
                           get_params(deep=True) (estimator)
        fold_scores.csv    AUC of every (repeat, fold)
        summary.json       aggregated scores
        oof.npy            out-of-fold scores, shape (n_repeats, 250)
        test_pred.npy      test scores, shape (19750,), if produced
        notes.md           hypothesis before the run; conclusion added after reading results
        ...                any extra tables / figures the run saved

    registry/experiments.csv   one row per run, append-only

run_id = <YYYYMMDD-HHMMSS>_<slug>. Run ids and created_at always use the fixed archive
clock ARCHIVE_TZ (see below), whatever the time zone of the machine or Colab VM.

Registry CSVs may be open in Excel, which locks the file: writers retry for a
while and then raise RegistryLockedError with a clear message instead of a raw
PermissionError traceback.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import paths
from .data import data_fingerprint

EXPERIMENT_COLUMNS = [
    "run_id", "created_at", "kind", "author", "name", "data_source",
    "model", "selector", "scaler", "cv", "cv_seed",
    "auc_mean", "auc_std_fold", "auc_std_repeat", "oof_auc_mean",
    "split_sig", "git_commit", "git_dirty", "hypothesis", "status",
]

PACKAGES = ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "joblib", "threadpoolctl")

# Fixed offset (HK has no DST); zoneinfo would need the tzdata package on Windows.
# One fixed archive clock for every machine (laptops in Japan, Colab VMs in UTC), so run_ids sort
# correctly wherever they were created. Deadlines: Kaggle closes 2026-11-30 00:59:59 JST (= 11-29 23:59 HKT).
ARCHIVE_TZ = timezone(timedelta(hours=9), "JST")
COMPUTE_ENV = "SSC_COMPUTE"  # JSON describing the machine, set by colab/remote.py on the VM
LOCK_RETRY_SECONDS = 15.0  # how long writers wait for a registry file locked by another program
GIT_COMMIT_ENV, GIT_DIRTY_ENV = "SSC_GIT_COMMIT", "SSC_GIT_DIRTY"  # set by colab/remote.py on the VM


class RegistryLockedError(PermissionError):
    """A registry file stayed locked (typically open in Excel) for LOCK_RETRY_SECONDS."""


def now_iso() -> str:
    return datetime.now(ARCHIVE_TZ).isoformat(timespec="seconds")


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-").lower()
    return (s or "run")[:max_len]


def git_info() -> dict:
    """Commit of the code that runs. On a remote machine the code arrives without .git,
    so the launcher passes the local state in SSC_GIT_COMMIT / SSC_GIT_DIRTY."""
    if os.environ.get(GIT_COMMIT_ENV):
        dirty = os.environ.get(GIT_DIRTY_ENV, "").strip().lower()
        return {"commit": os.environ[GIT_COMMIT_ENV],
                "dirty": True if dirty == "true" else False if dirty == "false" else None,
                "branch": None, "from_env": True}

    def _git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = _git("rev-parse", "--short=10", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit or "no-commit", "dirty": bool(status) if status is not None else None,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD")}


def compute_info() -> dict:
    """Where the run executed: the Colab job descriptor passed by colab/remote.py, else this machine."""
    raw = os.environ.get(COMPUTE_ENV)
    if raw:
        try:
            return {**json.loads(raw), "node": platform.node(), "cpus": os.cpu_count(), "cpu_model": cpu_model()}
        except json.JSONDecodeError:
            return {"host": "unknown", "raw": raw}
    return {"host": "local", "node": platform.node(), "cpus": os.cpu_count(), "cpu_model": cpu_model()}


def cpu_model() -> str:
    """CPU brand string, e.g. 'Intel(R) Xeon(R) CPU @ 2.20GHz' (platform.processor() is often just 'x86_64')."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return platform.processor() or "unknown"


def threadpools() -> list[dict]:
    """BLAS / OpenMP libraries loaded in this process (threadpoolctl.threadpool_info): API, implementation,
    version, CPU kernel (architecture) and thread count. Different OpenBLAS builds/kernels change the
    floating-point summation order: the S003 GMM differs by ~1e-11 between the laptop and Colab."""
    try:
        import scipy.linalg  # noqa: F401 - load scipy's own BLAS too, so both copies are listed
        from threadpoolctl import threadpool_info
        keys = ("user_api", "internal_api", "version", "architecture", "threading_layer", "num_threads")
        return [{**{k: d.get(k) for k in keys}, "library": Path(d.get("filepath") or "").name}
                for d in threadpool_info()]
    except Exception as e:  # noqa: BLE001 - metadata must never stop a run
        return [{"error": f"{type(e).__name__}: {e}"}]


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            out[name] = None
    return out


def project_author() -> str | None:
    """Default author: `default_author` in project.yaml (each member sets their own copy of
    the env var SSC_AUTHOR instead if several people share one machine)."""
    if os.environ.get("SSC_AUTHOR"):
        return os.environ["SSC_AUTHOR"]
    if paths.PROJECT_YAML.exists():
        cfg = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}
        return cfg.get("default_author")
    return None


def retry_locked(fn, path: Path, seconds: float | None = None):
    """Call fn(); while it raises PermissionError (file locked by another program, e.g.
    Excel) retry for `seconds`, then raise RegistryLockedError naming the file."""
    limit = LOCK_RETRY_SECONDS if seconds is None else seconds
    deadline, warned = time.monotonic() + limit, False
    while True:
        try:
            return fn()
        except PermissionError as e:
            if time.monotonic() >= deadline:
                raise RegistryLockedError(f"{path} is locked by another program (open in Excel?). "
                                          f"Close it and run the command again. ({e})") from None
            if not warned:
                print(f"note: {path} is locked by another program (Excel?); retrying for {limit:.0f} s ...",
                      file=sys.stderr, flush=True)
                warned = True
            time.sleep(0.5)


def append_csv_row(path: Path, columns: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {c: ("" if row.get(c) is None else row.get(c)) for c in columns}

    def _append():
        new = not path.exists() or path.stat().st_size == 0
        glue = False
        if not new:  # a hand-edited file may have lost its final newline: never join two rows
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                glue = f.read(1) != b"\n"
        with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            if glue:
                f.write("\r\n")
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(clean)

    retry_locked(_append, path)


def read_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:  # a program holding the file with share mode 0 blocks reads too: wait like the writers do
        return retry_locked(lambda: pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False), path)
    except pd.errors.ParserError as e:
        raise ValueError(f"cannot parse {path}: {e}. Check the last lines of the file: a hand edit or a "
                         "merge may have joined two rows or broken the quoting.") from None


class Run:
    """One archived experiment. Use as a context manager or call finish()."""

    def __init__(self, name: str, config: dict, kind: str = "experiment", author: str | None = None,
                 source: str | None = None, hypothesis: str = ""):
        self.source = paths.resolve_source(source)
        paths.ensure_archive(self.source)
        self.name = name
        self.kind = kind
        self.config = config
        self.author = author or config.get("author") or project_author() or "unknown"
        self.hypothesis = hypothesis or config.get("hypothesis", "")
        stamp = datetime.now(ARCHIVE_TZ).strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{slugify(name)}"
        run_id, i = base, 1
        while (paths.experiments_dir(self.source) / run_id).exists():
            i += 1
            run_id = f"{base}-{i}"
        self.run_id = run_id
        self.dir = paths.experiments_dir(self.source) / run_id
        self.dir.mkdir(parents=True)
        self.summary: dict = {}
        self._finished = False
        self.meta = {
            "run_id": run_id, "kind": kind, "name": name, "author": self.author,
            "created_at": now_iso(), "data_source": self.source,
            "data_fingerprint": data_fingerprint(self.source),
            "git": git_info(), "packages": package_versions(), "threadpools": threadpools(),
            "platform": platform.platform(), "compute": compute_info(), "command": " ".join(sys.argv),
        }
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        self._write_json("meta.json", self.meta)
        (self.dir / "notes.md").write_text(
            f"# {run_id}\n\n- author: {self.author}\n- kind: {kind}\n\n## Hypothesis (written before running)\n\n"
            f"{self.hypothesis or '(none given)'}\n\n## Result\n\n(see summary.json)\n\n"
            "## Conclusion (write after reading the results)\n\n", encoding="utf-8")

    # ---- saving helpers -------------------------------------------------
    def _write_json(self, name: str, obj) -> None:
        with open(self.dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)

    def record_estimator(self, estimator) -> dict:
        """Store estimator.get_params(deep=True) (JSON-safe) in meta.json under "estimator", so the
        archive has every hyperparameter, including the defaults the config did not name."""
        self.meta["estimator"] = estimator_params(estimator)
        self._write_json("meta.json", self.meta)
        return self.meta["estimator"]

    def save_cv(self, cv, estimator=None) -> dict:
        """Save a validation.CVResult and return its summary. The model's parameters are recorded
        (record_estimator) from `estimator`, else from the template cross_validate kept in cv.extra."""
        est = estimator if estimator is not None else (getattr(cv, "extra", None) or {}).get("estimator")
        if est is not None:
            self.record_estimator(est)
        cv.fold_scores.to_csv(self.dir / "fold_scores.csv", index=False)
        np.save(self.dir / "oof.npy", cv.oof)
        if cv.test_pred is not None:
            np.save(self.dir / "test_pred.npy", cv.test_pred)
        self.summary.update(cv.summary())
        self.summary["scheme"] = cv.scheme
        return self.summary

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        p = self.dir / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def save_array(self, name: str, arr) -> Path:
        p = self.dir / f"{name}.npy"
        np.save(p, np.asarray(arr))
        return p

    def save_figure(self, name: str, fig) -> Path:
        p = self.dir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        return p

    # ---- completion -----------------------------------------------------
    def finish(self, status: str = "done", extra_summary: dict | None = None,
               describe_spec: dict | None = None) -> dict:
        """Write summary.json, then append the registry row.

        describe_spec: pipeline spec for the registry's model/selector/scaler columns when it
        differs from config['pipeline'] (e.g. an extension whose test_pred.npy comes from a
        derived pipeline); config.yaml on disk is not affected."""
        if extra_summary:
            self.summary.update(extra_summary)
        self.summary["status"] = status
        self.summary["finished_at"] = now_iso()
        self._write_json("summary.json", self.summary)
        self._finished = True
        row = self._registry_row(status, describe_spec)
        try:
            append_csv_row(paths.experiments_csv(self.source), EXPERIMENT_COLUMNS, row)
        except RegistryLockedError as e:  # the run is complete; only its registry row is missing
            self._write_json("registry_row.json", row)
            raise RegistryLockedError(
                f"{e}\nThe run itself is complete ({self.dir}, status {status!r}); its registry row was "
                "saved to registry_row.json. After closing the file, append it with:\n"
                "  .venv\\Scripts\\python.exe -c \"from ssc.tracking import append_saved_row; "
                f"append_saved_row(r'{self.dir}')\"") from None
        return self.summary

    def _registry_row(self, status: str, describe_spec: dict | None = None) -> dict:
        pipe = describe_spec or self.config.get("pipeline") or {}
        from .models import describe  # local import keeps tracking importable without sklearn models
        desc = describe(pipe) if pipe.get("model") else {"model": "", "selector": "", "scaler": ""}
        scheme = self.summary.get("scheme") or self.config.get("cv") or {}
        cv_label = ""
        if scheme:
            from .validation import Scheme
            sch = Scheme.from_dict(scheme)
            cv_label, cv_seed = sch.label(), sch.seed
        else:
            cv_seed = ""
        return {
            "run_id": self.run_id, "created_at": self.meta["created_at"], "kind": self.kind,
            "author": self.author, "name": self.name, "data_source": self.source,
            **desc, "cv": cv_label, "cv_seed": cv_seed,
            "auc_mean": _fmt(self.summary.get("auc_mean")),
            "auc_std_fold": _fmt(self.summary.get("auc_std_fold")),
            "auc_std_repeat": _fmt(self.summary.get("auc_std_repeat")),
            "oof_auc_mean": _fmt(self.summary.get("oof_auc_mean")),
            "split_sig": self.summary.get("split_signature", ""),
            "git_commit": self.meta["git"]["commit"], "git_dirty": self.meta["git"]["dirty"],
            "hypothesis": self.hypothesis.replace("\n", " ")[:300], "status": status,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # finish() only once: an error after it (e.g. a locked registry) keeps the written summary
        if not self._finished:
            self.finish(status=f"failed: {exc_type.__name__}" if exc_type is not None else "done")
        return False


def append_saved_row(run_dir) -> None:
    """Append the registry row that Run.finish saved as registry_row.json because the
    registry was locked, then delete the json. Refuses if the run is already registered."""
    d = Path(run_dir)
    row = json.loads((d / "registry_row.json").read_text(encoding="utf-8"))
    path = paths.experiments_csv(row.get("data_source") or None)
    reg = read_registry(path)
    if not reg.empty and row["run_id"] in set(reg["run_id"]):
        raise ValueError(f"{row['run_id']} is already in {path}")
    append_csv_row(path, EXPERIMENT_COLUMNS, row)
    (d / "registry_row.json").unlink()
    print(f"appended {row['run_id']} to {path}")


def _fmt(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.5f}"


def estimator_params(estimator) -> dict:
    """{"class": module.Class, "params": get_params(deep=True)} with JSON-safe values: nested
    estimators and other objects become their repr, numpy scalars plain numbers, NaN/inf strings."""
    cls = type(estimator)
    out = {"class": f"{cls.__module__}.{cls.__qualname__}"}
    try:
        params = estimator.get_params(deep=True)
    except Exception as e:  # noqa: BLE001 - not an sklearn estimator: keep its repr
        return {**out, "repr": _json_safe(estimator), "error": f"get_params failed: {type(e).__name__}"}
    out["params"] = {str(k): _json_safe(v) for k, v in sorted(params.items(), key=lambda kv: str(kv[0]))}
    return out


def _json_safe(v):
    if v is None or isinstance(v, (bool, np.bool_)):
        return None if v is None else bool(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else str(float(v))
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, np.ndarray) and v.size <= 1000:
        return _json_safe(v.tolist())
    return repr(v)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_run(run_id: str, source: str | None = None) -> dict:
    d = paths.experiments_dir(source) / run_id
    if not d.exists():
        raise FileNotFoundError(f"run {run_id} not found in {d.parent}")
    out = {"dir": d,
           "config": yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")),
           "meta": json.loads((d / "meta.json").read_text(encoding="utf-8"))}
    if (d / "summary.json").exists():
        out["summary"] = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    if (d / "fold_scores.csv").exists():
        out["fold_scores"] = pd.read_csv(d / "fold_scores.csv")
    if (d / "test_pred.npy").exists():
        out["test_pred"] = np.load(d / "test_pred.npy")
    return out
