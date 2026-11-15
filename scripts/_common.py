"""Shared helpers for the command-line scripts in scripts/ (not part of the ssc package).

Scripts are run from the project root as
    .venv\\Scripts\\python.exe scripts\\<name>.py ...
so this directory is on sys.path and `import _common` works.

Console output must stay ASCII (Windows consoles may be GBK or cp1252); text
that comes from data files (names, hypotheses) is escaped instead of crashing.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import yaml

from ssc import paths

PY = r".venv\Scripts\python.exe"        # used in printed "next step" hints
SCRIPTS = "scripts"


def setup_console() -> None:
    """Never crash on a character the console encoding cannot show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def die(msg: str, code: int = 1) -> None:
    sys.stdout.flush()
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def run_main(main) -> int:
    """Run a script's main(); a registry locked by another program (Excel) ends with a
    clear message instead of a traceback."""
    from ssc.tracking import RegistryLockedError
    try:
        return main()
    except RegistryLockedError as e:
        die(str(e), 1)
        return 1  # unreachable


def resolve_path(p: Path, what: str = "file") -> Path:
    """A relative path is taken from the current folder, else from the project root."""
    if p.exists():
        return p
    if not p.is_absolute() and (paths.ROOT / p).exists():
        return paths.ROOT / p
    where = "" if p.is_absolute() else (f" (looked relative to the current folder {Path.cwd()} "
                                        f"and to the project root {paths.ROOT})")
    die(f"{what} {p} not found{where}", 2)
    return p  # unreachable


def banner(lines: list[str] | str, char: str = "#") -> None:
    """Loud multi-line warning that is hard to miss in a scrolling console."""
    if isinstance(lines, str):
        lines = [lines]
    width = max(60, max(len(s) for s in lines) + 4)
    print(char * width)
    for s in lines:
        print(f"{char} {s}")
    print(char * width)


def hint(script: str, args: str = "") -> str:
    return f"{PY} {SCRIPTS}\\{script} {args}".rstrip()


# ---- project.yaml ------------------------------------------------------------

def load_project() -> dict:
    if not paths.PROJECT_YAML.exists():
        die(f"{paths.PROJECT_YAML} not found")
    with open(paths.PROJECT_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def official_cv_dict() -> dict:
    cv = load_project().get("official_cv")
    if not cv:
        die("project.yaml has no official_cv block")
    return dict(cv)


def official_scheme():
    from ssc.validation import Scheme  # lazy: ssc.validation pulls in sklearn/scipy (slow --help)
    return Scheme.from_dict(official_cv_dict())


def daily_limit() -> int:
    return int((load_project().get("competition") or {}).get("daily_submission_limit", 5))


def public_fraction() -> float:
    return float((load_project().get("competition") or {}).get("public_fraction", 0.02))


# ---- argparse ------------------------------------------------------------------

def add_data_arg(parser: argparse.ArgumentParser, choices=paths.SOURCES) -> None:
    parser.add_argument("--data", choices=choices, default=None,
                        help="data source: raw = Kaggle data + real archive, synthetic = mock data + "
                             "sandbox archive (default: $SSC_DATA or raw)")


def data_flag(source: str) -> str:
    """The --data part of a printed hint (empty for the default source)."""
    return " --data synthetic" if source == "synthetic" else ""


def print_archive(source: str) -> None:
    import os
    root = paths.archive_root(source)
    note = f" (from ${paths.ARCHIVE_ROOT_ENV})" if os.environ.get(paths.ARCHIVE_ROOT_ENV) else ""
    print(f"data: {source}   archive: {root}{note}")


# ---- YAML values / dotted overrides ------------------------------------------

# PyYAML follows YAML 1.1, where "1e-3" (no dot) is a STRING, not a float.
_SCI = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def coerce_numbers(obj):
    """Recursively turn strings such as '1e-3' into floats."""
    if isinstance(obj, dict):
        return {k: coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_numbers(v) for v in obj]
    if isinstance(obj, str) and _SCI.match(obj.strip()):
        return float(obj)
    return obj


def parse_value(text: str):
    """Value of a --override key=value, parsed like YAML (0.5, true, null, [1, 2], {a: 1})."""
    try:
        return coerce_numbers(yaml.safe_load(text))
    except yaml.YAMLError as e:
        raise ValueError(f"cannot parse value {text!r}: {e}") from None


def apply_override(cfg: dict, item: str) -> tuple[str, object]:
    """Apply one 'a.b.c=value' override in place. Missing intermediate dicts are created."""
    if "=" not in item:
        raise ValueError(f"override {item!r} is not of the form key=value")
    key, raw = item.split("=", 1)
    key = key.strip()
    parts = key.split(".")
    if not key or any(not p for p in parts):
        raise ValueError(f"bad override key {key!r}")
    value = parse_value(raw)
    node = cfg
    for i, p in enumerate(parts[:-1]):
        if node.get(p) is None:
            node[p] = {}
        if not isinstance(node[p], dict):
            raise ValueError(f"override {key!r}: {'.'.join(parts[:i + 1])} is not a mapping")
        node = node[p]
    node[parts[-1]] = value
    return key, value


# ---- runs ------------------------------------------------------------------------

def resolve_run_id(run_id: str, source: str) -> str:
    """Exact run id, or a unique prefix of one in the source's experiments folder."""
    exp = paths.experiments_dir(source)
    if (exp / run_id).is_dir():
        return run_id
    matches = sorted(p.name for p in exp.glob(f"{run_id}*") if p.is_dir()) if exp.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        die(f"run id prefix {run_id!r} is ambiguous: {', '.join(matches[:6])}", 2)
    die(f"run {run_id!r} not found in {exp}", 2)
    return ""  # unreachable


def check_run_source(run: dict, source: str) -> None:
    """Refuse to mix sources (possible when $SSC_ARCHIVE_ROOT points both at one folder)."""
    run_source = run["meta"].get("data_source")
    if run_source and run_source != source:
        die(f"run {run['meta'].get('run_id')} was made on data source {run_source!r} but "
            f"--data is {source!r}; pass --data {run_source}", 2)


# ---- formatting ----------------------------------------------------------------

def fnum(x, nd: int = 4, signed: bool = False) -> str:
    """Format a number (or a numeric string from a registry CSV); blanks stay blank."""
    if x is None or x == "":
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def to_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rel(path: Path) -> str:
    """Path relative to the project root when possible (shorter console lines)."""
    try:
        return str(Path(path).resolve().relative_to(paths.ROOT))
    except ValueError:
        return str(path)
