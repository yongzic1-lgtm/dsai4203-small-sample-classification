"""Canonical project paths. Every other module gets locations from here.

Two archives exist side by side:
  * the real archive (repo root: experiments/, registry/, submissions/, reports/)
    used only with the Kaggle data (source "raw");
  * a throw-away sandbox (sandbox/...) used with synthetic data, so pipeline
    tests can never pollute the real records.
Setting SSC_ARCHIVE_ROOT redirects the archive (pytest uses a temp dir).
Note: under SSC_ARCHIVE_ROOT both sources share ONE flat archive (one registry),
so sub ids and the daily upload count mix raw and synthetic rows there. Scripts
filter registry rows by data_source and refuse runs of the other source; use a
separate SSC_ARCHIVE_ROOT per source when that matters. (The flat layout is kept
because colab/remote.py reads the VM's job archive as <root>/experiments.)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_SYNTHETIC = DATA / "synthetic"
CONFIGS = ROOT / "configs"
SANDBOX = ROOT / "sandbox"
PROJECT_YAML = ROOT / "project.yaml"

DATA_SOURCE_ENV = "SSC_DATA"            # "raw" (default) or "synthetic"
ARCHIVE_ROOT_ENV = "SSC_ARCHIVE_ROOT"   # optional override of the archive root

SOURCES = ("raw", "synthetic")


def resolve_source(source: str | None = None) -> str:
    source = source or os.environ.get(DATA_SOURCE_ENV, "raw")
    if source not in SOURCES:
        raise ValueError(f"unknown data source {source!r}; expected one of {SOURCES}")
    return source


def data_dir(source: str | None = None) -> Path:
    return DATA_RAW if resolve_source(source) == "raw" else DATA_SYNTHETIC


def archive_root(source: str | None = None) -> Path:
    override = os.environ.get(ARCHIVE_ROOT_ENV)
    if override:
        return Path(override)
    return ROOT if resolve_source(source) == "raw" else SANDBOX


def experiments_dir(source: str | None = None) -> Path:
    return archive_root(source) / "experiments"


def registry_dir(source: str | None = None) -> Path:
    return archive_root(source) / "registry"


def submissions_dir(source: str | None = None) -> Path:
    return archive_root(source) / "submissions"


def figures_dir(source: str | None = None) -> Path:
    return archive_root(source) / "reports" / "figures"


def tables_dir(source: str | None = None) -> Path:
    return archive_root(source) / "reports" / "tables"


def experiments_csv(source: str | None = None) -> Path:
    return registry_dir(source) / "experiments.csv"


def submissions_csv(source: str | None = None) -> Path:
    return registry_dir(source) / "submissions.csv"


def ensure_archive(source: str | None = None) -> None:
    for p in (experiments_dir(source), registry_dir(source), submissions_dir(source),
              figures_dir(source), tables_dir(source)):
        p.mkdir(parents=True, exist_ok=True)
