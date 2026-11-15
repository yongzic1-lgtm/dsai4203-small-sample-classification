"""Check the competition data before any experiment (read-only).

Runs ssc.data.verify (checksums from SHA256SUMS.txt, shapes, columns, NaNs, ids,
sample_submission order) and prints ssc.data.summary for each source.

    .venv\\Scripts\\python.exe scripts\\verify_data.py                 # data/raw
    .venv\\Scripts\\python.exe scripts\\verify_data.py --data synthetic
    .venv\\Scripts\\python.exe scripts\\verify_data.py --data all      # raw + synthetic if present

Exit code 0 = all checks passed, 1 = at least one problem.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import _common as C
from ssc import data, paths


def synthetic_extra_checks() -> list[str]:
    """Files only the mock data has: hidden test labels and the ground truth."""
    d = paths.DATA_SYNTHETIC
    problems = []
    for name in ("test_labels.csv", "truth.csv"):
        if not (d / name).exists():
            problems.append(f"{name} missing (re-create with scripts/make_synthetic.py --force)")
    if (d / "test_labels.csv").exists() and (d / "test.csv").exists():
        labels = pd.read_csv(d / "test_labels.csv")
        test_ids = pd.read_csv(d / "test.csv", usecols=["id"])["id"]
        if list(labels.columns) != ["id", "target", "public"]:
            problems.append(f"test_labels.csv columns {list(labels.columns)} != ['id', 'target', 'public']")
        elif not labels["id"].equals(test_ids):
            problems.append("test_labels.csv ids differ from test.csv ids")
    return problems


def check(source: str) -> list[str]:
    print(f"== {source}: {paths.data_dir(source)}")
    try:
        problems = data.verify(source)
    except FileNotFoundError as e:
        return [str(e)]
    if source == "synthetic":
        problems += synthetic_extra_checks()
    try:
        summ = data.summary(data.load(source))
    except (FileNotFoundError, ValueError) as e:
        return problems + [f"load failed: {e}"]
    width = max(len(k) for k in summ)
    for k, v in summ.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        elif isinstance(v, list):
            v = "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
        print(f"  {k:<{width}}  {v}")
    if source == "raw" and not (paths.DATA_RAW / "SHA256SUMS.txt").exists():
        problems.append("no SHA256SUMS.txt: integrity cannot be checked")
    return problems


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=("raw", "synthetic", "all"), default="raw",
                    help="which data to check (default raw; all = raw + synthetic if present)")
    args = ap.parse_args()

    sources = ["raw", "synthetic"] if args.data == "all" else [args.data]
    if args.data == "all" and not (paths.DATA_SYNTHETIC / "train.csv").exists():
        print("(data/synthetic not present: checking raw only)")
        sources = ["raw"]

    failed = False
    for source in sources:
        problems = check(source)
        if problems:
            failed = True
            print(f"  FAILED: {len(problems)} problem(s)")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  OK: all checks passed")
    if failed:
        print("\nData problems found. Do not run experiments until they are fixed (see docs/SETUP.md).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
