"""Create the mock dataset in data/synthetic/ (same layout as the Kaggle data).

Wraps ssc.synthetic.write: train.csv, test.csv, sample_submission.csv plus the
hidden test_labels.csv (labels + public mask) and truth.csv (informative
features). Use it to test scripts end-to-end and to simulate leaderboard noise;
results on synthetic data are NOT evidence about the real problem.

    .venv\\Scripts\\python.exe scripts\\make_synthetic.py                 # seed 2026, 10 informative
    .venv\\Scripts\\python.exe scripts\\make_synthetic.py --force --signal 0.7

The existing data/synthetic is never overwritten without --force, because other
people's sandbox runs (and tests) were computed on it. --out DIR writes a copy
somewhere else instead (nothing in data/ is touched).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import _common as C
from ssc import paths, synthetic

FILES = ("train", "test", "sample_submission", "test_labels", "truth")


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026, help="generator seed (default 2026, the team's default data)")
    ap.add_argument("--n-informative", type=int, default=10, help="number of informative features (default 10)")
    ap.add_argument("--signal", type=float, default=1.0, help="multiplier of the true logit (default 1.0)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data/synthetic")
    ap.add_argument("--out", type=Path, default=None,
                    help="write to this folder instead of data/synthetic (for experiments with other mock data)")
    args = ap.parse_args()
    if not 1 <= args.n_informative <= 150:
        C.die("--n-informative must be between 1 and 150", 2)

    out = args.out or paths.DATA_SYNTHETIC
    existing = [f"{n}.csv" for n in FILES if (out / f"{n}.csv").exists()]
    if existing and not args.force:
        C.die(f"{out} already contains {', '.join(existing)}.\n"
              "Refusing to overwrite: sandbox runs and tests were made on this data. "
              "Pass --force if you really want to regenerate it (or --out DIR for a separate copy).", 2)
    if existing and out == paths.DATA_SYNTHETIC:
        C.banner(["Overwriting data/synthetic: earlier sandbox runs used different data",
                  "and are no longer comparable with new ones (data_fingerprint changes)."])

    synthetic.write(seed=args.seed, out=out, n_informative=args.n_informative, signal=args.signal)

    train = pd.read_csv(out / "train.csv", usecols=["id", "target"])
    labels = pd.read_csv(out / "test_labels.csv")
    truth = pd.read_csv(out / "truth.csv")
    print(f"wrote {len(FILES)} files to {out}")
    print(f"  seed={args.seed}  n_informative={args.n_informative}  signal={args.signal}")
    print(f"  train: {len(train)} rows, positive rate {train['target'].mean():.3f}")
    print(f"  test:  {len(labels)} rows, public rows {int(labels['public'].sum())}, "
          f"positive rate {labels['target'].mean():.3f}")
    print(f"  informative features: {', '.join(truth['feature'].astype(str))}")
    if out == paths.DATA_SYNTHETIC:
        print("Check it with: " + C.hint("verify_data.py", "--data synthetic"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
