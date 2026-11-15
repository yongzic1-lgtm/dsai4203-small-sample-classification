"""Print the experiment registry (registry/experiments.csv) as a table.

    .venv\\Scripts\\python.exe scripts\\list_runs.py                      # all runs, best auc_mean first
    .venv\\Scripts\\python.exe scripts\\list_runs.py --official-only --top 10
    .venv\\Scripts\\python.exe scripts\\list_runs.py --author AB --kind experiment --data synthetic

Only rows on the official CV (project.yaml official_cv) can be ranked against
each other; the "off" column marks them. A higher auc_mean on a different
scheme or seed is not evidence that a run is better.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from tabulate import tabulate

import _common as C
from ssc import paths
from ssc.tracking import read_registry


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_data_arg(ap)
    ap.add_argument("--author", help="only runs by this member id")
    ap.add_argument("--kind", help="only runs of this kind (experiment, or an extension's kind)")
    ap.add_argument("--official-only", action="store_true", help="only runs on the official CV scheme and seed")
    ap.add_argument("--top", type=int, default=None, metavar="N", help="show only the N best rows")
    ap.add_argument("--wide", action="store_true", help="also show selector, scaler, split signature and hypothesis")
    args = ap.parse_args()
    source = paths.resolve_source(args.data)

    path = paths.experiments_csv(source)
    reg = read_registry(path)
    C.print_archive(source)
    if reg.empty:
        print(f"no runs yet in {path}")
        return 0
    # $SSC_ARCHIVE_ROOT may hold both sources in one registry: show only the requested one
    reg = reg[reg["data_source"] == source]
    total = len(reg)

    off = C.official_scheme()
    off_label, off_seed = off.label(), str(off.seed)
    reg = reg.assign(off=((reg["cv"] == off_label) & (reg["cv_seed"] == off_seed)).map({True: "yes", False: "-"}))
    if args.author:
        reg = reg[reg["author"] == args.author]
    if args.kind:
        reg = reg[reg["kind"] == args.kind]
    if args.official_only:
        reg = reg[reg["off"] == "yes"]

    reg = reg.assign(_auc=pd.to_numeric(reg["auc_mean"], errors="coerce"))
    reg = reg.sort_values("_auc", ascending=False, na_position="last", kind="stable")
    if args.top is not None:
        reg = reg.head(args.top)

    cols = ["run_id", "author", "kind", "model", "cv", "off", "auc_mean", "auc_std_repeat", "oof_auc_mean", "status"]
    if args.wide:
        cols[4:4] = ["selector", "scaler"]
        cols += ["split_sig", "hypothesis"]
    view = reg[cols].copy()
    for c in ("auc_mean", "auc_std_repeat", "oof_auc_mean"):
        view[c] = view[c].map(C.fnum)
    if not args.wide:  # compact: "selector+model" in one column
        sel = reg["selector"].where(~reg["selector"].isin(["", "none"]), "")
        view["model"] = [f"{s}+{m}" if s else m for s, m in zip(sel, view["model"])]
        view["model"] = view["model"].map(lambda s: s if len(s) <= 48 else s[:45] + "...")
    else:
        view["hypothesis"] = view["hypothesis"].map(lambda s: s if len(s) <= 60 else s[:57] + "...")
    print(f"{len(view)} of {total} runs shown (sorted by auc_mean; official CV = {off_label} seed {off_seed})")
    if len(view):
        print(tabulate(view.to_numpy().tolist(), headers=cols, tablefmt="simple", disable_numparse=True))
    if not args.official_only and (view["off"] != "yes").any():
        print("\nRows with off = '-' use another CV scheme/seed: do not rank them against official runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
