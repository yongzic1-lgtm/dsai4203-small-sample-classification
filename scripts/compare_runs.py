"""Paired comparison of archived runs evaluated on identical CV splits.

    .venv\\Scripts\\python.exe scripts\\compare_runs.py RUN_A RUN_B
    .venv\\Scripts\\python.exe scripts\\compare_runs.py --ref RUN_REF RUN_1 RUN_2 RUN_3

Reads each run's fold_scores.csv and calls ssc.validation.paired_compare:
mean AUC difference, its spread across repeats, the split SE of the mean
difference (std across repeats / sqrt(n_repeats)), the fraction of folds won,
and the Nadeau-Bengio corrected resampled t-test. Run ids may be abbreviated to
a unique prefix.

Two views of noise are printed. "2 x split SE": is the sign of the difference
an artefact of the particular split seeds? (It ignores that the 250 rows are
themselves a sample.) Corrected p: the conservative data-sampling view, roughly
"would it hold on new data?". Only a difference that passes both is a win.

A paired comparison subtracts the two AUCs fold by fold, so it is valid only
when both runs were scored on exactly the same validation rows, i.e. their
split signatures match (same scheme, same seed, same data). The script refuses
otherwise. Use the official CV for every run you want to compare.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from tabulate import tabulate

import _common as C
from ssc import paths
from ssc.models import describe
from ssc.tracking import load_run
from ssc.validation import Scheme, paired_compare

GUIDE = ("Guide: diff > 0 means the first run (A / the row) is better. split SE = std_diff_repeat / "
         "sqrt(n_repeats) is the split noise of the reported mean difference: |diff| < 2 x split SE means "
         "other split seeds of the same 250 rows could flip the sign. The corrected p is the data-sampling "
         "view (new rows, not new splits) and is far more conservative: claim an improvement only when "
         "|diff| > 2 x split SE AND corrected p < 0.05. A fold win rate near 0.5 is a coin flip. A real CV "
         "difference of ~0.01 is usually invisible on the ~395-row public LB (scripts/lb_noise.py).")


def load(run_id: str, source: str) -> dict:
    run = load_run(C.resolve_run_id(run_id, source), source)
    C.check_run_source(run, source)
    if "fold_scores" not in run or "summary" not in run:
        C.die(f"run {run['meta']['run_id']} has no fold_scores.csv/summary.json (failed or unfinished run?)", 2)
    return run


def label(run: dict) -> str:
    s = run["summary"]
    sch = Scheme.from_dict(s.get("scheme") or {})
    return f"{sch.label()} seed {sch.seed}, sig {s.get('split_signature')}"


def mismatch(a: dict, b: dict) -> str | None:
    """Why a and b cannot be paired, or None if they can."""
    fa, fb = a["meta"].get("data_fingerprint"), b["meta"].get("data_fingerprint")
    if fa and fb and fa != fb:
        return "the runs used different data files (data_fingerprint differs)"
    sa, sb = a["summary"].get("split_signature"), b["summary"].get("split_signature")
    if sa != sb:
        return f"split signatures differ ({label(a)} vs {label(b)})"
    return None


def explain_refusal(reason: str) -> None:
    sys.stdout.flush()
    print(f"REFUSED: {reason}.", file=sys.stderr)
    print("A paired comparison subtracts the two AUCs fold by fold. That cancels the split-to-split\n"
          "noise only when both models were validated on exactly the same rows. With different\n"
          "splits that noise (often 0.01-0.03 AUC with 250 rows) stays in the difference and would\n"
          "be mistaken for a model effect. Re-run both configs on the official CV (no cv block,\n"
          "no cv.* overrides) and compare those runs.", file=sys.stderr)


def verdict(r: dict) -> str:
    """Split-noise verdict: is |mean diff| larger than 2 split SEs of the mean difference?"""
    d, se = r["mean_diff"], r["se_split"]
    if d == 0 and r["std_diff_fold"] == 0:
        return "identical"
    if se != se:  # NaN: a single repeat gives no spread
        return "n/a (1 repeat)"
    return "within split noise" if abs(d) < 2 * se else ("higher" if d > 0 else "lower")


def model_text(run: dict, width: int = 36) -> str:
    pipe = run["config"].get("pipeline") or {}
    if not pipe.get("model"):
        return "?"
    d = describe(pipe)
    text = d["model"] if d["selector"] == "none" else f"{d['selector']}+{d['model']}"
    return text if len(text) <= width else text[: width - 3] + "..."


def describe_run(tag: str, run: dict) -> str:
    s = run["summary"]
    model = model_text(run, 60)
    return (f"{tag}: {run['meta']['run_id']}  [{model}]  auc_mean {C.fnum(s.get('auc_mean'))} "
            f"+/- {C.fnum(s.get('auc_std_repeat'))}  ({label(run)})")


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="RUN_A RUN_B, or the runs to compare with --ref")
    ap.add_argument("--ref", help="reference run: every positional run is compared with it")
    C.add_data_arg(ap)
    args = ap.parse_args()
    source = paths.resolve_source(args.data)

    if args.ref is None:
        if len(args.runs) != 2:
            ap.error("give exactly two runs (RUN_A RUN_B), or --ref RUN with one or more runs")
        a, b = load(args.runs[0], source), load(args.runs[1], source)
        print(describe_run("A", a))
        print(describe_run("B", b))
        why = mismatch(a, b)
        if why:
            explain_refusal(why)
            return 2
        r = paired_compare(a["fold_scores"], b["fold_scores"])
        print(f"\nPaired comparison on {r['n_folds']} identical folds:")
        rows = [
            ("mean diff (A - B)", C.fnum(r["mean_diff"], signed=True)),
            (f"split SE of the mean diff ({r['n_repeats']} repeats)", C.fnum(r["se_split"])),
            ("std of diff across repeats (one repeat)", C.fnum(r["std_diff_repeat"])),
            ("std of diff across folds (one fold)", C.fnum(r["std_diff_fold"])),
            ("fold win rate (A > B, ties 1/2)", C.fnum(r["win_rate_fold"], 2)),
            ("corrected t (Nadeau-Bengio)", C.fnum(r["corrected_t"], 2)),
            ("corrected p (two-sided)", C.fnum(r["corrected_p"], 3)),
            ("2 x split SE rule", verdict(r)),
        ]
        print(tabulate(rows, tablefmt="plain", disable_numparse=True))
        print("\n" + textwrap.fill(GUIDE, 100))
        return 0

    if not args.runs:
        ap.error("--ref needs at least one run to compare with it")
    ref = load(args.ref, source)
    print(describe_run("ref", ref))
    table, skipped = [], 0
    for rid in args.runs:
        run = load(rid, source)
        s = run["summary"]
        why = mismatch(run, ref)
        model = model_text(run)
        if why:
            skipped += 1
            table.append([run["meta"]["run_id"], model, C.fnum(s.get("auc_mean")),
                          "", "", "", "", "", "SKIPPED: " + why.split(" (")[0]])
            continue
        r = paired_compare(run["fold_scores"], ref["fold_scores"])
        table.append([run["meta"]["run_id"], model, C.fnum(s.get("auc_mean")),
                      C.fnum(r["mean_diff"], signed=True), C.fnum(r["se_split"]),
                      C.fnum(r["win_rate_fold"], 2), C.fnum(r["corrected_t"], 2),
                      C.fnum(r["corrected_p"], 3), verdict(r)])
    print()
    print(tabulate(table, headers=["run", "model", "auc_mean", "diff_vs_ref", "split_se",
                                   "win_rate", "t_corr", "p_corr", "2 x split SE"], tablefmt="simple",
                   disable_numparse=True))
    if skipped:
        print(f"\n{skipped} run(s) skipped: their split signature (or data) differs from the reference, "
              "so a paired comparison is invalid. Re-run them on the official CV.")
    print("\n" + textwrap.fill(GUIDE, 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
