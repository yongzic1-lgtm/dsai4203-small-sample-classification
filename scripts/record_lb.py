"""Record leaderboard results / upload time / final selection for a submission.

    .venv\\Scripts\\python.exe scripts\\record_lb.py --sub S003 --submitted-at now --public 0.8123
    .venv\\Scripts\\python.exe scripts\\record_lb.py --sub S003 --private 0.7954      # after the deadline
    .venv\\Scripts\\python.exe scripts\\record_lb.py --sub S003 --final yes

Fills fields of an existing row of registry/submissions.csv via
ssc.submission.record (rows are never deleted). --submitted-at takes "now" or
an ISO time; it is stored in UTC because the Kaggle day (5 uploads) is a UTC
day; a time in the future is refused. --notes is appended to the existing notes.

Private scores of many submissions after the deadline (PowerShell, one call each):
    $p = @{ S001 = 0.7954; S002 = 0.8012 }   # copied from Kaggle > My Submissions
    $p.GetEnumerator() | ForEach-Object { .venv\\Scripts\\python.exe scripts\\record_lb.py --sub $_.Key --private $_.Value }

When a public score is recorded, the "surprise" is printed: public minus the
pre-registered expected_public and minus the CV mean, also in units of the
public LB's standard error (Hanley-McNeil, ~395 rows), so a big-looking gap can
be judged against plain LB noise.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

import _common as C
from ssc import paths, submission
from ssc.submission import SUBMISSION_COLUMNS
from ssc.tracking import read_registry

FUTURE_TOLERANCE = timedelta(minutes=5)  # clock skew allowed for --submitted-at


def parse_when(text: str) -> str:
    if text.strip().lower() == "now":
        return submission.utc_now_iso()
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        C.die(f"--submitted-at {text!r} is neither 'now' nor an ISO time such as 2026-10-22T03:15:00Z", 2)
    if dt.tzinfo is None:
        print("note: --submitted-at has no time zone; taken as UTC")
        dt = dt.replace(tzinfo=timezone.utc)
    utc = dt.astimezone(timezone.utc)
    if utc > datetime.now(timezone.utc) + FUTURE_TOLERANCE:
        C.die(f"--submitted-at {text!r} = {utc:%Y-%m-%d %H:%M} UTC is in the future; check the date and "
              "time zone (JST = +09:00, HKT = +08:00)", 2)
    print(f"note: --submitted-at = {utc:%Y-%m-%d %H:%M} UTC = {utc.astimezone(timezone(timedelta(hours=9))):%H:%M} "
          f"JST, Kaggle (UTC) day {utc.date().isoformat()}")
    return utc.isoformat(timespec="seconds")


def public_se(auc: float, source: str) -> float | None:
    """SE of a public-LB AUC, using the train positive rate and the public fraction."""
    from ssc.validation import auc_standard_error  # lazy: sklearn/scipy are slow to import
    try:
        y = pd.read_csv(paths.data_dir(source) / "train.csv", usecols=["target"])["target"]
        n_test = len(pd.read_csv(paths.data_dir(source) / "sample_submission.csv", usecols=["id"]))
    except (FileNotFoundError, ValueError):
        return None
    n_pub = int(round(C.public_fraction() * n_test))
    n_pos = int(round(n_pub * float(y.mean())))
    return auc_standard_error(auc, n_pos, n_pub - n_pos)


def score(x: float | None, name: str) -> str | None:
    if x is None:
        return None
    if not 0.0 <= x <= 1.0:
        C.die(f"--{name} is an AUC and must be in [0, 1]", 2)
    return f"{x:.5f}"


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True, help="submission id, e.g. S003")
    ap.add_argument("--public", type=float, help="public leaderboard AUC")
    ap.add_argument("--private", type=float, help="private leaderboard AUC (revealed after the deadline)")
    ap.add_argument("--submitted-at", help="'now' or ISO time (stored as UTC)")
    ap.add_argument("--final", choices=("yes", "no"), help="selected as a final submission on Kaggle")
    ap.add_argument("--notes", help="text appended to the row's notes")
    C.add_data_arg(ap)
    args = ap.parse_args()
    source = paths.resolve_source(args.data)
    sub_id = args.sub.strip().upper()

    reg = read_registry(paths.submissions_csv(source))
    if reg.empty or sub_id not in set(reg["sub_id"]):
        C.die(f"{sub_id} not found in {paths.submissions_csv(source)}", 2)
    old = reg.loc[reg["sub_id"] == sub_id].iloc[0].to_dict()

    fields = {
        "public_lb": score(args.public, "public"),
        "private_lb": score(args.private, "private"),
        "submitted_at_utc": parse_when(args.submitted_at) if args.submitted_at else None,
        "selected_final": args.final,
    }
    if args.notes:
        fields["notes"] = f"{old['notes']} | {args.notes}" if old.get("notes") else args.notes
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        ap.error("nothing to record: give at least one of --public --private --submitted-at --final --notes")
    for k, v in fields.items():
        if old.get(k) and k != "notes" and old[k] != v:
            print(f"note: {k} changes from {old[k]!r} to {v!r}")

    row = submission.record(sub_id, source=source, **fields)
    C.print_archive(source)
    print(f"\nUpdated {sub_id} in {paths.submissions_csv(source)}:")
    width = max(len(c) for c in SUBMISSION_COLUMNS)
    for c in SUBMISSION_COLUMNS:
        print(f"  {c:<{width}}  {row.get(c, '')}")

    pub = C.to_float(row.get("public_lb"))
    if "public_lb" in fields and pub is not None:
        se = public_se(pub, source)
        print("\nSurprise (public LB minus what we believed):")
        for name, ref in (("expected_public", C.to_float(row.get("expected_public"))),
                          ("cv_auc_mean", C.to_float(row.get("cv_auc_mean")))):
            if ref is None:
                print(f"  vs {name:<15}: (not recorded)")
                continue
            z = f"  = {(pub - ref) / se:+.1f} x public-LB SE" if se else ""
            print(f"  vs {name:<15}: {pub - ref:+.4f}{z}")
        if se:
            print(f"  public-LB SE at AUC {pub:.3f} (Hanley-McNeil, ~{C.public_fraction():.0%} of test): {se:.4f}; "
                  "gaps within ~2 SE are ordinary LB noise.")
    priv = C.to_float(row.get("private_lb"))
    if "private_lb" in fields and priv is not None and pub is not None:
        print(f"\nprivate - public: {priv - pub:+.4f}")

    if "submitted_at_utc" in fields:
        used, limit = submission.submissions_today(source), C.daily_limit()
        print(f"\nuploads recorded for today's Kaggle day (UTC): {used} / {limit}")
        if used > limit:
            C.banner(f"More than {limit} uploads recorded today: check the timestamps.", "!")
    if args.final == "yes":
        n_final = int((read_registry(paths.submissions_csv(source))["selected_final"] == "yes").sum())
        print(f"\nsubmissions marked final: {n_final} (check the competition's limit on final selections)")
    return 0


if __name__ == "__main__":
    sys.exit(C.run_main(main))
